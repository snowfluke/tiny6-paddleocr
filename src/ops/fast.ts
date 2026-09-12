// Hot elementwise paths. The generic closure-per-element versions in
// tensor.ts cost more than the arithmetic once tensors reach a few million
// elements, which is where the detection model lives at full resolution.

import { binary, broadcastShape, make, type Tensor } from "../runtime/tensor.ts";
import { erf } from "./elementwise.ts";

export type BinOp = "add" | "sub" | "mul" | "div";

/**
 * Covers same-shape, scalar, and per-channel [1,C,1,1] operands, which is
 * every binary node in both models. Anything else defers to the generic path.
 */
export function binaryFast(a: Tensor, b: Tensor, op: BinOp): Tensor {
  const dims = broadcastShape(a.dims, b.dims);
  const out = make(dims);
  const n = out.data.length;
  const A = a.data;
  const B = b.data;
  const O = out.data;

  if (A.length === n && B.length === n) {
    switch (op) {
      case "add": for (let i = 0; i < n; i++) O[i] = A[i] + B[i]; return out;
      case "sub": for (let i = 0; i < n; i++) O[i] = A[i] - B[i]; return out;
      case "mul": for (let i = 0; i < n; i++) O[i] = A[i] * B[i]; return out;
      case "div": for (let i = 0; i < n; i++) O[i] = A[i] / B[i]; return out;
    }
  }

  if (B.length === 1 && A.length === n) {
    const y = B[0];
    switch (op) {
      case "add": for (let i = 0; i < n; i++) O[i] = A[i] + y; return out;
      case "sub": for (let i = 0; i < n; i++) O[i] = A[i] - y; return out;
      case "mul": for (let i = 0; i < n; i++) O[i] = A[i] * y; return out;
      case "div": for (let i = 0; i < n; i++) O[i] = A[i] / y; return out;
    }
  }
  if (A.length === 1 && B.length === n) {
    const x = A[0];
    switch (op) {
      case "add": for (let i = 0; i < n; i++) O[i] = x + B[i]; return out;
      case "sub": for (let i = 0; i < n; i++) O[i] = x - B[i]; return out;
      case "mul": for (let i = 0; i < n; i++) O[i] = x * B[i]; return out;
      case "div": for (let i = 0; i < n; i++) O[i] = x / B[i]; return out;
    }
  }

  // Per-channel: [N,C,H,W] against [1,C,1,1] or [C,1,1], either way round.
  const perChannel = (big: Float32Array, small: Float32Array, dims4: number[], flip: boolean) => {
    const C = dims4[1];
    const inner = n / (dims4[0] * C);
    let i = 0;
    for (let nb = 0; nb < dims4[0]; nb++) {
      for (let c = 0; c < C; c++) {
        const v = small[c];
        const end = i + inner;
        switch (op) {
          case "add": for (; i < end; i++) O[i] = big[i] + v; break;
          case "sub": for (; i < end; i++) O[i] = flip ? v - big[i] : big[i] - v; break;
          case "mul": for (; i < end; i++) O[i] = big[i] * v; break;
          case "div": for (; i < end; i++) O[i] = flip ? v / big[i] : big[i] / v; break;
        }
      }
    }
  };

  const chanCount = (t: Tensor) =>
    t.dims.length >= 2 && t.data.length === t.dims[t.dims.length - 3] ? t.data.length : -1;

  if (dims.length === 4 && A.length === n && chanCount(b) === dims[1]) {
    perChannel(A, B, dims, false);
    return out;
  }
  if (dims.length === 4 && B.length === n && chanCount(a) === dims[1]) {
    perChannel(B, A, dims, true);
    return out;
  }

  switch (op) {
    case "add": return binary(a, b, (p, q) => p + q);
    case "sub": return binary(a, b, (p, q) => p - q);
    case "mul": return binary(a, b, (p, q) => p * q);
    case "div": return binary(a, b, (p, q) => p / q);
  }
}

export function reluFast(a: Tensor): Tensor {
  const out = make(a.dims.slice());
  const A = a.data;
  const O = out.data;
  for (let i = 0; i < A.length; i++) O[i] = A[i] > 0 ? A[i] : 0;
  return out;
}

export function sigmoidFast(a: Tensor): Tensor {
  const out = make(a.dims.slice());
  const A = a.data;
  const O = out.data;
  for (let i = 0; i < A.length; i++) O[i] = 1 / (1 + Math.exp(-A[i]));
  return out;
}

export function erfFast(a: Tensor): Tensor {
  const out = make(a.dims.slice());
  const A = a.data;
  const O = out.data;
  for (let i = 0; i < A.length; i++) O[i] = erf(A[i]);
  return out;
}

/** The reference for the fused Gelu: post * x * (1 + erf(x * scale)). */
export function geluFast(a: Tensor, scale: number, post: number): Tensor {
  const out = make(a.dims.slice());
  const A = a.data;
  const O = out.data;
  for (let i = 0; i < A.length; i++) O[i] = post * A[i] * (1 + erf(A[i] * scale));
  return out;
}

export function hardSigmoidFast(a: Tensor, alpha: number, beta: number): Tensor {
  const out = make(a.dims.slice());
  const A = a.data;
  const O = out.data;
  for (let i = 0; i < A.length; i++) {
    const v = alpha * A[i] + beta;
    O[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

/**
 * Nearest-neighbour upsample by an integer factor on H and W only, which is
 * every Resize in the detection model. Copies whole rows instead of walking
 * strides per element.
 */
export function resizeNearestFast(x: Tensor, scales: number[]): Tensor {
  const [N, C, H, W] = x.dims;
  const sh = scales[2];
  const sw = scales[3];
  if (x.dims.length !== 4 || scales[0] !== 1 || scales[1] !== 1 || sh !== Math.floor(sh) || sw !== Math.floor(sw)) {
    return null as unknown as Tensor;
  }
  const OH = H * sh;
  const OW = W * sw;
  const out = make([N, C, OH, OW]);
  const row = new Float32Array(OW);
  for (let p = 0; p < N * C; p++) {
    const src = p * H * W;
    const dst = p * OH * OW;
    for (let y = 0; y < H; y++) {
      for (let i = 0, o = 0; i < W; i++) {
        const v = x.data[src + y * W + i];
        for (let k = 0; k < sw; k++) row[o++] = v;
      }
      for (let k = 0; k < sh; k++) out.data.set(row, dst + (y * sh + k) * OW);
    }
  }
  return out;
}

/**
 * Mean over the trailing contiguous axes, which is every ReduceMean in both
 * models (`axes=[2,3]`, a spatial global average). Sums a flat run per
 * channel instead of recomputing strides per element.
 */
export function reduceMeanTrailing(x: Tensor, axes: number[], keepdims: boolean): Tensor | null {
  const rank = x.dims.length;
  const norm = axes.map((a) => (a < 0 ? a + rank : a)).sort((a, b) => a - b);
  const isTrailing = norm.every((a, i) => a === rank - norm.length + i);
  if (!isTrailing || !norm.length) return null;

  const inner = x.dims.slice(rank - norm.length).reduce((a, b) => a * b, 1);
  const outer = x.data.length / inner;
  const dims = keepdims
    ? x.dims.map((d, i) => (norm.includes(i) ? 1 : d))
    : x.dims.slice(0, rank - norm.length);
  const out = make(dims.length ? dims : [1]);
  for (let o = 0; o < outer; o++) {
    let s = 0;
    const base = o * inner;
    for (let i = 0; i < inner; i++) s += x.data[base + i];
    out.data[o] = s / inner;
  }
  return out;
}

/**
 * 2x2 max pool, stride 1, SAME_UPPER padding, which is the detection model's
 * only pooling node. The interior needs no bounds checks; only the last row
 * and column read the padding.
 */
export function maxPool2x2Same(x: Tensor): Tensor | null {
  const [N, C, H, W] = x.dims;
  if (x.dims.length !== 4) return null;
  const out = make([N, C, H, W]);
  for (let p = 0; p < N * C; p++) {
    const base = p * H * W;
    for (let y = 0; y < H; y++) {
      const r0 = base + y * W;
      const r1 = y + 1 < H ? r0 + W : r0;
      for (let xi = 0; xi < W - 1; xi++) {
        const a = x.data[r0 + xi];
        const b = x.data[r0 + xi + 1];
        const c = x.data[r1 + xi];
        const d = x.data[r1 + xi + 1];
        const m1 = a > b ? a : b;
        const m2 = c > d ? c : d;
        out.data[r0 + xi] = m1 > m2 ? m1 : m2;
      }
      const a = x.data[r0 + W - 1];
      const c = x.data[r1 + W - 1];
      out.data[r0 + W - 1] = a > c ? a : c;
    }
  }
  return out;
}
