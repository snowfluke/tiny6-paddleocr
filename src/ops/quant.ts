import { make, type Tensor } from "../runtime/tensor.ts";

/**
 * QuantizeLinear / DequantizeLinear with a per-tensor or per-axis scale and an
 * int8 zero point. The quantized tensor keeps its values in a Float32Array,
 * which is exact for int8, so the pair runs in fp32 as fake quantization:
 * the error an int8 kernel would introduce, without the kernel. Reference
 * path only; the resident runtime falls back to these.
 */
export function quantizeLinear(x: Tensor, scale: Tensor, zeroPoint: Tensor | null, axis = 1): Tensor {
  const out = make(x.dims);
  const ch = channelOf(x.dims, scale.data.length, axis);
  for (let i = 0; i < x.data.length; i++) {
    const c = ch(i);
    const q = roundHalfEven(x.data[i] / scale.data[c]) + (zeroPoint ? zeroPoint.data[c] : 0);
    out.data[i] = q < -128 ? -128 : q > 127 ? 127 : q;
  }
  return out;
}

export function dequantizeLinear(q: Tensor, scale: Tensor, zeroPoint: Tensor | null, axis = 1): Tensor {
  const out = make(q.dims);
  const ch = channelOf(q.dims, scale.data.length, axis);
  for (let i = 0; i < q.data.length; i++) {
    const c = ch(i);
    out.data[i] = (q.data[i] - (zeroPoint ? zeroPoint.data[c] : 0)) * scale.data[c];
  }
  return out;
}

/** ONNX rounds ties to even, as f32x4.nearest does. Math.round rounds them up. */
export function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Flat index -> scale index. One scale means per-tensor. */
function channelOf(dims: number[], scales: number, axis: number): (i: number) => number {
  if (scales === 1) return () => 0;
  if (dims[axis] !== scales) throw new Error(`${scales} scales for axis ${axis} of [${dims}]`);
  const inner = dims.slice(axis + 1).reduce((a, b) => a * b, 1);
  return (i) => Math.floor(i / inner) % scales;
}

/**
 * Weights for the int8 GEMM: [K][N] row-major int8 packed as [K/4][N][4], so
 * one 16-byte load holds four K values for four output channels. K must be a
 * multiple of 4 and N of 8; the planner pads before packing.
 */
export function packWeights(K: number, N: number, w: Int8Array): Int8Array {
  if (K % 4 || N % 8 || w.length !== K * N) throw new Error(`packWeights: K=${K} N=${N} len=${w.length}`);
  const out = new Int8Array(K * N);
  for (let kb = 0; kb < K; kb += 4) {
    for (let n = 0; n < N; n++) {
      for (let t = 0; t < 4; t++) out[kb * N + n * 4 + t] = w[(kb + t) * N + n];
    }
  }
  return out;
}

export type QGemmEpilogue = {
  sw: Float32Array;
  bias: Float32Array;
  comp: Int32Array;
  act: number;
  p0?: number;
  p1?: number;
  res?: { q: Int8Array; scale: Float32Array; zp: Int32Array };
  out?: { inv: Float32Array; zp: Int32Array };
};

const f = Math.fround;

/**
 * The int8 GEMM in plain arithmetic: exact i32 sums, then the epilogue in
 * fp32 in the kernel's order. Sums of int8 products fit a double exactly, so
 * the kernel must match this bit for bit except through GELU, whose erf
 * polynomial only the kernel has.
 */
export function qgemmReference(
  M: number, K: number, N: number, a: Int8Array, w: Int8Array, e: QGemmEpilogue,
): Float32Array | Int8Array {
  const y = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += a[m * K + k] * w[k * N + n];
      let v = f(f(f(acc + e.comp[n]) * e.sw[n]) + e.bias[n]);
      if (e.act === 1) v = Math.max(v, 0);
      else if (e.act === 2) v = f(f(v * e.p1!) * f(1 + erf(f(v * e.p0!))));
      if (e.res) v = f(v + f((e.res.q[m * N + n] - e.res.zp[n]) * e.res.scale[n]));
      y[m * N + n] = v;
    }
  }
  if (!e.out) return y;
  const q = new Int8Array(M * N);
  for (let i = 0; i < y.length; i++) {
    const n = i % N;
    const v = roundHalfEven(f(y[i] * e.out.inv[n])) + e.out.zp[n];
    q[i] = v < -128 ? -128 : v > 127 ? 127 : v;
  }
  return q;
}

function erf(x: number): number {
  const s = Math.sign(x), a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}

/**
 * Depthwise int8 in plain arithmetic, the twin of qgemmReference. x is NHWC
 * int8, w is [tap][C]; taps outside the input are skipped.
 */
export function qdepthwiseReference(
  C: number, H: number, W: number, kh: number, kw: number, sy: number, sx: number, pt: number, pl: number,
  x: Int8Array, xzp: Int32Array, w: Int8Array, sw: Float32Array, bias: Float32Array, act: number,
  out?: { inv: Float32Array; zp: Int32Array },
): Float32Array | Int8Array {
  const OH = Math.floor((H + 2 * pt - kh) / sy) + 1, OW = Math.floor((W + 2 * pl - kw) / sx) + 1;
  const y = new Float32Array(OH * OW * C);
  for (let oy = 0; oy < OH; oy++) {
    for (let ox = 0; ox < OW; ox++) {
      for (let c = 0; c < C; c++) {
        let acc = 0;
        for (let ky = 0; ky < kh; ky++) {
          const iy = oy * sy + ky - pt;
          if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < kw; kx++) {
            const ix = ox * sx + kx - pl;
            if (ix < 0 || ix >= W) continue;
            acc += (x[(iy * W + ix) * C + c] - xzp[c]) * w[(ky * kw + kx) * C + c];
          }
        }
        let v = f(f(acc * sw[c]) + bias[c]);
        if (act === 1) v = Math.max(v, 0);
        y[(oy * OW + ox) * C + c] = v;
      }
    }
  }
  if (!out) return y;
  const q = new Int8Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const c = i % C;
    const v = roundHalfEven(f(y[i] * out.inv[c])) + out.zp[c];
    q[i] = v < -128 ? -128 : v > 127 ? 127 : v;
  }
  return q;
}
