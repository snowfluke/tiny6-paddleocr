// Executes a graph with every activation living in WASM linear memory.
//
// The TypeScript path in graph.ts allocates a Float32Array per node and copies
// in and out of WASM for each convolution. Here a tensor is a pointer, ops read
// and write in place, and nothing crosses the boundary until the graph output.
// Buffers are recycled as soon as their last reader has run.

import type { OnnxNode } from "../onnx/parse.ts";
import type { Arena } from "../wasm/backend.ts";
import { numel, type Tensor } from "./tensor.ts";

/** A tensor that lives in the arena. `ptr` is a byte offset, `len` a float count. */
export type RT = { dims: number[]; ptr: number; len: number };

export const BIN_OP = { add: 0, sub: 1, mul: 2, div: 3 } as const;
export const UN_OP = { relu: 0, sigmoid: 1, erf: 2, hardSigmoid: 3, gelu: 4 } as const;

export class Resident {
  constructor(readonly ar: Arena) {}

  alloc(dims: number[]): RT {
    const len = numel(dims);
    return { dims, ptr: this.ar.allocScratch(len), len };
  }

  /** A constant already uploaded below the scratch mark; never freed. */
  constant(data: Float32Array, dims: number[]): RT {
    return { dims, ptr: this.ar.persist(data), len: data.length };
  }

  upload(t: Tensor): RT {
    const r = this.alloc(t.dims);
    this.ar.write(r.ptr, t.data);
    return r;
  }

  download(r: RT): Tensor {
    const data = new Float32Array(r.len);
    this.ar.readInto(r.ptr, data);
    return { dims: r.dims, data };
  }

  /** Same buffer, different shape. The caller retains it like any other. */
  view(r: RT, dims: number[]): RT {
    return { dims, ptr: r.ptr, len: r.len };
  }

  binary(op: number, a: RT, b: RT, dims: number[]): RT {
    const out = this.alloc(dims);
    const n = out.len;
    if (a.len === n && b.len === n) {
      this.ar.pBinarySame(op, n, a.ptr, b.ptr, out.ptr);
    } else if (b.len === 1) {
      this.ar.k.binary(op, 1, n, 0, 0, a.ptr, b.ptr, out.ptr);
    } else if (a.len === 1) {
      // Rewritten as a per-element scalar on the left, which sub and div need.
      this.ar.k.binary(op, 3, n, n, 1, a.ptr, b.ptr, out.ptr);
    } else if (dims.length === 4 && a.len === n && b.len === dims[1]) {
      this.ar.k.binary(op, 2, n, n / (dims[0] * dims[1]), dims[1], a.ptr, b.ptr, out.ptr);
    } else if (dims.length === 4 && b.len === n && a.len === dims[1]) {
      this.ar.k.binary(op, 3, n, n / (dims[0] * dims[1]), dims[1], a.ptr, b.ptr, out.ptr);
    } else if (a.len === n && b.len === dims[dims.length - 1]) {
      // [1, R, C] + [C]: the trailing vector repeats over every row.
      this.ar.k.binary(op, 4, n, b.len, 0, a.ptr, b.ptr, out.ptr);
    } else {
      this.ar.release(out.ptr);
      return null as unknown as RT;
    }
    return out;
  }

  unary(op: number, a: RT, p0 = 0, p1 = 0): RT {
    const out = this.alloc(a.dims.slice());
    this.ar.pUnary(op, a.len, a.ptr, out.ptr, p0, p1);
    return out;
  }

  /** Mean over a contiguous trailing run; returns null if the axes are not trailing. */
  reduceMean(a: RT, axes: number[], keepdims: boolean): RT | null {
    const rank = a.dims.length;
    const norm = axes.map((x) => (x < 0 ? x + rank : x)).sort((p, q) => p - q);
    if (!norm.length || !norm.every((x, i) => x === rank - norm.length + i)) return null;
    const inner = a.dims.slice(rank - norm.length).reduce((p, q) => p * q, 1);
    const outer = a.len / inner;
    const dims = keepdims
      ? a.dims.map((d, i) => (norm.includes(i) ? 1 : d))
      : a.dims.slice(0, rank - norm.length);
    const out = this.alloc(dims.length ? dims : [1]);
    this.ar.k.reduce_mean(outer, inner, a.ptr, out.ptr);
    return out;
  }

  /**
   * Concatenation is a move inside the arena: for every slice along the axes
   * ahead of `axis`, each input's block lands end to end. The TypeScript
   * fallback instead downloads and re-uploads every input, which at 960x960
   * was 5.5% of detection.
   */
  concat(parts: RT[], axis: number, dims: number[]): RT {
    const out = this.alloc(dims);
    const outer = dims.slice(0, axis).reduce((a, b) => a * b, 1);
    const stride = out.len / outer;
    let off = 0;
    for (const p of parts) {
      const block = p.len / outer;
      for (let o = 0; o < outer; o++) {
        this.ar.move(out.ptr + (o * stride + off) * 4, p.ptr + o * block * 4, block);
      }
      off += block;
    }
    return out;
  }

  /**
   * Batch normalisation, folded to one pass. scale and shift are derived from
   * the four constants on every call, which is a few hundred flops against a
   * tensor of tens of thousands, and keeps them out of the sealed weight area.
   */
  batchNorm(a: RT, gamma: RT, beta: RT, mean: RT, variance: RT, eps: number): RT {
    const c = gamma.len;
    const axis = a.dims.length >= 2 ? 1 : 0;
    const inner = a.dims.slice(axis + 1).reduce((x, y) => x * y, 1);
    const g = this.download(gamma).data;
    const bt = this.download(beta).data;
    const m = this.download(mean).data;
    const v = this.download(variance).data;
    const s = new Float32Array(c);
    const t = new Float32Array(c);
    for (let i = 0; i < c; i++) {
      s[i] = g[i] / Math.sqrt(v[i] + eps);
      t[i] = bt[i] - m[i] * s[i];
    }
    const sRT = this.alloc([c]);
    const tRT = this.alloc([c]);
    this.ar.write(sRT.ptr, s);
    this.ar.write(tRT.ptr, t);
    const out = this.alloc(a.dims.slice());
    this.ar.pAffineChannels(a.len, inner, c, a.ptr, sRT.ptr, tRT.ptr, out.ptr);
    this.ar.release(sRT.ptr);
    this.ar.release(tRT.ptr);
    return out;
  }

  /** Softmax over the trailing axis, which is the only axis either model uses. */
  softmaxLast(a: RT): RT {
    const cols = a.dims[a.dims.length - 1];
    const out = this.alloc(a.dims.slice());
    this.ar.pSoftmaxRows(a.len / cols, cols, a.ptr, out.ptr);
    return out;
  }

  /** Rank-4 and below; the permutation is left-padded with identity axes. */
  transpose(a: RT, perm: number[]): RT {
    const rank = a.dims.length;
    if (rank > 4) return null as unknown as RT;
    const pad = 4 - rank;
    const dims = [...Array(pad).fill(1), ...a.dims];
    const p = [...Array(pad).fill(0).map((_, i) => i), ...perm.map((v) => v + pad)];
    const stride = [0, 0, 0, 1];
    for (let i = 2; i >= 0; i--) stride[i] = stride[i + 1] * dims[i + 1];
    const outDims = p.map((i) => dims[i]);
    const out = this.alloc(perm.map((i) => a.dims[i]));
    this.ar.k.transpose4(
      outDims[0], outDims[1], outDims[2], outDims[3],
      stride[p[0]], stride[p[1]], stride[p[2]], stride[p[3]],
      a.ptr, out.ptr,
    );
    return out;
  }

  maxPool2x2(a: RT): RT {
    const [n, c, h, w] = a.dims;
    const out = this.alloc([n, c, h, w]);
    this.ar.k.maxpool2x2(n * c, h, w, a.ptr, out.ptr, 0, n * c);
    return out;
  }

  resizeNearest(a: RT, sh: number, sw: number): RT {
    const [n, c, h, w] = a.dims;
    const out = this.alloc([n, c, h * sh, w * sw]);
    this.ar.k.resize_nearest(n * c, h, w, sh, sw, a.ptr, out.ptr, 0, n * c);
    return out;
  }
}

/**
 * Index of the last node that reads each tensor, so its buffer can go back on
 * the free list the moment that node returns. Graph outputs never expire.
 */
export function lastUseMap(nodes: OnnxNode[], outputs: string[]): Map<string, number> {
  const last = new Map<string, number>();
  nodes.forEach((n, i) => {
    for (const name of n.input) if (name) last.set(name, i);
  });
  for (const o of outputs) last.set(o, Number.MAX_SAFE_INTEGER);
  return last;
}
