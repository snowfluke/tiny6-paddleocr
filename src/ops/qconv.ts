// Convolutions on int8 activations. Tensors in the int8 region are NHWC with
// a per-channel scale and zero point; the weights are folded with the input
// scales offline and packed for the relaxed dot product (see kernels.rs).
import type { Resident, RT } from "../runtime/resident.ts";
import type { Tensor } from "../runtime/tensor.ts";
import { packWeights, roundHalfEven } from "./quant.ts";

export const ACT_GELU = 2;

/**
 * Per-channel quantization of one tensor, with its vectors already in the
 * arena: the kernels read scale, 1/scale and zero point by pointer, and
 * nothing may upload after the arena seals, so the planner makes these once.
 */
export type QParams = { scale: Float32Array; zp: Int32Array; ptr: { scale: number; inv: number; zp: number } };

export function uploadQParams(r: Resident, scale: Float32Array, zp: Int32Array): QParams {
  const inv = Float32Array.from(scale, (s) => 1 / s);
  return { scale, zp, ptr: { scale: r.ar.persist(scale), inv: r.ar.persist(inv), zp: r.ar.persistBytes(zp) } };
}

/** An int8 NHWC activation in the arena: `dims` stay NCHW for the graph's sake. */
export type QT = { dims: number[]; ptr: number; bytes: number; q: QParams };

/** Everything a Conv needs after its weights have been folded and quantized. */
export type QConvWeights = {
  K: number;
  N: number;
  packed: Int8Array;
  /** Per output channel: dequant scale, bias, and the zero-point compensation. */
  sw: Float32Array;
  bias: Float32Array;
  comp: Int32Array;
};

/**
 * Folds the per-input-channel activation scale into the weight, quantizes
 * per output channel symmetric int8, and packs. K runs (ky, kx, c) so an
 * NHWC im2col is a straight copy of Cin bytes per tap.
 *
 * y[n] = sw[n] * (sum_k Wq[k,n] * q[k] + comp[n]) + bias[n], with
 * comp[n] = -sum_k Wq[k,n] * zp[c(k)], which is what makes the kernel a plain
 * dot product on the raw int8 values.
 */
export function prepareQConv(w: Tensor, bias: Float32Array | null, inScale: Float32Array, inZp: Int32Array): QConvWeights {
  const [Cout, Cin, kh, kw] = w.dims;
  if (Cin % 4 || Cout % 8) throw new Error(`prepareQConv: Cin=${Cin} Cout=${Cout} need Cin%4==0 and Cout%8==0`);
  const K = kh * kw * Cin, N = Cout;
  const folded = new Float32Array(K * N);
  for (let n = 0; n < Cout; n++) {
    for (let c = 0; c < Cin; c++) {
      for (let t = 0; t < kh * kw; t++) {
        folded[(t * Cin + c) * N + n] = w.data[(n * Cin + c) * kh * kw + t] * inScale[c];
      }
    }
  }
  const wq = new Int8Array(K * N);
  const sw = new Float32Array(N), comp = new Int32Array(N);
  for (let n = 0; n < N; n++) {
    let amax = 0;
    for (let k = 0; k < K; k++) amax = Math.max(amax, Math.abs(folded[k * N + n]));
    const s = amax / 127 || 1;
    sw[n] = s;
    let acc = 0;
    for (let k = 0; k < K; k++) {
      const q = roundHalfEven(folded[k * N + n] / s);
      wq[k * N + n] = q;
      acc -= q * inZp[k % Cin];
    }
    comp[n] = acc;
  }
  return { K, N, packed: packWeights(K, N, wq), sw, bias: bias ?? new Float32Array(N), comp };
}

/** The packed weights and epilogue vectors, uploaded once. */
export type QConvResident = QConvWeights & { ptr: { packed: number; sw: number; bias: number; comp: number } };

export function uploadQConv(r: Resident, q: QConvWeights): QConvResident {
  return {
    ...q,
    ptr: {
      packed: r.ar.persistBytes(q.packed),
      sw: r.ar.persist(q.sw),
      bias: r.ar.persist(q.bias),
      comp: r.ar.persistBytes(q.comp),
    },
  };
}

/** Allocate an int8 NHWC tensor; the arena counts floats, so round the bytes up. */
export function allocQ(r: Resident, dims: number[], q: QParams): QT {
  const bytes = dims.reduce((a, b) => a * b, 1);
  return { dims, ptr: r.ar.allocScratch((bytes + 3) >> 2), bytes, q };
}

/**
 * 1x1 convolution on an int8 NHWC input. With `out` the result is int8 NHWC
 * at that scale; without it the result is fp32 NCHW, the way the fp32 graph
 * expects it, via a transpose of the kernel's NHWC rows.
 */
export function qconv1x1(
  r: Resident, x: QT, w: QConvResident, act: number, p0: number, p1: number,
  residual: QT | null, out: QParams | null,
): QT | RT {
  const [n, Cin, H, W] = x.dims;
  if (n !== 1) throw new Error("qconv1x1: batch 1 only");
  if (Cin !== w.K) throw new Error(`qconv1x1: Cin ${Cin} but K ${w.K}`);
  const M = H * W, N = w.N;
  const rptr = residual ? residual.ptr : 0;
  const rs = residual ? residual.q.ptr.scale : 0;
  const rzp = residual ? residual.q.ptr.zp : 0;
  if (out) {
    const y = allocQ(r, [1, N, H, W], out);
    r.ar.pQGemm([M, w.K, N, x.ptr, w.ptr.packed, y.ptr, w.ptr.sw, w.ptr.bias, w.ptr.comp, act,
      rptr, rs, rzp, out.ptr.inv, out.ptr.zp, 1], p0, p1);
    return y;
  }
  const tmp = r.alloc([M, N]);
  r.ar.pQGemm([M, w.K, N, x.ptr, w.ptr.packed, tmp.ptr, w.ptr.sw, w.ptr.bias, w.ptr.comp, act,
    rptr, rs, rzp, 0, 0, 0], p0, p1);
  const y = r.alloc([1, N, H, W]);
  r.ar.k.transpose_f32(M, N, tmp.ptr, y.ptr);
  r.ar.release(tmp.ptr);
  return y;
}

/** fp32 NCHW in the arena to int8 NHWC at the given per-channel scale. */
export function quantizeToQ(r: Resident, x: RT, p: QParams): QT {
  const [, C, H, W] = x.dims;
  const q = allocQ(r, x.dims, p);
  r.ar.k.quantize_nhwc(C, H * W, x.ptr, q.ptr, p.ptr.inv, p.ptr.zp, 0, H * W);
  return q;
}

export function dequantizeFromQ(r: Resident, q: QT): RT {
  const [, C, H, W] = q.dims;
  const y = r.alloc(q.dims);
  r.ar.k.dequantize_nchw(C, H * W, q.ptr, y.ptr, q.q.ptr.scale, q.q.ptr.zp, 0, H * W);
  return y;
}

// ---- depthwise ---------------------------------------------------------------

export type DwResident = {
  kh: number; kw: number; sy: number; sx: number; pt: number; pl: number;
  ptr: { w: number; sw: number; bias: number };
};

/**
 * Per-channel symmetric int8 of a [C,1,kh,kw] weight laid out [tap][C]. The
 * kernel sums wq * (xq - zp), so the dequant scale is the weight's times the
 * input's, per channel.
 */
export function uploadDepthwise(
  r: Resident, w: Tensor, bias: Float32Array | null, geom: { sy: number; sx: number; pt: number; pl: number },
  inScale: Float32Array,
): DwResident {
  const [C, , kh, kw] = w.dims;
  const taps = kh * kw;
  const wq = new Int8Array(taps * C), sw = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let amax = 0;
    for (let t = 0; t < taps; t++) amax = Math.max(amax, Math.abs(w.data[c * taps + t]));
    const s = amax / 127 || 1;
    sw[c] = s * inScale[c];
    for (let t = 0; t < taps; t++) wq[t * C + c] = roundHalfEven(w.data[c * taps + t] / s);
  }
  return {
    kh, kw, ...geom,
    ptr: { w: r.ar.persistBytes(wq), sw: r.ar.persist(sw), bias: r.ar.persist(bias ?? new Float32Array(C)) },
  };
}

export function qdepthwise(r: Resident, x: QT, d: DwResident, act: number, out: QParams | null): QT | RT {
  const [N, C, H, W] = x.dims;
  const OH = Math.floor((H + 2 * d.pt - d.kh) / d.sy) + 1;
  const OW = Math.floor((W + 2 * d.pl - d.kw) / d.sx) + 1;
  const plane = OH * OW;
  if (out) {
    const y = allocQ(r, [N, C, OH, OW], out);
    for (let n = 0; n < N; n++) {
      r.ar.pQDepthwise([C, H, W, OH, OW, d.kh, d.kw, d.sy, d.sx, d.pt, d.pl, x.ptr + n * C * H * W, x.q.ptr.zp,
        d.ptr.w, d.ptr.sw, d.ptr.bias, act, y.ptr + n * C * plane, out.ptr.inv, out.ptr.zp, 1]);
    }
    return y;
  }
  const tmp = r.alloc([N, plane, C]);
  const y = r.alloc([N, C, OH, OW]);
  for (let n = 0; n < N; n++) {
    const t = tmp.ptr + n * plane * C * 4;
    r.ar.pQDepthwise([C, H, W, OH, OW, d.kh, d.kw, d.sy, d.sx, d.pt, d.pl, x.ptr + n * C * H * W, x.q.ptr.zp,
      d.ptr.w, d.ptr.sw, d.ptr.bias, act, t, 0, 0, 0]);
    r.ar.k.transpose_f32(plane, C, t, y.ptr + n * C * plane * 4);
  }
  r.ar.release(tmp.ptr);
  return y;
}

