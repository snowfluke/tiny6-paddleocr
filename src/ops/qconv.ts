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

/**
 * An int8 NHWC activation in the arena: `dims` stay NCHW for the graph's
 * sake. `cs` is the channel stride of a pixel, equal to C except for a
 * boundary tensor padded so a dense kernel can read channels four at a time.
 */
export type QT = { dims: number[]; ptr: number; bytes: number; cs: number; q: QParams };

/** Everything a Conv needs after its weights have been folded and quantized. */
export type QConvWeights = {
  K: number;
  N: number;
  packed: Int8Array;
  /** Offset the stored weights carry: 0 on a signed-dot engine, 128 on x86. */
  wzp: number;
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
export function prepareQConv(w: Tensor, bias: Float32Array | null, inScale: Float32Array, inZp: Int32Array, wzp = 0): QConvWeights {
  const [Cout, Cin, kh, kw] = w.dims;
  // Input channels pad to a multiple of four with zero weights; the tensor
  // side pads to the same stride (see QT.cs).
  const cs = (Cin + 3) & ~3;
  if (Cout % 8) throw new Error(`prepareQConv: Cout=${Cout} must be a multiple of 8`);
  const K = kh * kw * cs, N = Cout;
  const folded = new Float32Array(K * N);
  for (let n = 0; n < Cout; n++) {
    for (let c = 0; c < Cin; c++) {
      for (let t = 0; t < kh * kw; t++) {
        folded[(t * cs + c) * N + n] = w.data[(n * Cin + c) * kh * kw + t] * inScale[c];
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
      if (k % cs < Cin) acc -= q * inZp[k % cs];
    }
    comp[n] = acc;
  }
  // An engine that reads the weight operand as unsigned gets w + 128 as
  // bytes; the GEMM epilogue takes 128 * (row sum) back off.
  if (wzp) for (let i = 0; i < wq.length; i++) wq[i] = (wq[i] + wzp) << 24 >> 24;
  return { K, N, packed: packWeights(K, N, wq), wzp, sw, bias: bias ?? new Float32Array(N), comp };
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
export function allocQ(r: Resident, dims: number[], q: QParams, cs = dims[1]): QT {
  const bytes = dims[0] * cs * dims[2] * dims[3];
  return { dims, ptr: r.ar.allocScratch((bytes + 3) >> 2), bytes, cs, q };
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
  const [B, , H, W] = x.dims;
  if (x.cs !== w.K) throw new Error(`qconv1x1: channel stride ${x.cs} but K ${w.K}`);
  const M = H * W, N = w.N;
  const rs = residual ? residual.q.ptr.scale : 0;
  const rzp = residual ? residual.q.ptr.zp : 0;
  const rowsum = w.wzp ? r.ar.allocScratch(M) : 0;
  const rows = (b: number) => {
    if (rowsum) r.ar.pRowsum(w.K, M, x.ptr + b * M * x.cs, rowsum);
    return rowsum;
  };
  // Batch items are independent; each runs at its own offset.
  if (out) {
    const y = allocQ(r, [B, N, H, W], out);
    for (let b = 0; b < B; b++) {
      r.ar.pQGemm([M, w.K, N, x.ptr + b * M * x.cs, w.ptr.packed, y.ptr + b * M * N, w.ptr.sw, w.ptr.bias, w.ptr.comp, act,
        residual ? residual.ptr + b * M * N : 0, rs, rzp, out.ptr.inv, out.ptr.zp, 1, w.wzp, rows(b)], p0, p1);
    }
    if (rowsum) r.ar.release(rowsum);
    return y;
  }
  const tmp = r.alloc([M, N]);
  const y = r.alloc([B, N, H, W]);
  for (let b = 0; b < B; b++) {
    r.ar.pQGemm([M, w.K, N, x.ptr + b * M * x.cs, w.ptr.packed, tmp.ptr, w.ptr.sw, w.ptr.bias, w.ptr.comp, act,
      residual ? residual.ptr + b * M * N : 0, rs, rzp, 0, 0, 0, w.wzp, rows(b)], p0, p1);
    r.ar.pTranspose(M, N, tmp.ptr, y.ptr + b * M * N * 4);
  }
  r.ar.release(tmp.ptr);
  if (rowsum) r.ar.release(rowsum);
  return y;
}

/** fp32 NCHW in the arena to int8 NHWC at the given per-channel scale. */
export function quantizeToQ(r: Resident, x: RT, p: QParams, cs = x.dims[1]): QT {
  const [N, C, H, W] = x.dims;
  const q = allocQ(r, x.dims, p, cs);
  for (let n = 0; n < N; n++) {
    r.ar.pQuantize(C, cs, H * W, x.ptr + n * C * H * W * 4, q.ptr + n * cs * H * W, p.ptr.inv, p.ptr.zp);
  }
  return q;
}

export type DenseGeom = { kh: number; kw: number; sy: number; sx: number; pt: number; pl: number; pb: number; pr: number };

/**
 * Dense convolution of any kernel size: int8 im2col into a scratch column
 * matrix, then the int8 GEMM, both split by output pixel. Batch 1.
 */
export function qconvDense(
  r: Resident, x: QT, w: QConvResident, g: DenseGeom, act: number, p0: number, p1: number, out: QParams | null,
): QT | RT {
  const [B, , H, W] = x.dims;
  const OH = Math.floor((H + g.pt + g.pb - g.kh) / g.sy) + 1;
  const OW = Math.floor((W + g.pl + g.pr - g.kw) / g.sx) + 1;
  const M = OH * OW, N = w.N;
  if (g.kh * g.kw * x.cs !== w.K) throw new Error(`qconvDense: K ${g.kh * g.kw * x.cs} but weights have ${w.K}`);
  const col = r.ar.allocScratch((M * w.K + 3) >> 2);
  const rowsum = w.wzp ? r.ar.allocScratch(M) : 0;
  const geom = (b: number) => [H, W, OH, OW, g.kh, g.kw, g.sy, g.sx, g.pt, g.pl, x.cs, x.ptr + b * H * W * x.cs, x.q.ptr.zp, col, rowsum];
  let y: QT | RT;
  if (out) {
    y = allocQ(r, [B, N, OH, OW], out);
    for (let b = 0; b < B; b++) {
      r.ar.pQConvDense([...geom(b), N, w.ptr.packed, y.ptr + b * M * N, w.ptr.sw, w.ptr.bias, w.ptr.comp, act, out.ptr.inv, out.ptr.zp, 1, w.wzp], p0, p1);
    }
  } else {
    const tmp = r.alloc([M, N]);
    y = r.alloc([B, N, OH, OW]);
    for (let b = 0; b < B; b++) {
      r.ar.pQConvDense([...geom(b), N, w.ptr.packed, tmp.ptr, w.ptr.sw, w.ptr.bias, w.ptr.comp, act, 0, 0, 0, w.wzp], p0, p1);
      r.ar.pTranspose(M, N, tmp.ptr, y.ptr + b * M * N * 4);
    }
    r.ar.release(tmp.ptr);
  }
  r.ar.release(col);
  if (rowsum) r.ar.release(rowsum);
  return y;
}

export function dequantizeFromQ(r: Resident, q: QT): RT {
  const [N, C, H, W] = q.dims;
  if (q.cs !== C) throw new Error("dequantizeFromQ: padded tensors are read by their convolution only");
  const y = r.alloc(q.dims);
  for (let n = 0; n < N; n++) {
    r.ar.pDequantize(C, H * W, q.ptr + n * C * H * W, y.ptr + n * C * H * W * 4, q.q.ptr.scale, q.q.ptr.zp);
  }
  return y;
}

// ---- depthwise ---------------------------------------------------------------

export type DwResident = { kh: number; kw: number; ptr: { w: number; sw: number; bias: number } };
export type DwGeom = { sy: number; sx: number; pt: number; pl: number; pb: number; pr: number };

/**
 * Per-channel symmetric int8 of a [C,1,kh,kw] weight laid out [tap][C]. The
 * kernel sums wq * (xq - zp), so the dequant scale is the weight's times the
 * input's, per channel.
 */
export function uploadDepthwise(r: Resident, w: Tensor, bias: Float32Array | null, inScale: Float32Array): DwResident {
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
    kh, kw,
    ptr: { w: r.ar.persistBytes(wq), sw: r.ar.persist(sw), bias: r.ar.persist(bias ?? new Float32Array(C)) },
  };
}

export function qdepthwise(r: Resident, x: QT, d: DwResident, g: DwGeom, act: number, out: QParams | null): QT | RT {
  const [N, C, H, W] = x.dims;
  const OH = Math.floor((H + g.pt + g.pb - d.kh) / g.sy) + 1;
  const OW = Math.floor((W + g.pl + g.pr - d.kw) / g.sx) + 1;
  const plane = OH * OW;
  if (out) {
    const y = allocQ(r, [N, C, OH, OW], out);
    for (let n = 0; n < N; n++) {
      r.ar.pQDepthwise([C, H, W, OH, OW, d.kh, d.kw, g.sy, g.sx, g.pt, g.pl, x.ptr + n * C * H * W, x.q.ptr.zp,
        d.ptr.w, d.ptr.sw, d.ptr.bias, act, y.ptr + n * C * plane, out.ptr.inv, out.ptr.zp, 1]);
    }
    return y;
  }
  const tmp = r.alloc([N, plane, C]);
  const y = r.alloc([N, C, OH, OW]);
  for (let n = 0; n < N; n++) {
    const t = tmp.ptr + n * plane * C * 4;
    r.ar.pQDepthwise([C, H, W, OH, OW, d.kh, d.kw, g.sy, g.sx, g.pt, g.pl, x.ptr + n * C * H * W, x.q.ptr.zp,
      d.ptr.w, d.ptr.sw, d.ptr.bias, act, t, 0, 0, 0]);
    r.ar.pTranspose(plane, C, t, y.ptr + n * C * plane * 4);
  }
  r.ar.release(tmp.ptr);
  return y;
}

