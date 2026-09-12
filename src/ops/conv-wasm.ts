import type { RT, Resident } from "../runtime/resident.ts";
import type { ConvAttrs } from "./nn.ts";

/**
 * Conv through the WASM kernels. Three paths, picked by shape:
 *   depthwise      group == Cout, one filter per channel
 *   1x1 dense      the input is already the GEMM operand, no im2col
 *   dense k>1      im2col then GEMM
 * Grouped-but-not-depthwise does not occur in these models and is rejected.
 */
export function convResident(
  r: Resident,
  x: RT,
  w: RT,
  b: RT | null,
  a: ConvAttrs,
): RT {
  const [N, Cin, H, W] = x.dims;
  const [Cout, CinPer, kh, kw] = w.dims;
  const [sy, sx] = a.strides;
  const [dy, dx] = a.dilations;
  const [pt, pl, pb, pr] = a.pads;
  const OH = Math.floor((H + pt + pb - (kh - 1) * dy - 1) / sy) + 1;
  const OW = Math.floor((W + pl + pr - (kw - 1) * dx - 1) / sx) + 1;
  const out = r.alloc([N, Cout, OH, OW]);
  const bPtr = b ? b.ptr : 0;
  if (a.group !== 1 && !(a.group === Cout && CinPer === 1)) {
    throw new Error(`convResident: group ${a.group} with ${CinPer} channels per group`);
  }

  const pointwise = kh === 1 && kw === 1 && sy === 1 && sx === 1 && !pt && !pl && !pb && !pr;
  const K = Cin * kh * kw;
  // One scratch buffer serves the whole batch; im2col rewrites it per item.
  const col = !pointwise && a.group === 1 ? r.alloc([K, OH * OW]) : null;

  // Batch items are independent, so each one runs the same kernel at its own
  // offset. Batching exists to amortise the per-node dispatch, not to widen
  // the GEMM: NCHW puts the batch stride outside the channel stride.
  for (let n = 0; n < N; n++) {
    const xi = x.ptr + n * Cin * H * W * 4;
    const yi = out.ptr + n * Cout * OH * OW * 4;
    if (a.group === Cout && CinPer === 1) {
      r.ar.pDepthwise([Cout, H, W, OH, OW, kh, kw, sy, sx, pt, pl, xi, w.ptr, bPtr, yi, 0]);
    } else if (pointwise) {
      r.ar.pGemm(Cout, Cin, OH * OW, w.ptr, xi, yi, bPtr, 0);
    } else {
      r.ar.pIm2col([Cin, H, W, OH, OW, kh, kw, sy, sx, pt, pl, dy, dx, xi, col!.ptr]);
      r.ar.pGemm(Cout, K, OH * OW, w.ptr, col!.ptr, yi, bPtr, 0);
    }
  }
  if (col) r.ar.release(col.ptr);
  return out;
}

/**
 * ConvTranspose with 2x2 kernel and stride 2: the output windows never
 * overlap, so each of the four taps is an independent [Cout,Cin] x [Cin,HW]
 * GEMM and the results interleave into the doubled grid.
 * Both ConvTranspose nodes in the detection model have this shape.
 */
export function convTranspose2x2Resident(r: Resident, x: RT, w: RT, b: RT | null): RT {
  const [, Cin, H, W] = x.dims;
  const Cout = w.dims[1];
  const HW = H * W;
  const OH = H * 2;
  const OW = W * 2;

  // Weights arrive as [Cin, Cout, 2, 2]; GEMM wants [Cout, Cin] per tap. The
  // regrouped copies live in scratch and are rebuilt per call, which is
  // nothing next to the GEMMs and keeps them out of the sealed weight region.
  const wData = new Float32Array(w.len);
  r.ar.readInto(w.ptr, wData);
  const out = r.alloc([1, Cout, OH, OW]);
  const outData = new Float32Array(out.len);
  const tapBuf = new Float32Array(Cout * Cin);
  const tapRT = r.alloc([Cout, Cin]);
  const acc = r.alloc([Cout, HW]);
  const accData = new Float32Array(Cout * HW);

  for (let t = 0; t < 4; t++) {
    for (let ic = 0; ic < Cin; ic++) {
      for (let oc = 0; oc < Cout; oc++) tapBuf[oc * Cin + ic] = wData[(ic * Cout + oc) * 4 + t];
    }
    r.ar.write(tapRT.ptr, tapBuf);
    r.ar.pGemm(Cout, Cin, HW, tapRT.ptr, x.ptr, acc.ptr, 0, 0);
    r.ar.readInto(acc.ptr, accData);

    const ky = t >> 1;
    const kx = t & 1;
    for (let oc = 0; oc < Cout; oc++) {
      const src = oc * HW;
      const dst = oc * OH * OW;
      for (let y = 0; y < H; y++) {
        const row = dst + (y * 2 + ky) * OW + kx;
        const srow = src + y * W;
        for (let xi = 0; xi < W; xi++) outData[row + xi * 2] = accData[srow + xi];
      }
    }
  }
  r.ar.release(tapRT.ptr);
  r.ar.release(acc.ptr);

  if (b) {
    const bias = new Float32Array(b.len);
    r.ar.readInto(b.ptr, bias);
    for (let oc = 0; oc < Cout; oc++) {
      const base = oc * OH * OW;
      for (let i = 0; i < OH * OW; i++) outData[base + i] += bias[oc];
    }
  }
  r.ar.write(out.ptr, outData);
  return out;
}

export function matmulResident(r: Resident, a: RT, b: RT): RT {
  const ad = a.dims;
  const bd = b.dims;
  const M = ad[ad.length - 2];
  const K = ad[ad.length - 1];
  const N = bd[bd.length - 1];
  const batchA = ad.slice(0, -2).reduce((p, q) => p * q, 1);
  const batchB = bd.slice(0, -2).reduce((p, q) => p * q, 1);
  if (batchB !== 1) throw new Error(`matmulResident: batched B not supported [${ad}] x [${bd}]`);

  const out = r.alloc([...ad.slice(0, -2), M, N]);
  for (let z = 0; z < batchA; z++) {
    r.ar.pGemm(M, K, N, a.ptr + z * M * K * 4, b.ptr, out.ptr + z * M * N * 4, 0, 0);
  }
  return out;
}
