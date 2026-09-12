import type { RT, Resident } from "../runtime/resident.ts";
import type { ConvAttrs } from "./nn.ts";

/**
 * How large one im2col strip may get. Four megabytes sits inside the shared L2
 * on this machine, so the GEMM's repeated passes over the strip never reach
 * memory. Bigger wastes cache; smaller multiplies the dispatch count.
 */
const STRIP_BYTES = 1 << 22;

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
  act = 0,
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
  const plane = OH * OW;

  // The column matrix is built a strip at a time, sized to stay in cache. The
  // GEMM re-reads all of B once per eight output channels, so a full-width
  // matrix is read many times over: a 3x3 convolution at 960x960 expands a
  // 14.7 MB input into 132 MB and the GEMM then moved about a gigabyte.
  const needsCol = !pointwise && a.group === 1;
  const strip = needsCol
    ? Math.max(8, Math.min(plane, Math.floor(STRIP_BYTES / (K * 4)) & ~7))
    : 0;
  // One scratch buffer serves the whole batch; each strip rewrites it.
  const col = needsCol ? r.alloc([K, strip]) : null;

  // Batch items are independent, so each one runs the same kernel at its own
  // offset. Batching exists to amortise the per-node dispatch, not to widen
  // the GEMM: NCHW puts the batch stride outside the channel stride.
  for (let n = 0; n < N; n++) {
    const xi = x.ptr + n * Cin * H * W * 4;
    const yi = out.ptr + n * Cout * OH * OW * 4;
    if (a.group === Cout && CinPer === 1) {
      r.ar.pDepthwise([Cout, H, W, OH, OW, kh, kw, sy, sx, pt, pl, xi, w.ptr, bPtr, yi, act]);
    } else if (pointwise) {
      r.ar.pGemm(Cout, Cin, OH * OW, w.ptr, xi, yi, bPtr, act);
    } else {
      for (let p0 = 0; p0 < plane; p0 += strip) {
        const width = Math.min(strip, plane - p0);
        r.ar.pIm2colStrip([Cin, H, W, OW, kh, kw, sy, sx, pt, pl, dy, dx, p0, width, xi, col!.ptr]);
        // B is this strip, `width` wide; C is the full output row, `plane` wide.
        r.ar.pGemm(Cout, K, width, w.ptr, col!.ptr, yi + p0 * 4, bPtr, act, width, plane);
      }
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

  // Weights arrive as [Cin, Cout, 2, 2]; GEMM wants [Cout, Cin] per tap. The
  // regrouped copies live in scratch and are rebuilt per call, which is
  // nothing next to the GEMMs and keeps them out of the sealed weight region.
  const wData = new Float32Array(w.len);
  r.ar.readInto(w.ptr, wData);
  const out = r.alloc([1, Cout, H * 2, W * 2]);
  const tapBuf = new Float32Array(Cout * Cin);
  const tapRT = r.alloc([Cout, Cin]);
  const acc = r.alloc([Cout, HW]);

  // The four taps write disjoint pixels of the output, so each one's GEMM can
  // carry the bias and nothing has to be accumulated afterwards.
  for (let t = 0; t < 4; t++) {
    for (let ic = 0; ic < Cin; ic++) {
      for (let oc = 0; oc < Cout; oc++) tapBuf[oc * Cin + ic] = wData[(ic * Cout + oc) * 4 + t];
    }
    r.ar.write(tapRT.ptr, tapBuf);
    r.ar.pGemm(Cout, Cin, HW, tapRT.ptr, x.ptr, acc.ptr, b ? b.ptr : 0, 0);
    r.ar.pScatter2x2(Cout, H, W, t >> 1, t & 1, acc.ptr, out.ptr);
  }
  r.ar.release(tapRT.ptr);
  r.ar.release(acc.ptr);
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
