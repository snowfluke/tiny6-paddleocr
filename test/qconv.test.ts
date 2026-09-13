import { expect, test } from "bun:test";
import { loadKernels } from "../src/wasm/backend.ts";
import { Resident } from "../src/runtime/resident.ts";
import { conv2d, type ConvAttrs } from "../src/ops/nn.ts";
import { dequantizeLinear, qgemmReference, quantizeLinear } from "../src/ops/quant.ts";
import { dequantizeFromQ, prepareQConv, qconv1x1, qdepthwise, quantizeToQ, uploadDepthwise, uploadQConv, uploadQParams } from "../src/ops/qconv.ts";
import type { Tensor } from "../src/runtime/tensor.ts";

const wasm = new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer());

let seed = 3;
const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
const tensor = (dims: number[], lo: number, hi: number): Tensor =>
  ({ dims, data: Float32Array.from({ length: dims.reduce((a, b) => a * b) }, () => lo + rnd() * (hi - lo)) });

/** Per-channel asymmetric int8 parameters from the tensor's own range, channel axis 1. */
function calibrate(t: Tensor) {
  const [, C, H, W] = t.dims;
  const scale = new Float32Array(C), zp = new Int32Array(C);
  for (let c = 0; c < C; c++) {
    let lo = 0, hi = 0;
    for (let i = 0; i < H * W; i++) { const v = t.data[c * H * W + i]; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    scale[c] = (hi - lo) / 255 || 1;
    zp[c] = Math.round(-128 - lo / scale[c]);
  }
  return { scale, zp };
}

const attrs: ConvAttrs = { kernel: [1, 1], strides: [1, 1], pads: [0, 0, 0, 0], dilations: [1, 1], group: 1 };

test("int8 1x1 conv equals the integer reference and tracks the fp32 conv", async () => {
  const [Cin, Cout, H, W] = [16, 24, 5, 7];
  const x = tensor([1, Cin, H, W], -3, 5);
  const w = tensor([Cout, Cin, 1, 1], -0.5, 0.5);
  const bias = tensor([Cout], -1, 1).data;
  const xq = calibrate(x);

  const arena = await loadKernels(wasm);
  const r = new Resident(arena);
  const xp = uploadQParams(r, xq.scale, xq.zp);
  const q = uploadQConv(r, prepareQConv(w, bias, xq.scale, xq.zp));
  const xr = quantizeToQ(r, r.upload(x), xp);

  // The same integer inputs through the TypeScript reference, rows = pixels.
  const a = new Int8Array(H * W * Cin);
  arena.readBytesInto(xr.ptr, a);
  const wq = new Int8Array(q.K * q.N);
  for (let kb = 0; kb < q.K; kb += 4) for (let n = 0; n < q.N; n++) for (let t = 0; t < 4; t++) wq[(kb + t) * q.N + n] = q.packed[kb * q.N + n * 4 + t];
  const want = qgemmReference(H * W, q.K, q.N, a, wq, { sw: q.sw, bias: q.bias, comp: q.comp, act: 1 }) as Float32Array;

  const y = qconv1x1(r, xr, q, 1, 0, 0, null, null) as { ptr: number; dims: number[]; len: number };
  const got = r.download(y);
  expect(got.dims).toEqual([1, Cout, H, W]);
  for (let n = 0; n < Cout; n++) for (let p = 0; p < H * W; p++) expect(got.data[n * H * W + p]).toBe(want[p * Cout + n]);

  // And against the fp32 convolution of the dequantized operands, with relu.
  const xdq = dequantizeLinear(quantizeLinear(x, { dims: [Cin], data: xq.scale }, { dims: [Cin], data: Float32Array.from(xq.zp) }, 1),
    { dims: [Cin], data: xq.scale }, { dims: [Cin], data: Float32Array.from(xq.zp) }, 1);
  const ref = conv2d(xdq, w, { dims: [Cout], data: bias }, attrs);
  let maxErr = 0, maxRef = 0;
  for (let i = 0; i < ref.data.length; i++) {
    const v = Math.max(ref.data[i], 0);
    maxErr = Math.max(maxErr, Math.abs(got.data[i] - v));
    maxRef = Math.max(maxRef, Math.abs(v));
  }
  // Weight quantization is the only error left: 127 levels per output channel.
  expect(maxErr).toBeLessThan(0.02 * maxRef);
  arena.destroy();
});

test("int8 output round-trips through dequantize within one step", async () => {
  const [Cin, Cout, H, W] = [8, 16, 3, 4];
  const x = tensor([1, Cin, H, W], -1, 1);
  const w = tensor([Cout, Cin, 1, 1], -0.5, 0.5);
  const xq = calibrate(x);
  const arena = await loadKernels(wasm);
  const r = new Resident(arena);
  const xp = uploadQParams(r, xq.scale, xq.zp);
  const q = uploadQConv(r, prepareQConv(w, null, xq.scale, xq.zp));
  const xr = quantizeToQ(r, r.upload(x), xp);
  const f32 = r.download(qconv1x1(r, xr, q, 0, 0, 0, null, null) as never);
  const oq = calibrate(f32);
  const op = uploadQParams(r, oq.scale, oq.zp);
  const back = r.download(dequantizeFromQ(r, qconv1x1(r, xr, q, 0, 0, 0, null, op) as never));
  for (let i = 0; i < f32.data.length; i++) {
    const c = Math.floor(i / (H * W));
    expect(Math.abs(back.data[i] - f32.data[i])).toBeLessThanOrEqual(oq.scale[c] * 0.5 + 1e-6);
  }
  arena.destroy();
});

for (const [C, H, W, k, s] of [[32, 9, 7, 3, 1], [16, 8, 10, 3, 2], [16, 7, 7, 5, 1]]) {
  test(`int8 depthwise ${k}x${k} stride ${s} tracks the fp32 conv`, async () => {
    const pad = Math.floor(k / 2);
    const x = tensor([1, C, H, W], -2, 3);
    const w = tensor([C, 1, k, k], -0.4, 0.4);
    const bias = tensor([C], -0.5, 0.5).data;
    const xq = calibrate(x);
    const arena = await loadKernels(wasm);
    const r = new Resident(arena);
    const xp = uploadQParams(r, xq.scale, xq.zp);
    const d = uploadDepthwise(r, w, bias, { sy: s, sx: s, pt: pad, pl: pad }, xq.scale);
    const xr = quantizeToQ(r, r.upload(x), xp);
    const got = r.download(qdepthwise(r, xr, d, 1, null) as never);
    const sc = { dims: [C], data: xq.scale }, zp = { dims: [C], data: Float32Array.from(xq.zp) };
    const xdq = dequantizeLinear(quantizeLinear(x, sc, zp, 1), sc, zp, 1);
    const ref = conv2d(xdq, w, { dims: [C], data: bias }, { kernel: [k, k], strides: [s, s], pads: [pad, pad, pad, pad], dilations: [1, 1], group: C });
    expect(got.dims).toEqual(ref.dims);
    let maxErr = 0, maxRef = 0;
    for (let i = 0; i < ref.data.length; i++) {
      const v = Math.max(ref.data[i], 0);
      maxErr = Math.max(maxErr, Math.abs(got.data[i] - v));
      maxRef = Math.max(maxRef, Math.abs(v));
    }
    expect(maxErr).toBeLessThan(0.02 * maxRef);
    arena.destroy();
  });
}
