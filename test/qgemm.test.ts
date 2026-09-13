import { expect, test } from "bun:test";
import { packWeights, qgemmReference, type QGemmEpilogue } from "../src/ops/quant.ts";

const bytes = new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer());
const { instance } = await WebAssembly.instantiate(bytes, {});
const k = instance.exports as Record<string, Function> & { memory: WebAssembly.Memory; heap_base: () => number };

let seed = 1;
const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
const i8 = (n: number, lo = -128, hi = 127) => Int8Array.from({ length: n }, () => Math.floor(lo + rnd() * (hi - lo + 1)));
const f32 = (n: number, lo: number, hi: number) => Float32Array.from({ length: n }, () => lo + rnd() * (hi - lo));

/** Bump-allocate in wasm memory, 16-byte aligned, and copy the data in. */
let top = (k.heap_base() + 15) & ~15;
function put(data: ArrayBufferView): number {
  const ptr = top;
  top = (top + data.byteLength + 15) & ~15;
  if (top > k.memory.buffer.byteLength) k.memory.grow(Math.ceil((top - k.memory.buffer.byteLength) / 65536) + 1);
  new Uint8Array(k.memory.buffer, ptr, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return ptr;
}

function run(M: number, K: number, N: number, e: QGemmEpilogue, a: Int8Array, w: Int8Array, lo = 0, hi = M) {
  const outI8 = e.out ? 1 : 0;
  const c = put(new Uint8Array(M * N * (outI8 ? 1 : 4)));
  k.qgemm(
    M, K, N, put(a), put(packWeights(K, N, w)), c,
    put(e.sw), put(e.bias), put(e.comp), e.act,
    e.res ? put(e.res.q) : 0, e.res ? put(e.res.scale) : 0, e.res ? put(e.res.zp) : 0,
    e.out ? put(e.out.inv) : 0, e.out ? put(e.out.zp) : 0, outI8, e.p0 ?? 0, e.p1 ?? 0, lo, hi,
  );
  return outI8 ? Int8Array.from(new Int8Array(k.memory.buffer, c, M * N)) : Float32Array.from(new Float32Array(k.memory.buffer, c, M * N));
}

const same = (got: ArrayLike<number>, want: ArrayLike<number>) => expect(Array.from(got)).toEqual(Array.from(want));

const epilogue = (N: number, extra: Partial<QGemmEpilogue> = {}): QGemmEpilogue => ({
  sw: f32(N, 0.001, 0.01), bias: f32(N, -1, 1), comp: Int32Array.from({ length: N }, () => Math.floor(rnd() * 20000 - 10000)), act: 0, ...extra,
});

const shapes: [number, number, number][] = [[8, 4, 8], [16, 64, 16], [13, 12, 24], [1, 8, 8], [9, 160, 32]];

for (const [M, K, N] of shapes) {
  test(`qgemm ${M}x${K}x${N} matches the integer reference bit for bit`, () => {
    // Weights stay inside 7 bits, the range the relaxed dot product promises.
    const a = i8(M * K), w = i8(K * N, -127, 127);
    for (const act of [0, 1]) {
      const e = epilogue(N, { act });
      same(run(M, K, N, e, a, w), qgemmReference(M, K, N, a, w, e));
      const res = { q: i8(M * N), scale: f32(N, 0.01, 0.1), zp: Int32Array.from({ length: N }, () => Math.floor(rnd() * 40 - 20)) };
      const out = { inv: f32(N, 5, 50), zp: Int32Array.from({ length: N }, () => Math.floor(rnd() * 40 - 20)) };
      const e2 = epilogue(N, { act, res, out });
      same(run(M, K, N, e2, a, w), qgemmReference(M, K, N, a, w, e2));
    }
  });
}

test("qgemm rows outside [lo, hi) are untouched", () => {
  const [M, K, N] = [24, 16, 16];
  const a = i8(M * K), w = i8(K * N, -127, 127), e = epilogue(N);
  const got = run(M, K, N, e, a, w, 8, 21) as Float32Array;
  const want = qgemmReference(M, K, N, a, w, e) as Float32Array;
  for (let m = 0; m < M; m++) {
    const row = got.subarray(m * N, (m + 1) * N);
    if (m >= 8 && m < 21) same(row, want.subarray(m * N, (m + 1) * N));
    else expect(row.every((v) => v === 0)).toBe(true);
  }
});

test("qgemm gelu epilogue is within fp32 noise of the reference", () => {
  const [M, K, N] = [16, 32, 16];
  const a = i8(M * K), w = i8(K * N, -127, 127);
  const e = epilogue(N, { act: 2, p0: Math.SQRT1_2, p1: 0.5 });
  const got = run(M, K, N, e, a, w) as Float32Array;
  const want = qgemmReference(M, K, N, a, w, e) as Float32Array;
  for (let i = 0; i < got.length; i++) expect(Math.abs(got[i] - want[i])).toBeLessThan(1e-5 * (1 + Math.abs(want[i])));
});
