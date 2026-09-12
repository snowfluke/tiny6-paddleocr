// Direct GEMM throughput, packed vs unpacked, on the shapes these models use.
const bytes = new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer());
const { instance } = await WebAssembly.instantiate(bytes, {});
const k = instance.exports as any;

const PANEL = 1 << 18;
const cases: [string, number, number, number][] = [
  ["1x1 conv, Cin=64  @240x240", 64, 64, 57600],
  ["1x1 conv, Cin=160 @120x120", 160, 160, 14400],
  ["3x3 dense im2col, K=576", 64, 576, 57600],
  ["rec 1x1, Cin=96 @6x40", 96, 96, 240],
];

for (const [label, M, K, N] of cases) {
  const base = (k.heap_base() + 15) & ~15;
  const needBytes = base + (M * K + K * N + M * N + PANEL + 4096) * 4;
  const pages = Math.ceil(needBytes / 65536) + 2;
  const have = k.memory.buffer.byteLength / 65536;
  if (pages > have) k.memory.grow(pages - have);

  const panel = base;
  const A = panel + PANEL * 4;
  const B = A + M * K * 4;
  const C = B + K * N * 4;
  const mem = () => new Float32Array(k.memory.buffer);
  const m0 = mem();
  for (let i = 0; i < M * K; i++) m0[A / 4 + i] = (i % 13) * 0.01;
  for (let i = 0; i < K * N; i++) m0[B / 4 + i] = (i % 7) * 0.1;

  const flops = 2 * M * K * N;
  const time = (_a: number, _b: number) => {
    k.gemm_range(M, K, N, N, N, A, B, C, 0, 0, 0, 0, N);
    const t = performance.now();
    const R = 5;
    for (let i = 0; i < R; i++) k.gemm_range(M, K, N, N, N, A, B, C, 0, 0, 0, 0, N);
    return (performance.now() - t) / R;
  };
  const ms = time(0, 0);
  console.log(`${label.padEnd(28)} ${(flops / ms / 1e6).toFixed(1).padStart(5)} GF/s`);
}
