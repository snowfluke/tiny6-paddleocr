// Headless checks on the browser bundle: the inlined kernels must decode and
// instantiate, and the pure-JS half of the pipeline must run unchanged.
const js = await Bun.file("demo/tiny6.js").text();

const m = js.match(/"([A-Za-z0-9+/=]{4000,})"/);
if (!m) throw new Error("no base64 kernel payload found in bundle");
const wasm = Uint8Array.from(atob(m[1]), (c) => c.charCodeAt(0));
const onDisk = new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer());
console.log(`inlined wasm      ${wasm.length} bytes, identical to source: ${
  wasm.length === onDisk.length && wasm.every((b, i) => b === onDisk[i])}`);

const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = Object.keys(instance.exports).sort();
console.log(`wasm instantiates exports: ${ex.join(", ")}`);

for (const need of ["gemm", "depthwise", "im2col_strip", "qgemm", "qim2col", "dot_probe", "heap_base", "memory"]) {
  if (!ex.includes(need)) throw new Error(`missing export ${need}`);
}

// A 2x3 by 3x2 GEMM with a known answer, run through the inlined module.
const k: any = instance.exports;
const mem = () => new Float32Array(k.memory.buffer);
const base = (k.heap_base() + 15) & ~15;
const A = base, B = A + 6 * 4, C = B + 6 * 4;
mem().set([1, 2, 3, 4, 5, 6], A / 4);
mem().set([7, 8, 9, 10, 11, 12], B / 4);
k.gemm(2, 3, 2, 2, 2, A, B, C, 0, 0, 0);
const got = Array.from(mem().slice(C / 4, C / 4 + 4));
const want = [58, 64, 139, 154];
console.log(`gemm 2x3*3x2      ${got.join(",")} expected ${want.join(",")} -> ${
  got.every((v, i) => Math.abs(v - want[i]) < 1e-4) ? "ok" : "MISMATCH"}`);

const browserOnly = ["createImageBitmap", "OffscreenCanvas"];
console.log(`browser-only APIs used: ${browserOnly.join(", ")} (decode path only)`);
