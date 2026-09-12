import { parseOnnx } from "../src/onnx/parse.ts";
import { Session } from "../src/runtime/graph.ts";
import { loadKernels, loadKernelsThreaded } from "../src/wasm/backend.ts";

const [file, dimsArg, threadsArg, runsArg] = process.argv.slice(2);
const dims = dimsArg.split("x").map(Number);
const threads = Number(threadsArg);
const runs = Number(runsArg ?? 5);
const n = dims.reduce((a, b) => a * b, 1);
const data = new Float32Array(n);
let seed = 7;
for (let i = 0; i < n; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; data[i] = (seed / 0xffffffff) * 2 - 1; }

const bytes = new Uint8Array(await Bun.file(threads === 1 ? "src/wasm/kernels.wasm" : "src/wasm/kernels.shared.wasm").arrayBuffer());
const arena = threads === 1 ? await loadKernels(bytes) : await loadKernelsThreaded(bytes, threads);
const g = parseOnnx(new Uint8Array(await Bun.file(file).arrayBuffer()));
const s = new Session(g, arena);
const feeds = { [g.inputs[0].name]: { dims, data } };
s.run(feeds);
const t = performance.now();
for (let i = 0; i < runs; i++) s.run(feeds);
console.log(((performance.now() - t) / runs).toFixed(1));
arena.destroy();
