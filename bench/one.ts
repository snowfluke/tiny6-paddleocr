// One engine, one case, one process. ORT's thread pool spins, so measuring
// both engines in a single process makes whichever runs second look slow.
import { parseOnnx } from "../src/onnx/parse.ts";
import { Session } from "../src/runtime/graph.ts";
import { loadKernels } from "../src/wasm/backend.ts";

const [engine, file, dimsArg, runsArg] = process.argv.slice(2);
const dims = dimsArg.split("x").map(Number);
const n = dims.reduce((a, b) => a * b, 1);
const runs = Number(runsArg ?? 5);
const data = new Float32Array(n);
let seed = 7;
for (let i = 0; i < n; i++) {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  data[i] = (seed / 0xffffffff) * 2 - 1;
}

let ms = 0;
if (engine === "ours") {
  const g = parseOnnx(new Uint8Array(await Bun.file(file).arrayBuffer()));
  const arena = await loadKernels(new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer()));
  const s = new Session(g, arena);
  const feeds = { [g.inputs[0].name]: { dims, data } };
  s.run(feeds);
  const t = performance.now();
  for (let i = 0; i < runs; i++) s.run(feeds);
  ms = (performance.now() - t) / runs;
} else {
  const ort = await import("onnxruntime-node");
  const s = await ort.InferenceSession.create(file);
  const feeds = { [s.inputNames[0]]: new ort.Tensor("float32", data, dims) };
  await s.run(feeds);
  const t = performance.now();
  for (let i = 0; i < runs; i++) await s.run(feeds);
  ms = (performance.now() - t) / runs;
}
console.log(ms.toFixed(1));
