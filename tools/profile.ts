import { parseOnnx } from "../src/onnx/parse.ts";
import { Session } from "../src/runtime/graph.ts";
import { loadKernels, loadKernelsThreaded } from "../src/wasm/backend.ts";

const which = process.argv[2] ?? "det";
const dims = (process.argv[3] ?? (which === "det" ? "1x3x256x256" : "1x3x48x320")).split("x").map(Number);
const g = parseOnnx(new Uint8Array(await Bun.file(`models/${which}.onnx`).arrayBuffer()));
const n = dims.reduce((a, b) => a * b, 1);
// --threads N profiles the pooled path, so an op that does not speed up
// shows itself as a rising share rather than a falling one.
const ti = process.argv.indexOf("--threads");
const threads = ti < 0 ? 1 : Number(process.argv[ti + 1]);
const arena = process.argv.includes("--wasm")
  ? threads > 1
    ? await loadKernelsThreaded(
      new Uint8Array(await Bun.file("src/wasm/kernels.shared.wasm").arrayBuffer()),
      threads,
    )
    : await loadKernels(new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer()))
  : undefined;
const int8 = process.argv.includes("--int8") && arena
  ? { int8: JSON.parse(await Bun.file(`models/${which}.calib.json`).text()) }
  : {};
const s = new Session(g, arena, int8);
if (s.int8Nodes) console.log(`${s.int8Nodes} nodes on int8`);
const feeds = { [g.inputs[0].name]: { dims, data: new Float32Array(n).fill(0.25) } };
s.run(feeds);

const byOp = new Map<string, { ms: number; n: number }>();
let last = performance.now();
const t0 = last;
s.run(feeds, {
  onNodeDone: (node) => {
    const now = performance.now();
    const label = s.planned(node) ? `${node.opType} (int8)` : node.opType;
    const e = byOp.get(label) ?? { ms: 0, n: 0 };
    e.ms += now - last;
    e.n++;
    byOp.set(label, e);
    last = now;
  },
});
const total = performance.now() - t0;
const stalls = arena?.stalls;
console.log(`${which} total ${total.toFixed(0)} ms, ${threads} thread(s)${stalls?.count ? `, ${stalls.count} late shares (${stalls.ms.toFixed(0)} ms)` : ""}`);
for (const [op, e] of [...byOp].sort((a, b) => b[1].ms - a[1].ms)) {
  if (e.ms < 1) continue;
  console.log(`  ${op.padEnd(20)} ${e.ms.toFixed(0).padStart(6)} ms  ${String(e.n).padStart(3)}x  ${((e.ms / total) * 100).toFixed(1)}%`);
}

arena?.destroy();
