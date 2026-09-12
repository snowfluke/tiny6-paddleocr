// Diff the hand-written runtime against the ORT golden dump, node by node.
// Reports the FIRST node that drifts, which is the one that is actually broken.

import { parseOnnx } from "../src/onnx/parse.ts";
import { Session } from "../src/runtime/graph.ts";
import type { Tensor } from "../src/runtime/tensor.ts";
import { loadKernels } from "../src/wasm/backend.ts";

export function unpack(buf: Uint8Array): Map<string, Tensor> {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const out = new Map<string, Tensor>();
  let o = 0;
  const count = dv.getUint32(o, true);
  o += 4;
  for (let i = 0; i < count; i++) {
    const nameLen = dv.getUint32(o, true);
    o += 4;
    const name = new TextDecoder().decode(buf.subarray(o, o + nameLen));
    o += nameLen;
    const rank = dv.getUint32(o, true);
    o += 4;
    const dims: number[] = [];
    for (let d = 0; d < rank; d++) {
      dims.push(dv.getInt32(o, true));
      o += 4;
    }
    const n = dims.reduce((a, b) => a * b, 1);
    out.set(name, { dims, data: new Float32Array(buf.slice(o, o + n * 4).buffer) });
    o += n * 4;
  }
  return out;
}

export function compare(got: Tensor, want: Tensor) {
  let maxAbs = 0;
  let maxRel = 0;
  const n = Math.min(got.data.length, want.data.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(got.data[i] - want.data[i]);
    if (d > maxAbs) maxAbs = d;
    const scale = Math.max(Math.abs(want.data[i]), 1e-3);
    if (d / scale > maxRel) maxRel = d / scale;
  }
  const shapeOk = got.dims.join() === want.dims.join();
  return { maxAbs, maxRel, shapeOk, n };
}

if (import.meta.main) {
  const which = process.argv[2] ?? "det";
  const tol = Number(process.argv.find((a) => a.startsWith("--tol="))?.slice(6) ?? 2e-3);
  const g = parseOnnx(new Uint8Array(await Bun.file(`models/${which}.onnx`).arrayBuffer()));
  const golden = unpack(new Uint8Array(await Bun.file(`test/golden/${which}.golden.bin`).arrayBuffer()));
  const inputRaw = new Uint8Array(await Bun.file(`test/golden/${which}.input.bin`).arrayBuffer());
  const dims = which === "det" ? [1, 3, 256, 256] : [1, 3, 48, 320];
  const feeds = { [g.inputs[0].name]: { dims, data: new Float32Array(inputRaw.buffer) } };

  let checked = 0;
  let firstBad: string | null = null;
  const t0 = performance.now();
  const useWasm = process.argv.includes("--wasm");
  const arena = useWasm
    ? await loadKernels(new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer()))
    : undefined;
  const s = new Session(g, arena);
  s.run(feeds, {
    onNode: (node, outs) => {
      for (let i = 0; i < node.output.length; i++) {
        const want = golden.get(node.output[i]);
        if (!want) continue;
        const r = compare(outs[i], want);
        checked++;
        const bad = !r.shapeOk || r.maxAbs > tol;
        if (bad && !firstBad) {
          firstBad = node.output[i];
          console.log(`FAIL at node ${checked}: ${node.opType} "${node.name}" -> ${node.output[i]}`);
          console.log(`  got  [${outs[i].dims}]  want [${want.dims}]`);
          console.log(`  maxAbs=${r.maxAbs.toExponential(3)}  maxRel=${r.maxRel.toExponential(3)}`);
          console.log(`  got  sample: ${[...outs[i].data.slice(0, 6)].map((v) => v.toFixed(5))}`);
          console.log(`  want sample: ${[...want.data.slice(0, 6)].map((v) => v.toFixed(5))}`);
        }
      }
    },
  });
  const ms = performance.now() - t0;
  if (!firstBad) console.log(`OK  ${which}${useWasm ? " [wasm]" : " [ts]"}: ${checked} tensors match within ${tol}  (${ms.toFixed(0)} ms)`);
  else process.exitCode = 1;
}
