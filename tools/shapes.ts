import { parseOnnx } from "../src/onnx/parse.ts";
import { unpack } from "./check.ts";
const g = parseOnnx(new Uint8Array(await Bun.file("models/det.onnx").arrayBuffer()));
const golden = unpack(new Uint8Array(await Bun.file("test/golden/det.golden.bin").arrayBuffer()));
const want = new Set(process.argv.slice(2));
for (const n of g.nodes) {
  if (!want.has(n.opType)) continue;
  const w = g.initializers.get(n.input[1]);
  const o = golden.get(n.output[0]);
  const i = golden.get(n.input[0]);
  console.log(`${n.opType} w=[${w?.dims ?? "-"}] in=[${i?.dims ?? "?"}] out=[${o?.dims ?? "?"}]`);
}
