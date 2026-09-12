import { parseOnnx } from "../src/onnx/parse.ts";
const g = parseOnnx(new Uint8Array(await Bun.file(process.argv[2]).arrayBuffer()));
const want = new Set(process.argv.slice(3));
const init = g.initializers;
for (const n of g.nodes) {
  if (!want.has(n.opType)) continue;
  const a = [...n.attrs.values()].map((x) =>
    `${x.name}=${x.ints ? `[${x.ints}]` : x.s ?? x.i ?? x.f?.toFixed(3) ?? "?"}`).join(" ");
  const ins = n.input.map((i) => {
    const t = init.get(i);
    if (!t) return i;
    const v = t.data instanceof Float32Array && t.data.length <= 4 ? `=${[...t.data]}` : "";
    return `${i}<init [${t.dims}]${v}>`;
  });
  console.log(`${n.opType.padEnd(14)} ${a}`);
  console.log(`  in : ${ins.join("  ")}`);
}
