import { parseOnnx } from "../src/onnx/parse.ts";
for (const f of ["models/det.onnx", "models/rec.onnx"]) {
  const g = parseOnnx(new Uint8Array(await Bun.file(f).arrayBuffer()));
  const kinds = new Map<string, number>();
  for (const n of g.nodes) {
    if (n.opType !== "Conv" && n.opType !== "ConvTranspose") continue;
    const w = g.initializers.get(n.input[1]);
    const grp = n.attrs.get("group")?.i ?? 1;
    const k = n.attrs.get("kernel_shape")?.ints ?? [];
    const s = n.attrs.get("strides")?.ints ?? [1, 1];
    const d = n.attrs.get("dilations")?.ints ?? [1, 1];
    const pads = n.attrs.get("pads")?.ints ?? [0, 0, 0, 0];
    const [co, ci] = w?.dims ?? [0, 0];
    const depthwise = grp > 1 && grp === co;
    const kind = `${n.opType} k${k.join("x")} s${s.join("")} d${d.join("")} ${
      depthwise ? "depthwise" : grp > 1 ? `group${grp}` : "dense"
    } bias=${n.input.length > 2} pads=${pads.join("")}`;
    kinds.set(kind, (kinds.get(kind) ?? 0) + 1);
  }
  console.log(`--- ${f}`);
  for (const [k, v] of [...kinds].sort((a, b) => b[1] - a[1])) console.log(`  ${String(v).padStart(3)}x  ${k}`);
}
