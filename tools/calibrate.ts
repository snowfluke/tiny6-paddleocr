// Records the per-channel range of every activation in both models over a
// set of images and writes models/{det,rec}.calib.json, which Session's
// int8 option reads. Ranges are taken on the fused fp32 graph, so the names
// match what the planner sees.
//
//   bun tools/calibrate.ts [image ...]        (default: test/images/receipt.png)
import { Ocr } from "../src/ocr.ts";
import type { Session } from "../src/runtime/graph.ts";
import type { Tensor } from "../src/runtime/tensor.ts";
import type { Calibration } from "../src/runtime/qplan.ts";
import { decodeImage } from "../src/image/jpeg.ts";
import { DEFAULT_DETECT } from "../src/pipeline/detect.ts";

const read = async (p: string) => new Uint8Array(await Bun.file(p).arrayBuffer());
const images = process.argv.length > 2 ? process.argv.slice(2) : ["test/images/receipt.png"];

const ocr = await Ocr.create({
  det: await read("models/det.onnx"),
  rec: await read("models/rec.onnx"),
  dict: await Bun.file("models/dict.txt").text(),
  wasm: await read("src/wasm/kernels.wasm"),
  wasmShared: await read("src/wasm/kernels.shared.wasm"),
  recWorkers: 1,
});

const out: Record<string, Calibration> = { det: {}, rec: {} };
for (const which of ["det", "rec"] as const) {
  const s = (ocr as unknown as Record<string, Session>)[which];
  const calib = out[which];
  const note = (name: string, t: Tensor) => {
    if (t.dims.length !== 4) return;
    const [, C] = t.dims;
    const inner = t.dims[2] * t.dims[3];
    const r = calib[name] ?? { dims: t.dims.slice(), min: new Array(C).fill(Infinity), max: new Array(C).fill(-Infinity) };
    for (let i = 0; i < t.data.length; i++) {
      const c = Math.floor(i / inner) % C;
      const v = t.data[i];
      if (v < r.min[c]) r.min[c] = v;
      if (v > r.max[c]) r.max[c] = v;
    }
    calib[name] = r;
  };
  const run = s.run.bind(s);
  s.run = (feeds, opts = {}) => {
    for (const [k, v] of Object.entries(feeds)) note(k, v);
    return run(feeds, { ...opts, onNode: (n, outs) => outs.forEach((t, i) => note(n.output[i], t)) });
  };
}
for (const path of images) {
  ocr.text(await decodeImage(await read(path)), DEFAULT_DETECT, { minConfidence: 0, dropSeparators: false });
}
ocr.destroy();
for (const which of ["det", "rec"] as const) {
  const round = (v: number) => Number(v.toPrecision(6));
  const compact = Object.fromEntries(Object.entries(out[which]).map(([k, r]) => [k, { dims: r.dims, min: r.min.map(round), max: r.max.map(round) }]));
  await Bun.write(`models/${which}.calib.json`, JSON.stringify(compact));
  console.log(`models/${which}.calib.json: ${Object.keys(compact).length} tensors from ${images.length} image(s)`);
}
