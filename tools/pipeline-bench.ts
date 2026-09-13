// End to end on one image, warm. Prints one line so tools/ab.ts can read it.
//
// The reference (ppu-paddle-ocr) is measured this way: several runs in one
// process, first discarded, minimum reported. A cold single run is a
// different number by 40% and is what tools/ocr.ts prints.
//
//   bun tools/pipeline-bench.ts [image] [recWorkers] [runs]

import { defaultRecWorkers, Ocr } from "../src/ocr.ts";
import { makeRecWorker } from "../src/node.ts";
import { decodeImage } from "../src/image/jpeg.ts";

const int8 = process.argv.includes("--int8");
const [imageArg, workersArg, runsArg] = process.argv.slice(2).filter((a) => a !== "--int8");
const image = imageArg ?? "test/images/receipt.jpg";
const recWorkers = workersArg ? Number(workersArg) : defaultRecWorkers();
const runs = Number(runsArg ?? 6);
const read = async (p: string) => new Uint8Array(await Bun.file(p).arrayBuffer());

const ocr = await Ocr.create({
  det: await read("models/det.onnx"),
  rec: await read("models/rec.onnx"),
  dict: await Bun.file("models/dict.txt").text(),
  wasm: await read("src/wasm/kernels.wasm"),
  wasmShared: await read("src/wasm/kernels.shared.wasm"),
  recWorkers,
  makeRecWorker: recWorkers > 1 ? makeRecWorker : undefined,
  ...(int8 ? { detCalib: await Bun.file("models/det.calib.json").text(), recCalib: await Bun.file("models/rec.calib.json").text() } : {}),
});
const img = await decodeImage(await read(image));

let det = Infinity;
let rec = Infinity;
let total = Infinity;
let lines = 0;
let boxes = 0;
for (let i = 0; i <= runs; i++) {
  const t0 = performance.now();
  const d = ocr.detect(img);
  const t1 = performance.now();
  const ls = await ocr.recognizeBoxesAsync(img, d.boxes);
  const t2 = performance.now();
  if (i === 0) continue;
  det = Math.min(det, t1 - t0);
  rec = Math.min(rec, t2 - t1);
  total = Math.min(total, t2 - t0);
  lines = ls.length;
  boxes = d.boxes.length;
}
console.error(
  `${image} ${ocr.arena.threads} det threads, ${ocr.recWorkers} rec workers: detect ${det.toFixed(0)} ms, ${boxes} boxes, recognize ${rec.toFixed(0)} ms, ${lines} lines`,
);
console.log(total.toFixed(1));
ocr.destroy();
