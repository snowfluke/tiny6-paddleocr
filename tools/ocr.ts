import { Ocr } from "../src/ocr.ts";
import { makeRecWorker } from "../src/node.ts";
import { decodeImage } from "../src/image/jpeg.ts";
import { DEFAULT_DETECT } from "../src/pipeline/detect.ts";

const path = process.argv[2] ?? "test/images/receipt.png";
const read = async (p: string) => new Uint8Array(await Bun.file(p).arrayBuffer());

const ocr = await Ocr.create({
  det: await read("models/det.onnx"),
  rec: await read("models/rec.onnx"),
  dict: await Bun.file("models/dict.txt").text(),
  wasm: await read("src/wasm/kernels.wasm"),
  wasmShared: await read("src/wasm/kernels.shared.wasm"),
  makeRecWorker,
});
const img = await decodeImage(await read(path));

const t0 = performance.now();
const opts = { ...DEFAULT_DETECT, rotated: process.argv.includes("--rotated") };
const { boxes } = ocr.detect(img, opts);
const tDet = performance.now();
const lines = await ocr.recognizeBoxesAsync(img, boxes);
const tAll = performance.now();

if (process.argv.includes("--text")) {
  console.log(lines.map((l) => l.text).join("\n"));
} else {
  for (const l of lines) console.log(`${l.confidence.toFixed(2)}  ${l.text}`);
  console.log(`\n${path}  ${img.width}x${img.height}`);
  console.log(`${ocr.arena.threads} det threads, ${ocr.recWorkers} rec workers  detect ${(tDet - t0).toFixed(0)} ms  ${boxes.length} boxes  recognize ${(tAll - tDet).toFixed(0)} ms  total ${(tAll - t0).toFixed(0)} ms  ${lines.length} lines`);
}

ocr.destroy();
