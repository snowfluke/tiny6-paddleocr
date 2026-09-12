import { Ocr } from "../src/ocr.ts";
import { decodePng } from "../src/image/png.ts";
import { DEFAULT_DETECT } from "../src/pipeline/detect.ts";

const read = async (p: string) => new Uint8Array(await Bun.file(p).arrayBuffer());
const ocr = await Ocr.create({
  det: await read("models/det.onnx"), rec: await read("models/rec.onnx"),
  dict: await Bun.file("models/dict.txt").text(),
  wasm: await read("src/wasm/kernels.wasm"),
  wasmShared: await read("src/wasm/kernels.shared.wasm"),
});
const img = await decodePng(await read("test/images/receipt.png"));
const { boxes } = ocr.detect(img, DEFAULT_DETECT);
ocr.recognizeBoxes(img, boxes);

const widths = boxes.map((b) => Math.max(8, Math.round(48 * (b.width / b.height))));
widths.sort((a, b) => a - b);
console.log(`threads ${ocr.arena.threads}   crops ${boxes.length}`);
console.log(`crop widths: min ${widths[0]} median ${widths[widths.length >> 1]} max ${widths[widths.length - 1]}`);
console.log(`total tensor columns: ${widths.reduce((a, b) => a + b, 0)}`);

const t = performance.now();
for (let i = 0; i < 3; i++) ocr.recognizeBoxes(img, boxes);
const per = (performance.now() - t) / 3;
console.log(`recognize all ${boxes.length}: ${per.toFixed(0)} ms  -> ${(per / boxes.length).toFixed(1)} ms/crop`);

// Same total work as one wide tensor, to show what batching could reach.
const wide = widths.reduce((a, b) => a + b, 0);
console.log(`\nif the same columns ran as ONE tensor of width ${wide}:`);
ocr.destroy();
