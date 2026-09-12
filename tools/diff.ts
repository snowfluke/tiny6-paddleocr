import { Ocr } from "../src/ocr.ts";
import { decodePng } from "../src/image/png.ts";
const read = async (p: string) => new Uint8Array(await Bun.file(p).arrayBuffer());
const ocr = await Ocr.create({
  det: await read("models/det.onnx"), rec: await read("models/rec.onnx"),
  dict: await Bun.file("models/dict.txt").text(), wasm: await read("src/wasm/kernels.wasm"),
});
const got = ocr.text(await decodePng(await read("test/images/receipt.png")), undefined, { minConfidence: 0, dropSeparators: false }).split("\n");
const want = (await Bun.file("test/images/receipt-reference.txt").text()).trimEnd().split("\n");
let same = 0;
for (let i = 0; i < Math.max(got.length, want.length); i++) {
  const ok = got[i] === want[i];
  if (ok) { same++; console.log(`  = ${got[i]}`); }
  else { console.log(`  ! ours: ${got[i]}`); console.log(`    ref : ${want[i]}`); }
}
console.log(`\n${same}/${want.length} identical`);

ocr.destroy();
