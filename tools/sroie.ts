// The int8 accuracy question on enough text to resolve it: SROIE receipts,
// calibrated on one subset and scored on another, fp32 against fake-quant.
//
//   bun tools/sroie.ts [dir] [count] [pct]   (default: ../data/sroie/data, 60, 99.99)
//
// SROIE ground truth is uppercased and split inconsistently, so the score is
// order-independent token F1 (bag of words, uppercased) and a normalized CER
// (uppercase, whitespace stripped). Both are paired per image against fp32.
import { assets, calibrate, type Mode, quantizedOcr } from "./fakequant.ts";
import { Ocr } from "../src/ocr.ts";
import { decodeImage } from "../src/image/jpeg.ts";
import { DEFAULT_DETECT } from "../src/pipeline/detect.ts";
import type { RGBA } from "../src/image/png.ts";

const dir = process.argv[2] ?? "../data/sroie/data";
const count = Number(process.argv[3] ?? 60);
const clip = Number(process.argv[4] ?? 99.99);
const CALIB = 16;

const names = [...new Bun.Glob("*.jpg").scanSync(`${dir}/img`)].sort();
const stride = Math.floor(names.length / (count + CALIB));
const picked = names.filter((_, i) => i % stride === 0).slice(0, count + CALIB);
const calibNames = picked.slice(0, CALIB);
const evalNames = picked.slice(CALIB);

const load = async (name: string) => decodeImage(new Uint8Array(await Bun.file(`${dir}/img/${name}`).arrayBuffer()));
const truth = async (name: string) =>
  (await Bun.file(`${dir}/box/${name.replace(".jpg", ".csv")}`).text()).trim().split("\n")
    .map((l) => l.split(",").slice(8).join(",")).join("\n");

const tokens = (s: string) => s.toUpperCase().split(/\s+/).filter(Boolean);
function tokenF1(pred: string, gt: string): number {
  const want = new Map<string, number>();
  for (const t of tokens(gt)) want.set(t, (want.get(t) ?? 0) + 1);
  const p = tokens(pred);
  let hit = 0;
  for (const t of p) { const n = want.get(t) ?? 0; if (n > 0) { hit++; want.set(t, n - 1); } }
  const g = tokens(gt).length;
  return p.length + g === 0 ? 1 : (2 * hit) / (p.length + g);
}
function editDistance(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}
const flat = (s: string) => s.toUpperCase().replace(/\s+/g, "");
const cer = (pred: string, gt: string) => editDistance(flat(pred), flat(gt)) / flat(gt).length;

// The eight recognition activations the single-receipt sweep flagged.
const SENSITIVE = ["multiply.0.0", "gelu.1.0", "add.6.0", "add.12.0", "add.25.0", "gelu.7.0", "add.31.0", "gelu.9.0"]
  .map((n) => `p2o.pd_op.${n}`);

const cal = await calibrate(await Promise.all(calibNames.map(load)), clip);
const images: [string, RGBA, string][] = [];
for (const n of evalNames) images.push([n, await load(n), await truth(n)]);
console.log(`calibrated on ${CALIB} receipts at p${clip}, scoring ${images.length} others, ${images.reduce((a, [, , g]) => a + flat(g).length, 0)} GT characters`);

type Config = { label: string; det: Mode; rec: Mode; only?: Set<string>; kernels?: boolean };
const configs: Config[] = [
  { label: "fp32", det: "fp32", rec: "fp32" },
  { label: "int8 kernels", det: "wa-chan", rec: "wa-chan", kernels: true },
  { label: "both int8 chan 7b", det: "wa-chan7", rec: "wa-chan7" },
  { label: "both int8 chan 7a", det: "wa-chan7a", rec: "wa-chan7a" },
  { label: "det int8", det: "wa-asym", rec: "fp32" },
  { label: "rec int8", det: "fp32", rec: "wa-asym" },
  { label: "both int8", det: "wa-asym", rec: "wa-asym" },
  { label: "rec int8 chan", det: "fp32", rec: "wa-chan" },
  { label: "both int8 chan", det: "wa-chan", rec: "wa-chan" },
];
const recActs = new Set(cal.rec.keys());
for (const n of SENSITIVE) recActs.delete(n);
configs.push({ label: "both, 8 rec fp32", det: "wa-asym", rec: "wa-asym", only: recActs });

// SROIE_CONFIGS=fp32,rec int8 limits the run; SROIE_DUMP=dir writes each config's text.
const wanted = process.env.SROIE_CONFIGS?.split(",");
/** The real int8 path: calibrate with tools/calibrate.ts on the same receipts, then load the JSON. */
async function kernelOcr(): Promise<Ocr> {
  const p = Bun.spawnSync(["bun", "tools/calibrate.ts", ...calibNames.map((n) => `${dir}/img/${n}`)]);
  if (p.exitCode !== 0) throw new Error(new TextDecoder().decode(p.stderr));
  const read = async (f: string) => new Uint8Array(await Bun.file(f).arrayBuffer());
  return Ocr.create({
    ...assets, det: await read("models/det.onnx"), rec: await read("models/rec.onnx"),
    detCalib: await Bun.file("models/det.calib.json").text(), recCalib: await Bun.file("models/rec.calib.json").text(),
  });
}

let base: { f1: number[]; cer: number[] } | null = null;
console.log("\nconfig             tokenF1   CER     vs fp32: F1 delta   worse/better images");
for (const c of configs) {
  if (wanted && !wanted.includes(c.label)) continue;
  const ocr = c.kernels ? await kernelOcr() : await quantizedOcr(c.det, c.rec, cal, c.only);
  const f1: number[] = [], ce: number[] = [], texts: string[] = [];
  const t0 = performance.now();
  for (const [, img, gt] of images) {
    const text = ocr.text(img, DEFAULT_DETECT);
    texts.push(text);
    f1.push(tokenF1(text, gt));
    ce.push(cer(text, gt));
  }
  const stalls = ocr.arena.stalls;
  ocr.destroy();
  if (process.env.SROIE_DUMP) await Bun.write(`${process.env.SROIE_DUMP}/${c.label.replace(/\W+/g, "-")}.txt`, texts.join("\n=====\n"));
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  let line = `${c.label.padEnd(18)} ${(100 * mean(f1)).toFixed(2).padStart(6)}%  ${(100 * mean(ce)).toFixed(2).padStart(5)}%`;
  if (base) {
    const d = f1.map((v, i) => v - base!.f1[i]);
    const worse = d.filter((x) => x < -1e-9).length, better = d.filter((x) => x > 1e-9).length;
    line += `   ${(100 * mean(d)).toFixed(2).padStart(6)} pt        ${worse}/${better}`;
  } else {
    base = { f1, cer: ce };
    // One run under memory pressure produced different fp32 text; the hash flags a repeat.
    const h = new Bun.CryptoHasher("sha1");
    h.update(texts.join("\n"));
    line += `   fp32 text sha ${h.digest("hex").slice(0, 12)}`;
  }
  console.log(line + `   (${((performance.now() - t0) / 1000).toFixed(0)} s, ${stalls.count} late shares, ${stalls.ms.toFixed(0)} ms)`);
}
