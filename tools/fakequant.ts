// Day-1 int8 question: does post-training quantization of the Conv and MatMul
// weights and activations change what the receipt reads? Runs the models with
// QuantizeLinear/DequantizeLinear pairs in fp32 (fake quantization), so the
// error is the kernel's, not its speed. Calibration is on the receipt itself,
// which is the optimistic bound: a failure here is final, a pass is not.
//
//   bun tools/fakequant.ts [--calib tilted] [--head] [--pct 99.99] [--sweep]
//
// Weights: per-output-channel symmetric int8. Activations: per-tensor int8 at
// every Conv/MatMul input, min/max calibrated, symmetric or asymmetric.
// --head also quantizes the 6906-class CTC head, which the plan keeps in fp32.
// --pct clips the calibration range to that percentile instead of min/max.
// --sweep quantizes one recognition activation at a time to find the ones
// that flip characters.
import { Ocr } from "../src/ocr.ts";
import { Session } from "../src/runtime/graph.ts";
import { type OnnxGraph, type OnnxNode, type OnnxTensor, parseOnnx } from "../src/onnx/parse.ts";
import { decodePng, type RGBA } from "../src/image/png.ts";
import { DEFAULT_DETECT } from "../src/pipeline/detect.ts";
import type { Tensor } from "../src/runtime/tensor.ts";

const read = async (p: string) => new Uint8Array(await Bun.file(p).arrayBuffer());
const argv = process.argv.slice(2);
const quantHead = argv.includes("--head");
const calibName = argv[argv.indexOf("--calib") + 1];
const calibImages = argv.includes("--calib") ? [calibName] : ["receipt"];
const pct = argv.includes("--pct") ? Number(argv[argv.indexOf("--pct") + 1]) : 100;
const sweep = argv.includes("--sweep");

export const assets = {
  dict: await Bun.file("models/dict.txt").text(),
  wasm: await read("src/wasm/kernels.wasm"),
  wasmShared: await read("src/wasm/kernels.shared.wasm"),
  recWorkers: 1,
};
export const detGraph = parseOnnx(await read("models/det.onnx"));
export const recGraph = parseOnnx(await read("models/rec.onnx"));
export const receipt = await decodePng(await read("test/images/receipt.png"));
const reference = (await Bun.file("test/images/receipt-reference.txt").text()).trimEnd().split("\n");
const gt = (await Bun.file("test/images/receipt-gt.txt").text()).trimEnd().split("\n");

/** Per-tensor bounds, plus per-channel bounds along `axis` for the wa-chan mode. */
type Range = { min: number; max: number; axis: number; cmin: Float32Array; cmax: Float32Array };
export type Mode = "fp32" | "w" | "wa-sym" | "wa-asym" | "wa-chan";

const isGemm = (n: OnnxNode) => n.opType === "Conv" || n.opType === "MatMul";
const isHead = (g: OnnxGraph, n: OnnxNode) => (g.initializers.get(n.input[1])?.dims.at(-1) ?? 0) > 1000;

/** Run the pipeline once with fp32 sessions and record min/max of every tensor a GEMM reads. */
export type Calibration = { det: Map<string, Range>; rec: Map<string, Range> };

export async function calibrate(images?: RGBA[], clip = pct): Promise<Calibration> {
  const ocr = await Ocr.create({ ...assets, det: await read("models/det.onnx"), rec: await read("models/rec.onnx") });
  const out = { det: new Map<string, Range>(), rec: new Map<string, Range>() };
  // Percentile clipping needs the values, so keep a strided sample per tensor.
  const samples = { det: new Map<string, Float32Array[]>(), rec: new Map<string, Float32Array[]>() };
  for (const which of ["det", "rec"] as const) {
    const s = (ocr as unknown as Record<string, Session>)[which];
    const wanted = new Set(s.graph.nodes.filter(isGemm).map((n) => n.input[0]));
    const ranges = out[which];
    const note = (name: string, t: Tensor) => {
      if (!wanted.has(name)) return;
      let lo = Infinity, hi = -Infinity;
      for (const v of t.data) { if (v < lo) lo = v; if (v > hi) hi = v; }
      // NCHW activations carry channels on axis 1; the 3-D sequence into the
      // recognition MatMul carries them last.
      const axis = t.dims.length === 4 ? 1 : t.dims.length - 1;
      const C = t.dims[axis];
      const r = ranges.get(name) ??
        { min: Infinity, max: -Infinity, axis, cmin: new Float32Array(C).fill(Infinity), cmax: new Float32Array(C).fill(-Infinity) };
      const inner = t.dims.slice(axis + 1).reduce((a, b) => a * b, 1);
      for (let i = 0; i < t.data.length; i++) {
        const c = Math.floor(i / inner) % C;
        const v = t.data[i];
        if (v < r.cmin[c]) r.cmin[c] = v;
        if (v > r.cmax[c]) r.cmax[c] = v;
      }
      ranges.set(name, { ...r, min: Math.min(r.min, lo), max: Math.max(r.max, hi) });
      const step = Math.max(1, Math.floor(t.data.length / 16384));
      const pick = new Float32Array(Math.ceil(t.data.length / step));
      for (let i = 0, j = 0; i < t.data.length; i += step) pick[j++] = t.data[i];
      samples[which].set(name, [...(samples[which].get(name) ?? []), pick]);
    };
    const run = s.run.bind(s);
    s.run = (feeds, opts = {}) => {
      for (const [k, v] of Object.entries(feeds)) note(k, v);
      return run(feeds, { ...opts, onNode: (n, outs) => outs.forEach((t, i) => note(n.output[i], t)) });
    };
  }
  images ??= await Promise.all(calibImages.map(async (name) => decodePng(await read(`test/images/${name}.png`))));
  for (const img of images) ocr.text(img, DEFAULT_DETECT, { minConfidence: 0, dropSeparators: false });
  ocr.destroy();
  if (clip < 100) {
    for (const which of ["det", "rec"] as const) {
      for (const [name, parts] of samples[which]) {
        const all = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
        for (let i = 0, o = 0; i < parts.length; o += parts[i++].length) all.set(parts[i], o);
        all.sort();
        const tail = (all.length - 1) * (1 - clip / 100);
        out[which].set(name, { ...out[which].get(name)!, min: all[Math.floor(tail)], max: all[Math.ceil(all.length - 1 - tail)] });
      }
    }
  }
  return out;
}

const f32 = (name: string, v: number[], dims: number[] = []): OnnxTensor =>
  ({ name, dims, dataType: 1, data: new Float32Array(v) });

/**
 * Per-output-channel symmetric int8 round trip of a Conv [Cout,...] or MatMul
 * [K,N] weight. With per-input-channel activation scales the kernel would
 * fold them into the weights first (W' = W * s_in), so the round trip
 * quantizes W' and unfolds, which is what the int8 result would see.
 */
function fakeQuantWeight(w: OnnxTensor, op: string, sIn?: Float32Array): OnnxTensor {
  const src = w.data as Float32Array;
  const out = new Float32Array(src.length);
  const channels = op === "MatMul" ? w.dims[1] : w.dims[0];
  const per = src.length / channels;
  const at = op === "MatMul"
    ? (c: number, i: number) => i * w.dims[1] + c
    : (c: number, i: number) => c * per + i;
  // Input channel of element i within output channel c.
  const inCh = op === "MatMul"
    ? (i: number) => i
    : (i: number) => Math.floor(i / (w.dims[2] * w.dims[3]));
  const fold = (c: number, i: number) => (sIn ? sIn[op === "MatMul" ? i : inCh(i)] : 1);
  for (let c = 0; c < channels; c++) {
    let amax = 0;
    for (let i = 0; i < per; i++) amax = Math.max(amax, Math.abs(src[at(c, i)] * fold(c, i)));
    const s = amax / 127 || 1;
    for (let i = 0; i < per; i++) out[at(c, i)] = Math.round(src[at(c, i)] * fold(c, i) / s) * s / fold(c, i);
  }
  return { ...w, data: out };
}

export function quantizeGraph(g: OnnxGraph, ranges: Map<string, Range>, mode: Mode, only?: Set<string>): OnnxGraph {
  if (mode === "fp32") return g;
  const initializers = new Map(g.initializers);
  const nodes: OnnxNode[] = [];
  const dq = new Map<string, string>();
  for (const n of g.nodes) {
    if (!isGemm(n) || (!quantHead && isHead(g, n))) { nodes.push(n); continue; }
    const w = initializers.get(n.input[1])!;
    const x = n.input[0];
    const r = ranges.get(x);
    if (!r && mode !== "w") throw new Error(`no calibration range for ${x}`);
    const chan = mode === "wa-chan" && !(only && !only.has(x));
    const isDepthwise = n.opType === "Conv" && (n.attrs.get("group")?.i ?? 1) > 1;
    // A depthwise weight is per channel already, so folding a per-channel
    // activation scale into it changes nothing after symmetric quantization.
    initializers.set(w.name, fakeQuantWeight(w, n.opType, chan && !isDepthwise ? chanScales(r!)[0] : undefined));
    if (mode === "w" || (only && !only.has(x))) { nodes.push(n); continue; }
    if (!dq.has(x)) {
      const sym = mode === "wa-sym";
      const [scale, zp] = chan
        ? chanScales(r!)
        : sym
        ? [[Math.max(Math.abs(r!.min), Math.abs(r!.max)) / 127], [0]]
        : [[(r!.max - r!.min) / 255], [Math.round(-128 - r!.min / ((r!.max - r!.min) / 255))]];
      initializers.set(`${x}_s`, f32(`${x}_s`, [...scale], scale.length > 1 ? [scale.length] : []));
      initializers.set(`${x}_zp`, f32(`${x}_zp`, [...zp], zp.length > 1 ? [zp.length] : []));
      const attrs = new Map([["axis", { name: "axis", type: 2, i: r!.axis }]]);
      nodes.push({ name: `${x}_q`, opType: "QuantizeLinear", input: [x, `${x}_s`, `${x}_zp`], output: [`${x}_q`], attrs });
      nodes.push({ name: `${x}_dq`, opType: "DequantizeLinear", input: [`${x}_q`, `${x}_s`, `${x}_zp`], output: [`${x}_dq`], attrs });
      dq.set(x, `${x}_dq`);
    }
    nodes.push({ ...n, input: [dq.get(x)!, ...n.input.slice(1)] });
  }
  return { ...g, nodes, initializers };
}

/** Asymmetric int8 scale and zero point per channel from the calibrated bounds. */
function chanScales(r: Range): [Float32Array, Float32Array] {
  const C = r.cmin.length;
  const scale = new Float32Array(C), zp = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    const lo = Math.min(r.cmin[c], 0), hi = Math.max(r.cmax[c], 0);
    scale[c] = Math.max(hi - lo, 1e-6) / 255;
    zp[c] = Math.max(-128, Math.min(127, Math.round(-128 - lo / scale[c])));
  }
  return [scale, zp];
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

const cer = (got: string[], want: string[]) => editDistance(got.join("\n"), want.join("\n")) / want.join("\n").length;

/** Weights must upload before any run, so each configuration gets a fresh arena. */
export async function quantizedOcr(detMode: Mode, recMode: Mode, cal: Calibration, only?: Set<string>): Promise<Ocr> {
  const ocr = await Ocr.create({ ...assets, det: await read("models/det.onnx"), rec: await read("models/rec.onnx") });
  const o = ocr as unknown as Record<string, Session>;
  o.det = new Session(quantizeGraph(detGraph, cal.det, detMode), ocr.arena);
  o.rec = new Session(quantizeGraph(recGraph, cal.rec, recMode, only), ocr.arena);
  return ocr;
}

export async function evaluate(detMode: Mode, recMode: Mode, cal: Calibration, only?: Set<string>) {
  const ocr = await quantizedOcr(detMode, recMode, cal, only);
  const all = ocr.text(receipt, DEFAULT_DETECT, { minConfidence: 0, dropSeparators: false }).split("\n");
  const kept = ocr.text(receipt).split("\n");
  ocr.destroy();
  const same = all.filter((l, i) => l === reference[i]).length;
  return { all, same, cerRef: cer(all, reference), cerGt: cer(kept, gt) };
}

if (import.meta.main) {
const cal = await calibrate();
console.log(`calibrated on ${calibImages.join(",")} at p${pct}: det ${cal.det.size} tensors, rec ${cal.rec.size} tensors${quantHead ? ", head quantized" : ""}`);

if (sweep) {
  // One activation at a time, weights already int8, against the weights-only run.
  const base = await evaluate("fp32", "w", cal);
  console.log(`weights-only rec: ${base.same}/${reference.length} same, CER/gt ${(100 * base.cerGt).toFixed(2)}%`);
  for (const n of recGraph.nodes) {
    if (!isGemm(n) || isHead(recGraph, n)) continue;
    const x = n.input[0];
    const e = await evaluate("fp32", "wa-asym", cal, new Set([x]));
    const diffs = e.all.filter((l, i) => l !== base.all[i]).length;
    const w = recGraph.initializers.get(n.input[1])!.dims.join("x");
    const r = cal.rec.get(x)!;
    const flag = e.cerGt > base.cerGt ? " <-- worse" : "";
    console.log(`${n.opType.padEnd(6)} ${w.padEnd(12)} ${x.padEnd(32)} [${r.min.toFixed(2)}, ${r.max.toFixed(2)}]  lines changed ${diffs}  CER/gt ${(100 * e.cerGt).toFixed(2)}%${flag}`);
  }
  process.exit(0);
}
const configs: [Mode, Mode][] = [
  ["fp32", "fp32"], ["w", "w"], ["wa-sym", "wa-sym"], ["wa-asym", "wa-asym"],
  ["wa-asym", "fp32"], ["fp32", "wa-asym"], ["wa-chan", "wa-chan"], ["fp32", "wa-chan"],
];
console.log("\ndet      rec      lines  same/ref  CER/ref  CER/gt");
const base = await evaluate("fp32", "fp32", cal);
for (const [d, r] of configs) {
  const e = d === "fp32" && r === "fp32" ? base : await evaluate(d, r, cal);
  console.log(`${d.padEnd(8)} ${r.padEnd(8)} ${String(e.all.length).padStart(5)}  ${String(e.same).padStart(4)}/${reference.length}  ${(100 * e.cerRef).toFixed(2).padStart(6)}%  ${(100 * e.cerGt).toFixed(2).padStart(6)}%`);
  if (e !== base) {
    for (let i = 0; i < Math.max(e.all.length, base.all.length); i++) {
      if (e.all[i] !== base.all[i]) console.log(`    ${i}: fp32 "${base.all[i]}"  ->  "${e.all[i]}"`);
    }
  }
}
}
