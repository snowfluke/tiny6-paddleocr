import { expect, test } from "bun:test";
import { defaultRecWorkers, isSeparator, Ocr } from "../src/ocr.ts";
import { DEFAULT_DETECT, REFERENCE_BINARIZE } from "../src/pipeline/detect.ts";
import { convexHull, mergeOverlapping, minAreaRect } from "../src/pipeline/boxes.ts";
import { makeRecWorker } from "../src/node.ts";
import { decodeImage, decodeJpeg } from "../src/image/jpeg.ts";
import { cropForBox, recognizeBatch } from "../src/pipeline/recognize.ts";
import { decodePng } from "../src/image/png.ts";
import { parseOnnx } from "../src/onnx/parse.ts";
import { Session } from "../src/runtime/graph.ts";
import { dropIdentity, foldBatchNorm, fuseConvEpilogue, fuseGelu } from "../src/runtime/fuse.ts";
import { loadKernels, loadKernelsThreaded } from "../src/wasm/backend.ts";
import { compare, unpack } from "../tools/check.ts";
import { sharedWasm, wasm } from "./kernels.ts";

const read = async (p: string) => new Uint8Array(await Bun.file(p).arrayBuffer());

// The goldens are ~100 MB of ORT output, too big to commit, so a fresh clone
// regenerates them here. Needs onnxruntime-node, which is a dev dependency.
if (!(await Bun.file("test/golden/det.golden.bin").exists())) {
  console.log("generating ORT goldens (one time, ~1 min)...");
  const r = Bun.spawnSync(["bun", "run", "golden"], { stdout: "inherit", stderr: "inherit" });
  if (r.exitCode !== 0) throw new Error("golden generation failed");
}
if (!(await Bun.file("src/wasm/kernels.wasm").exists())) {
  const r = Bun.spawnSync(["bun", "tools/build-wasm.ts"], { stdout: "inherit", stderr: "inherit" });
  if (r.exitCode !== 0) throw new Error("wasm build failed");
}

/**
 * Every intermediate tensor is diffed against values ORT produced once and
 * committed. This is what catches a silently wrong op: a bad stride or pad
 * still yields plausible-looking text, but never matching activations.
 */
for (const [which, dims] of [["det", [1, 3, 256, 256]], ["rec", [1, 3, 48, 320]]] as const) {
  for (const backend of ["ts", "wasm"] as const) {
    test(`${which} matches ORT golden tensors (${backend})`, async () => {
      const g = parseOnnx(await read(`models/${which}.onnx`));
      const golden = unpack(await read(`test/golden/${which}.golden.bin`));
      const input = await read(`test/golden/${which}.input.bin`);
      const arena = backend === "wasm" ? await loadKernels(wasm) : undefined;
      // Unfused, so every intermediate the golden holds still exists to diff.
      // The fused graph is covered by the equivalence test below.
      const s = new Session(g, arena, { fuse: false });

      let checked = 0;
      const bad: string[] = [];
      s.run(
        { [g.inputs[0].name]: { dims: [...dims], data: new Float32Array(input.buffer) } },
        {
          onNode: (node, outs) => {
            for (let i = 0; i < node.output.length; i++) {
              const want = golden.get(node.output[i]);
              if (!want) continue;
              checked++;
              const r = compare(outs[i], want);
              if (!r.shapeOk || r.maxAbs > 2e-3) {
                bad.push(`${node.opType} ${node.output[i]} maxAbs=${r.maxAbs.toExponential(2)}`);
              }
            }
          },
        },
      );
      expect(bad).toEqual([]);
      expect(checked).toBeGreaterThan(200);
    }, 120_000);
  }
}

/**
 * The goldens above run unfused, so this is what covers the fused graph: no
 * fusion may move the output. It also pins each pattern actually firing, since
 * a matcher that silently matches nothing would pass every other test.
 */
const FUSIONS = [
  // model, input dims, batchnorms folded, gelus folded, conv biases folded, relus folded, residuals
  ["det", [1, 3, 256, 256], 0, 13, 0, 19, 10],
  ["rec", [1, 3, 48, 320], 2, 10, 33, 3, 7],
] as const;

for (const [which, dims, bns, gelus, biases, relus, residuals] of FUSIONS) {
  test(`fusing leaves ${which} output unchanged`, async () => {
    const g = parseOnnx(await read(`models/${which}.onnx`));
    const afterBn = foldBatchNorm(dropIdentity(g).graph);
    expect(afterBn.folded).toBe(bns);
    const afterGelu = fuseGelu(afterBn.graph);
    expect(afterGelu.fused).toBe(gelus);
    const epilogue = fuseConvEpilogue(afterGelu.graph);
    expect([epilogue.bias, epilogue.act, epilogue.residual]).toEqual([biases, relus, residuals]);

    const n = (dims as readonly number[]).reduce((a, b) => a * b, 1);
    const data = new Float32Array(n);
    let seed = 7;
    for (let i = 0; i < n; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      data[i] = (seed / 0xffffffff) * 2 - 1;
    }
    const feeds = { [g.inputs[0].name]: { dims: [...dims], data } };

    const run = async (fuse: boolean) => {
      const arena = await loadKernels(wasm);
      const out = [...new Session(g, arena, { fuse }).run(feeds).values()][0];
      arena.destroy();
      return out;
    };
    const plain = await run(false);
    const fused = await run(true);
    let max = 0;
    for (let i = 0; i < plain.data.length; i++) {
      max = Math.max(max, Math.abs(plain.data[i] - fused.data[i]));
    }
    expect(max).toBeLessThan(2e-3);
  }, 120_000);
}

test("PNG decoder reads both colour types", async () => {
  const rgb = await decodePng(await read("test/images/receipt.png"));
  expect([rgb.width, rgb.height]).toEqual([720, 1280]);
  expect(rgb.data.length).toBe(720 * 1280 * 4);

  const rgba = await decodePng(await read("test/images/tilted.png"));
  expect([rgba.width, rgba.height]).toEqual([475, 179]);
});


const makeOcr = async (threads?: number, recWorkers?: number, int8 = false) =>
  Ocr.create({
    det: await read("models/det.onnx"),
    rec: await read("models/rec.onnx"),
    dict: await Bun.file("models/dict.txt").text(),
    wasm,
    wasmShared: sharedWasm,
    threads,
    recWorkers,
    makeRecWorker: recWorkers && recWorkers > 1 ? makeRecWorker : undefined,
    ...(int8 ? { detCalib: await Bun.file("models/det.calib.json").text(), recCalib: await Bun.file("models/rec.calib.json").text() } : {}),
  });

test("reads short upright text exactly", async () => {
  const ocr = await makeOcr();
  const lines = ocr.recognize(await decodePng(await read("test/images/tilted.png")));
  expect(lines[0].text).toBe("Hello, mom!");
  ocr.destroy();
}, 120_000);

/**
 * Text equality is too coarse: a kernel that raced across workers corrupted
 * a handful of activations in a 5x5 depthwise layer and the text still came
 * out right most runs. Every node's output must match the single-threaded
 * run exactly, and repeatedly, since a race only shows on some schedules.
 */
for (const threads of [4, 8]) {
test(`worker threads are bit-identical at every node (${threads} threads)`, async () => {
  const g = parseOnnx(await read("models/det.onnx"));
  const dims = [1, 3, 256, 320];
  const n = dims.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  let seed = 3;
  for (let i = 0; i < n; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    data[i] = (seed / 0xffffffff) * 2 - 1;
  }
  const feeds = { [g.inputs[0].name]: { dims, data } };
  const one = new Session(g, await loadKernels(wasm));
  const ref = new Map<string, Float32Array>();
  one.run(feeds, { onNode: (node, outs) => node.output.forEach((o, i) => ref.set(o, outs[i].data)) });

  const arena = await loadKernelsThreaded(sharedWasm, threads);
  const four = new Session(g, arena);
  const bad: string[] = [];
  for (let rep = 0; rep < 6; rep++) {
    four.run(feeds, {
      onNode: (node, outs) => node.output.forEach((o, i) => {
        const a = ref.get(o)!;
        const b = outs[i].data;
        for (let j = 0; j < a.length; j++) {
          if (a[j] !== b[j]) {
            bad.push(`run ${rep}: ${node.opType} ${o} differs at ${j}`);
            return;
          }
        }
      }),
    });
  }
  arena.destroy();
  expect(bad).toEqual([]);
}, 120_000);
}

test("worker threads change nothing but the wall clock", async () => {
  const img = await decodePng(await read("test/images/receipt.png"));
  const single = await makeOcr(1);
  const many = await makeOcr(4);
  expect(many.arena.threads).toBeGreaterThan(1);
  expect(many.text(img)).toBe(single.text(img));
  single.destroy();
  many.destroy();
}, 180_000);

/**
 * Checked against what ppu-paddle-ocr (ORT) produces on the same file.
 * 13 of 18 lines are byte-identical. The five that differ: two are barcode
 * noise both implementations read as garbage, and three differ by one
 * character because the crops differ by a pixel or two - independent box
 * and resize code, not a decode bug.
 */
test("receipt matches the ORT reference on most lines", async () => {
  const ocr = await makeOcr();
  const img = await decodePng(await read("test/images/receipt.png"));
  // The reference was captured with filtering off, so compare unfiltered.
  const got = ocr.text(img, DEFAULT_DETECT, { minConfidence: 0, dropSeparators: false }).split("\n");
  const want = (await Bun.file("test/images/receipt-reference.txt").text()).trimEnd().split("\n");

  expect(got.length).toBe(want.length);
  const same = got.filter((l, i) => l === want[i]).length;
  expect(same).toBeGreaterThanOrEqual(13);
  ocr.destroy();
}, 180_000);

/**
 * The reference transcript above is what pins the default: the reference path
 * rounds the probability to a byte, so its effective cut is one 8-bit step.
 * Raising the cut drops the low-probability regions that step lets through -
 * on SROIE that is +1.7 points of character similarity for 5% more time, with
 * the median box count unchanged. It must never invent regions, and the
 * default must stay the reference or every compared transcript shifts.
 */
test("raising binarizeThreshold drops regions and never adds them", async () => {
  const ocr = await makeOcr(4);
  const img = await decodePng(await read("test/images/receipt.png"));
  const at960 = { ...DEFAULT_DETECT, maxSideLength: 960 };

  const reference = ocr.detect(img, at960).boxes.length;
  const raised = ocr.detect(img, { ...at960, binarizeThreshold: 0.5 }).boxes.length;

  expect(raised).toBeLessThanOrEqual(reference);
  expect(DEFAULT_DETECT.binarizeThreshold).toBe(REFERENCE_BINARIZE);
  ocr.destroy();
}, 180_000);

test("the confidence filter drops barcode noise and keeps the text", async () => {
  const ocr = await makeOcr();
  const img = await decodePng(await read("test/images/receipt.png"));
  const all = ocr.recognize(img, DEFAULT_DETECT, { minConfidence: 0, dropSeparators: false });
  const kept = ocr.recognize(img);

  expect(all.length).toBeGreaterThan(kept.length);
  expect(kept.map((l) => l.text)).toContain("NPWP : 01.336.238.9-054.000");
  expect(kept.map((l) => l.text)).toContain("SMS/WA:081110640888");
  for (const l of kept) expect(l.confidence).toBeGreaterThanOrEqual(0.5);
  ocr.destroy();
}, 180_000);

test("minimum-area rect recovers a known rotation", () => {
  // A 40x10 rectangle rotated 30 degrees about the origin.
  const angle = Math.PI / 6;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const pts: number[] = [];
  for (const [x, y] of [[-20, -5], [20, -5], [20, 5], [-20, 5]]) {
    pts.push(Math.round(100 + x * c - y * s), Math.round(100 + x * s + y * c));
  }
  const r = minAreaRect(convexHull(Int32Array.from(pts), 4))!;
  const long = Math.max(r.width, r.height);
  const short = Math.min(r.width, r.height);
  // The corners are rounded to whole pixels, so allow a pixel of slack.
  expect(long).toBeGreaterThan(39);
  expect(long).toBeLessThan(41.5);
  expect(short).toBeGreaterThan(8.5);
  expect(short).toBeLessThan(11);
  expect(r.cx).toBeCloseTo(100, 0);
  expect(r.cy).toBeCloseTo(100, 0);
  const fitted = Math.abs(r.width >= r.height ? r.angle : r.angle + Math.PI / 2);
  expect(fitted).toBeCloseTo(angle, 1);
});

test("overlapping fragments merge, separate words do not", () => {
  const stacked = [
    { x: 10, y: 10, width: 20, height: 20 },
    { x: 14, y: 12, width: 20, height: 20 },
    { x: 18, y: 14, width: 20, height: 20 },
  ];
  expect(mergeOverlapping(stacked, 0.5)).toHaveLength(1);

  const words = [
    { x: 0, y: 0, width: 30, height: 10 },
    { x: 32, y: 0, width: 30, height: 10 },
  ];
  expect(mergeOverlapping(words, 0.5)).toHaveLength(2);
});

test("rotated boxes read tilted text the upright path garbles", async () => {
  const ocr = await makeOcr();
  const img = await decodePng(await read("test/images/tilted.png"));
  const rotated = ocr.recognize(img, { ...DEFAULT_DETECT, rotated: true });
  expect(rotated.map((l) => l.text)).toContain("I am slightly tilted hehe");

  const upright = ocr.recognize(img, DEFAULT_DETECT, { minConfidence: 0 });
  expect(upright.map((l) => l.text)).not.toContain("I am slightly tilted hehe");
  ocr.destroy();
}, 120_000);

test("the recognition pool returns exactly what the serial path does", async () => {
  const img = await decodePng(await read("test/images/receipt.png"));
  const serial = await makeOcr(1, 0);
  const pooled = await makeOcr(1, 4);
  expect(pooled.recWorkers).toBe(4);

  const { boxes } = serial.detect(img);
  const want = serial.recognizeBoxes(img, boxes, { minConfidence: 0 });
  const got = await pooled.recognizeBoxesAsync(img, boxes, { minConfidence: 0 });

  expect(got.map((l) => l.text)).toEqual(want.map((l) => l.text));
  serial.destroy();
  pooled.destroy();
}, 180_000);

/**
 * The pool only pays from a few crops up: below that recognizeBoxesAsync
 * falls back to the shared arena instead of one thread per crop. Both sides
 * of that boundary must still return what the serial path returns.
 */
test("the pool boundary changes nothing but the wall clock", async () => {
  const img = await decodePng(await read("test/images/receipt.png"));
  const serial = await makeOcr(1, 0);
  const pooled = await makeOcr(4, 4);
  expect(pooled.recWorkers).toBe(4);

  const { boxes } = serial.detect(img);
  for (const n of [1, 3, 4, 8]) {
    const slice = boxes.slice(0, n);
    if (slice.length < n) continue;
    const want = serial.recognizeBoxes(img, slice, { minConfidence: 0 });
    const got = await pooled.recognizeBoxesAsync(img, slice, { minConfidence: 0 });
    expect(got.map((l) => l.text)).toEqual(want.map((l) => l.text));
  }
  serial.destroy();
  pooled.destroy();
}, 180_000);

test("decodes progressive and baseline JPEG to the same OCR text", async () => {
  const ocr = await makeOcr();
  const fromPng = ocr.text(await decodePng(await read("test/images/receipt.png")));
  const fromJpeg = ocr.text(decodeJpeg(await read("test/images/receipt.jpg")));
  expect(fromJpeg).toBe(fromPng);
  ocr.destroy();
}, 180_000);

test("the JPEG decoder lands close to the lossless original", async () => {
  const png = await decodePng(await read("test/images/receipt.png"));
  const jpg = decodeJpeg(await read("test/images/receipt.jpg"));
  expect([jpg.width, jpg.height]).toEqual([png.width, png.height]);

  let sum = 0;
  for (let i = 0; i < jpg.data.length; i++) sum += Math.abs(jpg.data[i] - png.data[i]);
  // The PNG was produced from this JPEG, so the gap is chroma upsampling and
  // IDCT rounding, not decode error.
  expect(sum / jpg.data.length).toBeLessThan(2);
});

test("the recognition pool is not started where it would lose", () => {
  // Two workers measured slower than inline; six is the ceiling that pays.
  expect([1, 2, 3, 4, 8, 16].map((cores) => defaultRecWorkers(cores))).toEqual([0, 0, 3, 4, 6, 6]);
});

test("a scaled decode halves the raster and preserves its levels", async () => {
  const buf = await read("test/images/receipt.jpg");
  const full = decodeJpeg(buf);
  const half = decodeJpeg(buf, 2);

  expect([half.width, half.height]).toEqual([full.width / 2, full.height / 2]);

  // The low-frequency corner must reconstruct the same block means. A wrong
  // IDCT prefactor scales every sample instead (measured as -15.8pp of OCR
  // similarity when the divisor was taken as n/2 rather than 4), and reading
  // the flat first n*n coefficients instead of the 2D corner loses the
  // vertical frequencies (-23.3pp). Both show up here as a channel mean that
  // has drifted, which the OCR score would only reveal much later.
  const channelMean = (img: { data: Uint8Array }, c: number) => {
    let s = 0;
    for (let p = c; p < img.data.length; p += 4) s += img.data[p];
    return s / (img.data.length / 4);
  };
  for (let c = 0; c < 3; c++) {
    expect(channelMean(half, c)).toBeCloseTo(channelMean(full, c), 0);
    expect(Math.abs(channelMean(half, c) - channelMean(full, c))).toBeLessThan(2);
  }

  // And it has to agree with a plain box downsample of the full-resolution
  // decode, which is the resample it replaces.
  let sum = 0;
  const w = half.width;
  const h = half.height;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const sx = Math.min(full.width - 1, x * 2 + dx);
            const sy = Math.min(full.height - 1, y * 2 + dy);
            s += full.data[(sy * full.width + sx) * 4 + c];
          }
        }
        sum += Math.abs(s / 4 - half.data[(y * w + x) * 4 + c]);
      }
    }
  }
  expect(sum / (w * h * 3)).toBeLessThan(5);
});

test("the decode raster is bit-identical to what it has always been", async () => {
  // The inverse transform is the hot loop of the decoder and it is written for
  // speed, with a shortcut for blocks whose only non-zero coefficient is DC
  // (they reconstruct to a flat patch, and 91% of the blocks in these receipts
  // are flat). Nothing about that is allowed to move a pixel: the shortcut
  // reproduces the general path's float ops exactly, because rounding an
  // algebraically equal but differently-ordered expression moves Math.round
  // across its .5 boundary for a small share of blocks.
  //
  // These hashes were taken before the shortcut existed and must not change.
  // A failure here means decoded pixels moved, which the OCR score would only
  // report as a fraction of a percentage point much later.
  const fnv = (d: Uint8Array) => {
    let h = 0x811c9dc5;
    for (let i = 0; i < d.length; i++) {
      h ^= d[i];
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
  };
  const buf = await read("test/images/receipt.jpg");
  const full = decodeJpeg(buf);
  const half = decodeJpeg(buf, 2);
  expect([full.width, full.height, full.data.length]).toEqual([720, 1280, 3686400]);
  expect([half.width, half.height, half.data.length]).toEqual([360, 640, 921600]);
  expect(fnv(full.data)).toBe("92f8be10");
  expect(fnv(half.data)).toBe("bfa9ed72");
});

test("an unsupported downscale is rejected rather than silently ignored", async () => {
  const buf = await read("test/images/receipt.jpg");
  expect(() => decodeJpeg(buf, 3 as 1)).toThrow(/downscale/);
  const png = await read("test/images/receipt.png");
  await expect(decodeImage(png, 2)).rejects.toThrow(/JPEG/);
});

test("rule characters are dropped, text with letters or digits is not", () => {
  for (const t of ["+-", "----", "===", "...", "|", "_ _ _"]) expect(isSeparator(t)).toBe(true);
  for (const t of ["232:", "PPN ( 0)", "A", "0", "Tunai 44,900"]) expect(isSeparator(t)).toBe(false);
});

for (const int8 of [false, true]) {
test(`batched recognition agrees with one crop at a time${int8 ? " (int8)" : ""}`, async () => {
  const ocr = await makeOcr(1, 0, int8);
  if (int8 && !ocr.int8) return; // engine without a signed dot product
  const img = await decodePng(await read("test/images/tilted.png"));
  const { boxes } = ocr.detect(img);
  const crops = boxes.map((b) => cropForBox(img, b));
  const serial = ocr.recognizeBoxes(img, boxes, { minConfidence: 0 }).map((l) => l.text);
  const batched = recognizeBatch(ocr.recSession, crops, ocr.dictionary).map((d) => d.text)
    .filter((t) => t.trim());
  expect(batched).toEqual(serial);
  ocr.destroy();
}, 120_000);
}
