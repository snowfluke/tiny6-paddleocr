// Recognition: crop -> 48px tall grayscale tensor -> CTC greedy decode.
// The decode rules (blank index, dictionary padding, gap-space injection,
// fullwidth folding) are ported from ppu-paddle-ocr so output can be diffed.

import type { Session } from "../runtime/graph.ts";
import type { RGBA } from "../image/png.ts";
import { crop, cropRotated, resize } from "../image/ops.ts";
import type { Box } from "./detect.ts";

export const BLANK_INDEX = 0;
export const UNK_TOKEN = "<unk>";
export const MIN_CROP_WIDTH = 8;
export const REC_HEIGHT = 48;

const GAP_QUANTA_CROSS_CLASS = 1.5;
const GAP_QUANTA_SAME_CLASS = 2.5;
const FULLWIDTH_OFFSET = 0xfee0;
const CJK_PATTERN = /[\u2E80-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]/;

function charClass(c: string): number {
  if (/\p{L}/u.test(c)) return 0;
  if (/\p{N}/u.test(c)) return 1;
  return 2;
}

/** CTC under-emits spaces; a gap much wider than the glyph pitch is one. */
export function injectGapSpaces(chars: string[], positions: number[]): void {
  if (chars.length < 4) return;
  const deltas: number[] = [];
  for (let i = 1; i < positions.length; i++) deltas.push(positions[i] - positions[i - 1]);
  const sorted = [...deltas].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  if (median <= 0) return;
  const quantum = sorted.find((d) => d > 0) ?? 0;
  if (quantum <= 0) return;

  for (let i = chars.length - 1; i >= 1; i--) {
    const prev = positions[i - 1];
    const curr = positions[i];
    const k = charClass(chars[i]) === charClass(chars[i - 1])
      ? GAP_QUANTA_SAME_CLASS
      : GAP_QUANTA_CROSS_CLASS;
    // Repeated characters need a blank between them, so their gap is
    // structurally wider and must not count as a space.
    if (
      curr - prev > median + k * quantum &&
      chars[i] !== " " && chars[i - 1] !== " " && chars[i] !== chars[i - 1]
    ) {
      chars.splice(i, 0, " ");
      positions.splice(i, 0, (prev + curr) / 2);
    }
  }
}

export function refineDecodedChars(chars: string[], positions: number[]): void {
  for (let i = chars.length - 1; i >= 1; i--) {
    if (chars[i] === " " && chars[i - 1] === " ") {
      chars.splice(i, 1);
      positions.splice(i, 1);
    }
  }
  if (CJK_PATTERN.test(chars.join(""))) return;
  for (let i = 0; i < chars.length; i++) {
    const code = chars[i].codePointAt(0) ?? 0;
    if (code >= 0xff01 && code <= 0xff5e) chars[i] = String.fromCodePoint(code - FULLWIDTH_OFFSET);
    else if (code === 0x3000) chars[i] = " ";
  }
}

export type Decoded = { text: string; confidence: number };

export function ctcGreedyDecode(
  logits: Float32Array,
  seqLen: number,
  numClasses: number,
  dict: string[],
): Decoded {
  const lastDictIndex = dict.length - 1;
  const emitted: string[] = [];
  const positions: number[] = [];
  let lastIndex = -1;
  let confSum = 0;
  let confCount = 0;

  for (let t = 0; t < seqLen; t++) {
    const base = t * numClasses;
    let maxProb = logits[base];
    let maxIndex = 0;
    for (let c = 1; c < numClasses; c++) {
      if (logits[base + c] > maxProb) {
        maxProb = logits[base + c];
        maxIndex = c;
      }
    }
    if (maxIndex === BLANK_INDEX || maxIndex === lastIndex) {
      lastIndex = maxIndex;
      continue;
    }
    if (maxIndex < dict.length) {
      const char = dict[maxIndex];
      if (maxIndex === lastDictIndex && char === UNK_TOKEN) {
        lastIndex = maxIndex;
        continue;
      }
      emitted.push(maxIndex === lastDictIndex ? " " : char);
      confSum += maxProb;
      confCount++;
      positions.push((t + 0.5) / seqLen);
    }
    lastIndex = maxIndex;
  }

  injectGapSpaces(emitted, positions);
  refineDecodedChars(emitted, positions);
  return { text: emitted.join(""), confidence: confCount ? confSum / confCount : 0 };
}

/** The model reads the red channel replicated across three planes. */
/**
 * Crop widths are padded up to this. An odd width makes every downstream
 * convolution's column count ragged, and the GEMM's ragged tail is scalar:
 * measured 16 -> 17 px at 2.0 -> 2.9 ms, 64 -> 65 at 2.9 -> 4.1. Wider alignment
 * measured no faster end to end, because the wall is set by the widest crops
 * and those lose only 8% to raggedness. The padding
 * is zero in normalised space, mid-grey, which is what PaddleOCR trains with.
 */
export const REC_WIDTH_ALIGN = 8;

export function cropToTensor(img: RGBA): { data: Float32Array; width: number } {
  const aspect = img.width / img.height;
  const w = Math.max(MIN_CROP_WIDTH, Math.round(REC_HEIGHT * aspect));
  const wp = Math.ceil(w / REC_WIDTH_ALIGN) * REC_WIDTH_ALIGN;
  const r = resize(img, w, REC_HEIGHT);
  const n = wp * REC_HEIGHT;
  const t = new Float32Array(3 * n);
  for (let y = 0; y < REC_HEIGHT; y++) {
    for (let x = 0, p = y * w * 4, o = y * wp; x < w; x++, p += 4, o++) t[o] = r.data[p] / 127.5 - 1;
  }
  t.copyWithin(n, 0, n);
  t.copyWithin(2 * n, 0, n);
  return { data: t, width: wp };
}

/**
 * Cut a box out of the page. A fitted rect samples straight to the model's
 * input height, so the rotation and the scale to 48px happen in one pass.
 * Split out from recognition so a worker can be handed the pixels alone.
 */
export function cropForBox(img: RGBA, box: Box): RGBA {
  if (!box.rect) return crop(img, box.x, box.y, box.width, box.height);
  const { cx, cy, width, height, angle } = box.rect;
  const w = Math.max(MIN_CROP_WIDTH, Math.round(REC_HEIGHT * (width / height)));
  return cropRotated(img, cx, cy, width, height, angle, w, REC_HEIGHT);
}

export function recognizeCrop(session: Session, patch: RGBA, dict: string[]): Decoded {
  const { data, width } = cropToTensor(patch);
  const inputName = session.graph.inputs[0].name;
  const out = session.run({ [inputName]: { dims: [1, 3, REC_HEIGHT, width], data } });
  const t = [...out.values()][0];
  const [, seqLen, numClasses] = t.dims;
  // PaddleOCR dictionaries omit the blank slot the model reserves at index 0.
  const padded = dict.length === numClasses - 1 ? ["", ...dict] : dict;
  return ctcGreedyDecode(t.data, seqLen, numClasses, padded);
}

export function recognizeBox(session: Session, img: RGBA, box: Box, dict: string[]): Decoded {
  return recognizeCrop(session, cropForBox(img, box), dict);
}

/**
 * No trimming: entry 0 is the CTC blank and the trailing empty line is the
 * space class, so a dictionary file's line count already equals the model's
 * class count. Dropping the last line shifts every character by one.
 */
export function parseDictionary(text: string): string[] {
  return text.split(/\r?\n/);
}

/** Crops whose widths are within this factor share a batch. */
const BATCH_WIDTH_SPREAD = 1.15;
const MAX_BATCH = 8;

/**
 * Recognise several crops per graph run.
 *
 * Each run costs about 4 ms before any arithmetic happens - 219 nodes of
 * dispatch, allocation and bookkeeping - so short crops spend more time on
 * overhead than on the model. A batch pays that once.
 *
 * Crops are sorted by width and grouped only with near neighbours, because
 * every member is padded to the batch's widest and that padding is wasted
 * work. Padding replicates the crop's last column so the background carries
 * on, and each row is decoded over only the timesteps its real width covers,
 * which keeps the padding out of the text.
 *
 * MEASURED SLOWER on this model and not used by default: 0.87x at the
 * tightest grouping, 0.72x at the loosest, on a 28-crop receipt. A batch
 * makes every intermediate N times larger, and losing cache locality costs
 * more than the dispatch it saves. Kept because it is correct and because a
 * runtime with heavier call overhead than Bun may see the opposite.
 */
export function recognizeBatch(session: Session, crops: RGBA[], dict: string[]): Decoded[] {
  const prepared = crops.map((c, index) => ({ index, ...cropToTensor(c) }));
  prepared.sort((a, b) => a.width - b.width);

  const out = new Array<Decoded>(crops.length);
  const inputName = session.graph.inputs[0].name;

  for (let i = 0; i < prepared.length;) {
    const first = prepared[i];
    let end = i + 1;
    while (
      end < prepared.length &&
      end - i < MAX_BATCH &&
      prepared[end].width <= first.width * BATCH_WIDTH_SPREAD
    ) end++;

    const group = prepared.slice(i, end);
    const maxW = group[group.length - 1].width;
    const plane = REC_HEIGHT * maxW;
    const data = new Float32Array(group.length * 3 * plane);

    group.forEach((g, row) => {
      const base = row * 3 * plane;
      for (let y = 0; y < REC_HEIGHT; y++) {
        const src = y * g.width;
        const dst = base + y * maxW;
        data.set(g.data.subarray(src, src + g.width), dst);
        // Extend the last column rather than padding with a constant, so a
        // dark background does not turn into a bright block.
        const edge = g.data[src + g.width - 1];
        for (let x = g.width; x < maxW; x++) data[dst + x] = edge;
      }
      data.copyWithin(base + plane, base, base + plane);
      data.copyWithin(base + 2 * plane, base, base + plane);
    });

    const t = [...session.run({ [inputName]: { dims: [group.length, 3, REC_HEIGHT, maxW], data } }).values()][0];
    const [, seqLen, numClasses] = t.dims;
    const padded = dict.length === numClasses - 1 ? ["", ...dict] : dict;

    group.forEach((g, row) => {
      const used = Math.max(1, Math.round((seqLen * g.width) / maxW));
      const rowData = t.data.subarray(row * seqLen * numClasses, (row * seqLen + used) * numClasses);
      out[g.index] = ctcGreedyDecode(rowData, used, numClasses, padded);
    });
    i = end;
  }
  return out;
}
