// Detection: preprocess -> DB probability map -> boxes.
// Constants match ppu-paddle-ocr so results can be diffed against it.

import type { Session } from "../runtime/graph.ts";
import type { RGBA } from "../image/png.ts";
import { padTo, resize } from "../image/ops.ts";
import { connectedRegions } from "./components.ts";
import { convexHull, minAreaRect, mergeOverlapping, type RotatedRect } from "./boxes.ts";

export type Box = {
  x: number;
  y: number;
  width: number;
  height: number;
  /**
   * Set when `rotated` detection is on. x/y/width/height stay the upright
   * bounding box, for reading order and drawing; recognition samples this.
   */
  rect?: RotatedRect;
};

export type DetectOptions = {
  mean: [number, number, number];
  stdDeviation: [number, number, number];
  maxSideLength: number | "auto";
  minimumAreaThreshold: number;
  paddingVertical: number;
  paddingHorizontal: number;
  /**
   * Collapse boxes whose intersection covers this much of the smaller one.
   * 0 disables merging. Logos and barcodes shatter into overlapping
   * fragments; merging them costs one recognition pass instead of twenty.
   */
  mergeOverlap: number;
  /**
   * Probability a pixel must reach to count as text. The default reproduces
   * the reference path exactly: cv.findContours counts any nonzero pixel and
   * the reference rounds the probability to a byte first, so its effective
   * cut is one 8-bit step, 0.5/255. That is orders below the 0.3 a DB
   * post-process is usually thresholded at, which is why the detector also
   * fires on logos, rules and barcodes and the recogniser has to filter them
   * out. Raising it drops those regions before recognition instead.
   */
  binarizeThreshold?: number;
  /**
   * Fit each region with a minimum-area rectangle instead of an upright box,
   * and straighten the crop before recognition. Off by default: it changes
   * every crop slightly, and upright pages do not need it.
   */
  rotated: boolean;
};

/** One 8-bit step: the cut the reference path's byte round-trip produces. */
export const REFERENCE_BINARIZE = 0.5 / 255;

export const DEFAULT_DETECT: DetectOptions = {
  mean: [0.485, 0.456, 0.406],
  stdDeviation: [0.229, 0.224, 0.225],
  maxSideLength: "auto",
  minimumAreaThreshold: 20,
  paddingVertical: 0.4,
  paddingHorizontal: 0.6,
  mergeOverlap: 0.5,
  binarizeThreshold: REFERENCE_BINARIZE,
  rotated: false,
};

export function resolveMaxSideLength(maxSideLength: number | "auto", longestSide: number): number {
  if (maxSideLength !== "auto") return maxSideLength;
  return Math.min(1920, Math.max(960, Math.round((longestSide * 0.75) / 32) * 32));
}

function imageToTensor(img: RGBA, mean: number[], std: number[]): Float32Array {
  const n = img.width * img.height;
  const t = new Float32Array(3 * n);
  const scale = [1 / (255 * std[0]), 1 / (255 * std[1]), 1 / (255 * std[2])];
  const shift = [mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]];
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    t[i] = img.data[p] * scale[0] - shift[0];
    t[n + i] = img.data[p + 1] * scale[1] - shift[1];
    t[2 * n + i] = img.data[p + 2] * scale[2] - shift[2];
  }
  return t;
}

export function detect(session: Session, img: RGBA, opts: DetectOptions = DEFAULT_DETECT) {
  const maxSide = resolveMaxSideLength(opts.maxSideLength, Math.max(img.width, img.height));
  let ratio = 1;
  let rw = img.width;
  let rh = img.height;
  if (Math.max(rw, rh) > maxSide) {
    ratio = maxSide / Math.max(rw, rh);
    rw = Math.round(rw * ratio);
    rh = Math.round(rh * ratio);
  }
  // The detector's downsampling path needs both sides on a 32-pixel grid.
  const width = Math.ceil(rw / 32) * 32;
  const height = Math.ceil(rh / 32) * 32;
  const padded = padTo(resize(img, rw, rh), width, height);

  const tensor = imageToTensor(padded, opts.mean, opts.stdDeviation);
  const inputName = session.graph.inputs[0].name;
  const out = session.run({ [inputName]: { dims: [1, 3, height, width], data: tensor } });
  const prob = [...out.values()][0].data;

  // cv.findContours counts any nonzero pixel, and the reference path rounds
  // the probability to a byte first, so the effective cut is p >= 0.5/255.
  // Raise binarizeThreshold to drop the low-probability regions that cut lets
  // through; DEFAULT_DETECT keeps the reference's value.
  const mask = new Uint8Array(width * height);
  const cut = opts.binarizeThreshold ?? REFERENCE_BINARIZE;
  for (let i = 0; i < mask.length; i++) mask[i] = prob[i] >= cut ? 1 : 0;

  const boxes: Box[] = [];
  for (const r of connectedRegions(mask, width, height, opts.rotated)) {
    const bw = r.x1 - r.x0 + 1;
    const bh = r.y1 - r.y0 + 1;
    if (bw * bh <= opts.minimumAreaThreshold) continue;

    const padV = Math.round(bh * opts.paddingVertical);
    const padH = Math.round(bh * opts.paddingHorizontal);
    const px = Math.max(0, r.x0 - padH);
    const py = Math.max(0, r.y0 - padV);
    const right = Math.min(width, r.x1 + 1 + padH);
    const bottom = Math.min(height, r.y1 + 1 + padV);

    const x = Math.max(0, Math.round(px / ratio));
    const y = Math.max(0, Math.round(py / ratio));
    const w = Math.min(img.width - x, Math.round((right - px) / ratio));
    const h = Math.min(img.height - y, Math.round((bottom - py) / ratio));
    if (w <= 5 || h <= 5) continue;

    if (!opts.rotated || !r.edge?.length) {
      boxes.push({ x, y, width: w, height: h });
      continue;
    }
    const fitted = minAreaRect(convexHull(Int32Array.from(r.edge), r.edge.length / 2));
    if (!fitted) {
      boxes.push({ x, y, width: w, height: h });
      continue;
    }
    const rect = toOriginal(grow(normalize(fitted), opts), ratio);
    boxes.push({ ...boundsOf(rect, img.width, img.height), rect });
  }
  // Merging unions upright extents, which would discard the fitted angle.
  const merged = opts.mergeOverlap > 0 && !opts.rotated
    ? mergeOverlapping(boxes, opts.mergeOverlap)
    : boxes;
  return { boxes: merged, probability: prob, width, height, ratio };
}

/** Text reads along the long side, so make that the width and fold the angle. */
function normalize(r: RotatedRect): RotatedRect {
  let { width, height, angle } = r;
  if (width < height) {
    [width, height] = [height, width];
    angle += Math.PI / 2;
  }
  while (angle > Math.PI / 2) angle -= Math.PI;
  while (angle <= -Math.PI / 2) angle += Math.PI;
  return { cx: r.cx, cy: r.cy, width, height, angle };
}

/** Same padding rule as the upright path: both margins scale with the height. */
function grow(r: RotatedRect, opts: DetectOptions): RotatedRect {
  return {
    ...r,
    width: r.width + 2 * r.height * opts.paddingHorizontal,
    height: r.height + 2 * r.height * opts.paddingVertical,
  };
}

function toOriginal(r: RotatedRect, ratio: number): RotatedRect {
  return {
    cx: r.cx / ratio,
    cy: r.cy / ratio,
    width: r.width / ratio,
    height: r.height / ratio,
    angle: r.angle,
  };
}

function boundsOf(r: RotatedRect, maxW: number, maxH: number): Box {
  const c = Math.abs(Math.cos(r.angle));
  const s = Math.abs(Math.sin(r.angle));
  const w = r.width * c + r.height * s;
  const h = r.width * s + r.height * c;
  const x = Math.max(0, Math.round(r.cx - w / 2));
  const y = Math.max(0, Math.round(r.cy - h / 2));
  return {
    x,
    y,
    width: Math.min(maxW - x, Math.round(w)),
    height: Math.min(maxH - y, Math.round(h)),
  };
}
