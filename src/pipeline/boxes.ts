import type { Box } from "./detect.ts";

const area = (b: Box) => b.width * b.height;

function intersection(a: Box, b: Box): number {
  const w = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
  return w > 0 && h > 0 ? w * h : 0;
}

function union(a: Box, b: Box): Box {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return {
    x,
    y,
    width: Math.max(a.x + a.width, b.x + b.width) - x,
    height: Math.max(a.y + a.height, b.y + b.height) - y,
  };
}

/**
 * Merge boxes that cover mostly the same pixels.
 *
 * Padding each component by a fraction of its height makes neighbouring
 * fragments overlap, and a logo or barcode shatters into dozens of fragments
 * that all overlap. Recognising each one separately is wasted work and turns
 * one piece of noise into dozens of junk lines. Two boxes merge when their
 * intersection covers `overlap` of the smaller one, which keeps adjacent
 * words apart (they touch at the edges) while collapsing stacked fragments.
 *
 * Runs to a fixed point: merging creates a larger box that may now swallow a
 * third, so one pass is not enough.
 */
export function mergeOverlapping(boxes: Box[], overlap = 0.5): Box[] {
  let current = [...boxes];
  for (let pass = 0; pass < 8; pass++) {
    const merged: Box[] = [];
    const used = new Array(current.length).fill(false);
    let changed = false;

    for (let i = 0; i < current.length; i++) {
      if (used[i]) continue;
      let box = current[i];
      for (let j = i + 1; j < current.length; j++) {
        if (used[j]) continue;
        const inter = intersection(box, current[j]);
        if (inter <= 0) continue;
        if (inter >= overlap * Math.min(area(box), area(current[j]))) {
          box = union(box, current[j]);
          used[j] = true;
          changed = true;
        }
      }
      merged.push(box);
    }
    current = merged;
    if (!changed) break;
  }
  return current;
}

/** Andrew's monotone chain. Points come back counter-clockwise. */
export function convexHull(pts: Int32Array, count: number): number[][] {
  const p: number[][] = [];
  for (let i = 0; i < count; i++) p.push([pts[i * 2], pts[i * 2 + 1]]);
  p.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  if (p.length < 3) return p;

  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const build = (src: number[][]) => {
    const out: number[][] = [];
    for (const q of src) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return [...build(p), ...build([...p].reverse())];
}

export type RotatedRect = {
  cx: number;
  cy: number;
  width: number;
  height: number;
  /** Radians, counter-clockwise, of the edge treated as the rectangle's width. */
  angle: number;
};

/**
 * Minimum-area enclosing rectangle by rotating calipers.
 *
 * The smallest such rectangle always has a side flush with a hull edge, so
 * trying every edge as the candidate orientation is exact, not a search.
 * This is the piece cv.minAreaRect provided.
 */
export function minAreaRect(hull: number[][]): RotatedRect | null {
  if (hull.length === 0) return null;
  if (hull.length < 3) {
    const xs = hull.map((p) => p[0]);
    const ys = hull.map((p) => p[1]);
    const x0 = Math.min(...xs);
    const y0 = Math.min(...ys);
    return {
      cx: (x0 + Math.max(...xs)) / 2,
      cy: (y0 + Math.max(...ys)) / 2,
      width: Math.max(...xs) - x0 || 1,
      height: Math.max(...ys) - y0 || 1,
      angle: 0,
    };
  }

  let best: RotatedRect | null = null;
  let bestArea = Infinity;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);
    if (len < 1e-9) continue;
    const ux = dx / len;
    const uy = dy / len;

    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;
    for (const p of hull) {
      const u = p[0] * ux + p[1] * uy;
      const v = -p[0] * uy + p[1] * ux;
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const w = maxU - minU;
    const h = maxV - minV;
    const rectArea = w * h;
    if (rectArea >= bestArea) continue;

    bestArea = rectArea;
    const mu = (minU + maxU) / 2;
    const mv = (minV + maxV) / 2;
    best = {
      cx: mu * ux - mv * uy,
      cy: mu * uy + mv * ux,
      width: w,
      height: h,
      angle: Math.atan2(uy, ux),
    };
  }
  return best;
}

/**
 * Grow a rectangle the way DB postprocess unclips a shrunk polygon: offset
 * every edge outward by `area * ratio / perimeter`.
 */
export function unclip(r: RotatedRect, ratio: number): RotatedRect {
  const a = r.width * r.height;
  const perimeter = 2 * (r.width + r.height);
  if (perimeter <= 0) return r;
  const d = (a * ratio) / perimeter;
  return { ...r, width: r.width + 2 * d, height: r.height + 2 * d };
}

/** The rectangle's four corners, counter-clockwise from the local origin. */
export function corners(r: RotatedRect): number[][] {
  const c = Math.cos(r.angle);
  const s = Math.sin(r.angle);
  const hw = r.width / 2;
  const hh = r.height / 2;
  return [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ].map(([x, y]) => [r.cx + x * c - y * s, r.cy + x * s + y * c]);
}
