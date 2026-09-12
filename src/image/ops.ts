import type { RGBA } from "./png.ts";

/**
 * Bilinear resize, matching what a canvas drawImage downscale gives closely
 * enough for the detector. Box-filters first when shrinking by more than 2x,
 * otherwise a straight bilinear sample aliases text into noise.
 */
export function resize(src: RGBA, width: number, height: number): RGBA {
  let cur = src;
  while (cur.width >= width * 2 && cur.height >= height * 2 && cur.width > 1 && cur.height > 1) {
    cur = halve(cur);
  }
  if (cur.width === width && cur.height === height) return cur;

  const out = new Uint8Array(width * height * 4);
  const xr = cur.width / width;
  const yr = cur.height / height;
  for (let y = 0; y < height; y++) {
    const sy = Math.min(cur.height - 1, (y + 0.5) * yr - 0.5);
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(cur.height - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = Math.min(cur.width - 1, (x + 0.5) * xr - 0.5);
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(cur.width - 1, x0 + 1);
      const fx = sx - x0;
      const i00 = (y0 * cur.width + x0) * 4;
      const i01 = (y0 * cur.width + x1) * 4;
      const i10 = (y1 * cur.width + x0) * 4;
      const i11 = (y1 * cur.width + x1) * 4;
      const d = (y * width + x) * 4;
      for (let c = 0; c < 4; c++) {
        const top = cur.data[i00 + c] + (cur.data[i01 + c] - cur.data[i00 + c]) * fx;
        const bot = cur.data[i10 + c] + (cur.data[i11 + c] - cur.data[i10 + c]) * fx;
        out[d + c] = top + (bot - top) * fy + 0.5;
      }
    }
  }
  return { width, height, data: out };
}

/** 2x2 box reduction, the cheap antialiasing step for large downscales. */
function halve(src: RGBA): RGBA {
  const w = src.width >> 1;
  const h = src.height >> 1;
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = ((y * 2) * src.width + x * 2) * 4;
      const b = a + 4;
      const c = a + src.width * 4;
      const d = c + 4;
      const o = (y * w + x) * 4;
      for (let k = 0; k < 4; k++) {
        out[o + k] = (src.data[a + k] + src.data[b + k] + src.data[c + k] + src.data[d + k] + 2) >> 2;
      }
    }
  }
  return { width: w, height: h, data: out };
}

export function crop(src: RGBA, x: number, y: number, w: number, h: number): RGBA {
  const out = new Uint8Array(w * h * 4);
  for (let j = 0; j < h; j++) {
    const sy = Math.min(src.height - 1, Math.max(0, y + j));
    for (let i = 0; i < w; i++) {
      const sx = Math.min(src.width - 1, Math.max(0, x + i));
      const s = (sy * src.width + sx) * 4;
      const d = (j * w + i) * 4;
      out[d] = src.data[s];
      out[d + 1] = src.data[s + 1];
      out[d + 2] = src.data[s + 2];
      out[d + 3] = src.data[s + 3];
    }
  }
  return { width: w, height: h, data: out };
}

/** Place `src` at the top-left of a black `w` x `h` field. */
export function padTo(src: RGBA, w: number, h: number): RGBA {
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < Math.min(h, src.height); y++) {
    const row = src.data.subarray(y * src.width * 4, y * src.width * 4 + Math.min(w, src.width) * 4);
    out.set(row, y * w * 4);
  }
  return { width: w, height: h, data: out };
}

/**
 * Sample a rotated rectangle into an upright image, bilinearly.
 *
 * Text detected at an angle has to be straightened before recognition: the
 * model reads a 48-pixel-tall horizontal strip, and feeding it the rect's
 * axis-aligned bounding box would include the neighbouring lines that the
 * diagonal sweeps through.
 */
export function cropRotated(
  src: RGBA,
  cx: number,
  cy: number,
  rw: number,
  rh: number,
  angle: number,
  outW: number,
  outH: number,
): RGBA {
  const out = new Uint8Array(outW * outH * 4);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  for (let y = 0; y < outH; y++) {
    // Map the output pixel back into the source through the rect's frame.
    const v = (y + 0.5) / outH * rh - rh / 2;
    for (let x = 0; x < outW; x++) {
      const u = (x + 0.5) / outW * rw - rw / 2;
      const sx = cx + u * c - v * s;
      const sy = cy + u * s + v * c;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const cl = (n: number, hi: number) => (n < 0 ? 0 : n > hi ? hi : n);
      const xa = cl(x0, src.width - 1);
      const xb = cl(x0 + 1, src.width - 1);
      const ya = cl(y0, src.height - 1);
      const yb = cl(y0 + 1, src.height - 1);
      const i00 = (ya * src.width + xa) * 4;
      const i01 = (ya * src.width + xb) * 4;
      const i10 = (yb * src.width + xa) * 4;
      const i11 = (yb * src.width + xb) * 4;
      const d = (y * outW + x) * 4;
      for (let k = 0; k < 4; k++) {
        const top = src.data[i00 + k] + (src.data[i01 + k] - src.data[i00 + k]) * fx;
        const bot = src.data[i10 + k] + (src.data[i11 + k] - src.data[i10 + k]) * fx;
        out[d + k] = top + (bot - top) * fy + 0.5;
      }
    }
  }
  return { width: outW, height: outH, data: out };
}
