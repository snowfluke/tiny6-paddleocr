// JPEG decoder, baseline and progressive. No dependencies.
//
// Covers SOF0/SOF1 (baseline sequential) and SOF2 (progressive), Huffman
// coded, 8-bit, 1 or 3 components, any integer subsampling. Arithmetic coding
// and 12-bit samples are rejected by name rather than guessed at.
//
// Scans write into a per-component coefficient array and the inverse DCT runs
// once at the end. Progressive needs that anyway - a coefficient is refined by
// later scans - and baseline costs nothing extra for it.
//
// The browser entry never uses this; createImageBitmap decodes JPEG there.
// It exists so Node and Bun can read a photo without a native dependency.

import type { RGBA } from "./png.ts";

const ZIGZAG = new Int32Array([
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]);

type Huffman = { lookup: Int32Array; maxLen: number };

/**
 * Flat code table: index by the next `maxLen` bits and read back the symbol
 * and its true length at once, instead of walking a tree bit by bit.
 */
function buildHuffman(counts: Uint8Array, symbols: Uint8Array): Huffman {
  let maxLen = 0;
  for (let i = 0; i < 16; i++) if (counts[i]) maxLen = i + 1;
  if (!maxLen) return { lookup: new Int32Array(1), maxLen: 1 };
  const lookup = new Int32Array(1 << maxLen).fill(-1);

  let code = 0;
  let k = 0;
  for (let len = 1; len <= maxLen; len++) {
    for (let i = 0; i < counts[len - 1]; i++) {
      const shift = maxLen - len;
      const base = code << shift;
      const entry = (symbols[k] << 8) | len;
      for (let f = 0; f < 1 << shift; f++) lookup[base + f] = entry;
      code++;
      k++;
    }
    code <<= 1;
  }
  return { lookup, maxLen };
}

class BitReader {
  private pos: number;
  private bits = 0;
  private count = 0;

  constructor(private readonly buf: Uint8Array, start: number) {
    this.pos = start;
  }

  get offset(): number {
    return this.pos;
  }

  /** 0xFF inside entropy data is followed by a stuffed 0x00; a marker ends the scan. */
  private fill() {
    while (this.count <= 24) {
      let byte = 0;
      if (this.pos < this.buf.length) {
        byte = this.buf[this.pos++];
        if (byte === 0xff) {
          const next = this.buf[this.pos];
          if (next === 0x00) this.pos++;
          else {
            this.pos--;
            byte = 0;
          }
        }
      }
      this.bits = ((this.bits << 8) | byte) >>> 0;
      this.count += 8;
    }
  }

  peek(n: number): number {
    this.fill();
    return (this.bits >>> (this.count - n)) & ((1 << n) - 1);
  }

  bit(): number {
    this.fill();
    this.count--;
    return (this.bits >>> this.count) & 1;
  }

  receive(n: number): number {
    if (n === 0) return 0;
    this.fill();
    this.count -= n;
    return (this.bits >>> this.count) & ((1 << n) - 1);
  }

  /** JPEG stores a signed value as a magnitude whose sign is in the top bit. */
  static extend(v: number, n: number): number {
    return v < 1 << (n - 1) ? v - (1 << n) + 1 : v;
  }

  decode(h: Huffman): number {
    const entry = h.lookup[this.peek(h.maxLen)];
    if (entry < 0) throw new Error("bad Huffman code in JPEG scan");
    this.count -= entry & 0xff;
    return entry >> 8;
  }

  /** A restart marker resets the bit buffer and the DC predictors. */
  restart() {
    this.bits = 0;
    this.count = 0;
    while (this.pos + 1 < this.buf.length) {
      if (this.buf[this.pos] === 0xff) {
        const m = this.buf[this.pos + 1];
        if (m >= 0xd0 && m <= 0xd7) {
          this.pos += 2;
          return;
        }
        if (m !== 0x00 && m !== 0xff) return;
      }
      this.pos++;
    }
  }

  /** Leave the reader on the next marker so the parser can continue. */
  finish(): number {
    let p = this.pos - (this.count >> 3);
    if (p < 0) p = 0;
    while (p + 1 < this.buf.length) {
      if (this.buf[p] === 0xff && this.buf[p + 1] !== 0x00 && this.buf[p + 1] !== 0xff) return p;
      p++;
    }
    return this.buf.length;
  }
}

/**
 * Basis for the n-point inverse DCT, cached per n. The 8-point table is the
 * one every full-resolution decode uses; 4 and 2 exist so a half or quarter
 * scale decode can reconstruct a block from its low-frequency corner alone.
 */
const COS_CACHE = new Map<number, Float32Array>();
function scaledCos(n: number): Float32Array {
  const hit = COS_CACHE.get(n);
  if (hit) return hit;
  const t = new Float32Array(n * n);
  for (let u = 0; u < n; u++) {
    for (let x = 0; x < n; x++) {
      t[u * n + x] = Math.cos(((2 * x + 1) * u * Math.PI) / (2 * n)) * (u === 0 ? Math.SQRT1_2 : 1);
    }
  }
  COS_CACHE.set(n, t);
  return t;
}

const COS = scaledCos(8);


type Component = {
  id: number;
  h: number;
  v: number;
  tq: number;
  dcTable: number;
  acTable: number;
  pred: number;
  blocksPerLine: number;
  blocksPerColumn: number;
  coeffs: Int16Array;
  pixels: Uint8Array;
  lineWidth: number;
};

/**
 * Decode a JPEG. `downscale` of 2 or 4 returns the image at a half or quarter
 * of its encoded size, reconstructed from the low-frequency DCT coefficients
 * of each block instead of a full decode followed by a resample — the entropy
 * decode is the same, the inverse transform shrinks quadratically with it.
 *
 * The box filter it replaces was measured on SROIE receipts with the detection
 * ceiling at 960: 1/2 cost 214 ms -> ~54 ms of decode against -1.2 pp of
 * character similarity (81.5% -> 80.3%), 1/4 collapsed to 61.8% and is not
 * usable. The loss at 1/2 is the receipts whose long edge lands under 1920,
 * where the halved raster falls below the 960 ceiling and detection loses the
 * detail it was going to keep.
 */
export function decodeJpeg(buf: Uint8Array, downscale: 1 | 2 | 4 = 1): RGBA {
  if (buf[0] !== 0xff || buf[1] !== 0xd8) throw new Error("not a JPEG");
  if (downscale !== 1 && downscale !== 2 && downscale !== 4) {
    throw new Error(`downscale must be 1, 2 or 4, got ${downscale}`);
  }

  const quant: Int32Array[] = [];
  const dcTables: Huffman[] = [];
  const acTables: Huffman[] = [];
  let width = 0;
  let height = 0;
  let components: Component[] = [];
  let restartInterval = 0;
  let progressive = false;
  let hMax = 1;
  let vMax = 1;

  let o = 2;
  while (o < buf.length - 1) {
    if (buf[o] !== 0xff) {
      o++;
      continue;
    }
    const marker = buf[o + 1];
    o += 2;
    if (marker === 0xd9) break;
    if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01 || marker === 0xff) {
      continue;
    }
    const len = (buf[o] << 8) | buf[o + 1];
    const seg = o + 2;
    const segEnd = o + len;

    if (marker === 0xdb) {
      let p = seg;
      while (p < segEnd) {
        const pq = buf[p] >> 4;
        const tq = buf[p] & 15;
        p++;
        const table = new Int32Array(64);
        for (let i = 0; i < 64; i++) {
          table[ZIGZAG[i]] = pq ? (buf[p] << 8) | buf[p + 1] : buf[p];
          p += pq ? 2 : 1;
        }
        quant[tq] = table;
      }
    } else if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2) {
      progressive = marker === 0xc2;
      if (buf[seg] !== 8) throw new Error(`JPEG sample precision ${buf[seg]} is not supported`);
      height = (buf[seg + 1] << 8) | buf[seg + 2];
      width = (buf[seg + 3] << 8) | buf[seg + 4];
      const count = buf[seg + 5];
      components = [];
      for (let i = 0; i < count; i++) {
        const p = seg + 6 + i * 3;
        components.push({
          id: buf[p],
          h: buf[p + 1] >> 4 || 1,
          v: (buf[p + 1] & 15) || 1,
          tq: buf[p + 2],
          dcTable: 0,
          acTable: 0,
          pred: 0,
          blocksPerLine: 0,
          blocksPerColumn: 0,
          coeffs: new Int16Array(0),
          pixels: new Uint8Array(0),
          lineWidth: 0,
        });
      }
      hMax = Math.max(...components.map((c) => c.h));
      vMax = Math.max(...components.map((c) => c.v));
      const mcusPerLine = Math.ceil(width / (8 * hMax));
      const mcusPerColumn = Math.ceil(height / (8 * vMax));
      for (const c of components) {
        c.blocksPerLine = mcusPerLine * c.h;
        c.blocksPerColumn = mcusPerColumn * c.v;
        c.lineWidth = c.blocksPerLine * 8;
        c.coeffs = new Int16Array(c.blocksPerLine * c.blocksPerColumn * 64);
      }
    } else if (marker === 0xc3 || (marker >= 0xc5 && marker <= 0xcf && marker !== 0xc8 && marker !== 0xcc)) {
      throw new Error(`unsupported JPEG frame type 0x${marker.toString(16)}`);
    } else if (marker === 0xcc) {
      throw new Error("arithmetic-coded JPEG is not supported");
    } else if (marker === 0xc4) {
      let p = seg;
      while (p < segEnd) {
        const cls = buf[p] >> 4;
        const id = buf[p] & 15;
        p++;
        const counts = buf.subarray(p, p + 16);
        p += 16;
        let total = 0;
        for (let i = 0; i < 16; i++) total += counts[i];
        const symbols = buf.subarray(p, p + total);
        p += total;
        const table = buildHuffman(counts, symbols);
        if (cls === 0) dcTables[id] = table;
        else acTables[id] = table;
      }
    } else if (marker === 0xdd) {
      restartInterval = (buf[seg] << 8) | buf[seg + 1];
    } else if (marker === 0xda) {
      const count = buf[seg];
      const scan: Component[] = [];
      for (let i = 0; i < count; i++) {
        const id = buf[seg + 1 + i * 2];
        const tables = buf[seg + 2 + i * 2];
        const c = components.find((x) => x.id === id) ?? components[i];
        if (!c) throw new Error(`scan names unknown component ${id}`);
        c.dcTable = tables >> 4;
        c.acTable = tables & 15;
        scan.push(c);
      }
      const p = seg + 1 + count * 2;
      const ss = progressive ? buf[p] : 0;
      const se = progressive ? buf[p + 1] : 63;
      const ah = progressive ? buf[p + 2] >> 4 : 0;
      const al = progressive ? buf[p + 2] & 15 : 0;
      o = decodeScan(
        buf, segEnd, width, height, components, scan,
        dcTables, acTables, restartInterval, hMax, vMax,
        progressive, ss, se, ah, al,
      );
      continue;
    }
    o = segEnd;
  }

  if (!width || !height) throw new Error("JPEG has no frame header");
  // A decode at 1/downscale reconstructs each block from its low-frequency
  // corner: outSize is the n-point inverse DCT that produces one output pixel
  // per block, 8 at full scale. Every component is scaled the same way, so the
  // subsampling ratios in toRgba still hold and only the loop bounds shrink.
  const outSize = 8 / downscale;
  for (const c of components) renderComponent(c, quant[c.tq], outSize);
  return toRgba(
    components,
    Math.ceil(width / downscale),
    Math.ceil(height / downscale),
    hMax,
    vMax,
  );
}

function decodeScan(
  buf: Uint8Array,
  start: number,
  width: number,
  height: number,
  components: Component[],
  scan: Component[],
  dcTables: Huffman[],
  acTables: Huffman[],
  restartInterval: number,
  hMax: number,
  vMax: number,
  progressive: boolean,
  ss: number,
  se: number,
  ah: number,
  al: number,
): number {
  const reader = new BitReader(buf, start);
  for (const c of components) c.pred = 0;
  let eobrun = 0;

  const decodeBlock = (c: Component, offset: number) => {
    const coeffs = c.coeffs;
    if (!progressive) {
      const t = reader.decode(dcTables[c.dcTable]);
      c.pred += t === 0 ? 0 : BitReader.extend(reader.receive(t), t);
      coeffs[offset] = c.pred;
      let k = 1;
      while (k < 64) {
        const rs = reader.decode(acTables[c.acTable]);
        const s = rs & 15;
        const r = rs >> 4;
        if (s === 0) {
          if (r !== 15) break;
          k += 16;
          continue;
        }
        k += r;
        if (k > 63) break;
        coeffs[offset + ZIGZAG[k]] = BitReader.extend(reader.receive(s), s);
        k++;
      }
      return;
    }

    if (ss === 0) {
      // DC scan: first pass sets the high bits, later passes append one each.
      if (ah === 0) {
        const t = reader.decode(dcTables[c.dcTable]);
        c.pred += t === 0 ? 0 : BitReader.extend(reader.receive(t), t);
        coeffs[offset] = c.pred << al;
      } else if (reader.bit()) {
        coeffs[offset] |= 1 << al;
      }
      return;
    }

    if (ah === 0) {
      // AC first pass over the band [ss, se].
      if (eobrun > 0) {
        eobrun--;
        return;
      }
      let k = ss;
      while (k <= se) {
        const rs = reader.decode(acTables[c.acTable]);
        const s = rs & 15;
        const r = rs >> 4;
        if (s === 0) {
          if (r < 15) {
            eobrun = (1 << r) - 1;
            if (r) eobrun += reader.receive(r);
            break;
          }
          k += 16;
          continue;
        }
        k += r;
        if (k > se) break;
        coeffs[offset + ZIGZAG[k]] = BitReader.extend(reader.receive(s), s) * (1 << al);
        k++;
      }
      return;
    }

    // AC refinement: one correction bit for every coefficient already set,
    // interleaved with the run-length coding of the newly non-zero ones.
    const p1 = 1 << al;
    const m1 = -1 << al;
    let k = ss;
    if (eobrun <= 0) {
      while (k <= se) {
        const rs = reader.decode(acTables[c.acTable]);
        const s = rs & 15;
        let r = rs >> 4;
        let value = 0;
        if (s === 0) {
          if (r < 15) {
            eobrun = (1 << r);
            if (r) eobrun += reader.receive(r);
            break;
          }
        } else {
          value = reader.bit() ? p1 : m1;
        }
        while (k <= se) {
          const z = offset + ZIGZAG[k];
          if (coeffs[z] !== 0) {
            if (reader.bit() && (coeffs[z] & p1) === 0) {
              coeffs[z] += coeffs[z] >= 0 ? p1 : m1;
            }
          } else {
            if (r === 0) {
              if (s !== 0) coeffs[z] = value;
              k++;
              break;
            }
            r--;
          }
          k++;
        }
      }
    }
    if (eobrun > 0) {
      while (k <= se) {
        const z = offset + ZIGZAG[k];
        if (coeffs[z] !== 0 && reader.bit() && (coeffs[z] & p1) === 0) {
          coeffs[z] += coeffs[z] >= 0 ? p1 : m1;
        }
        k++;
      }
      eobrun--;
    }
  };

  let sinceRestart = 0;
  const maybeRestart = () => {
    if (!restartInterval || sinceRestart !== restartInterval) return;
    reader.restart();
    for (const c of components) c.pred = 0;
    eobrun = 0;
    sinceRestart = 0;
  };

  if (scan.length === 1) {
    // Non-interleaved: the unit is one block of this component's own grid,
    // sized from the component's pixels rather than the MCU grid.
    const c = scan[0];
    const cols = Math.ceil((width * c.h) / (hMax * 8));
    const rows = Math.ceil((height * c.v) / (vMax * 8));
    for (let by = 0; by < rows; by++) {
      for (let bx = 0; bx < cols; bx++) {
        maybeRestart();
        decodeBlock(c, (by * c.blocksPerLine + bx) * 64);
        sinceRestart++;
      }
    }
  } else {
    const mcusPerLine = Math.ceil(width / (8 * hMax));
    const mcusPerColumn = Math.ceil(height / (8 * vMax));
    for (let my = 0; my < mcusPerColumn; my++) {
      for (let mx = 0; mx < mcusPerLine; mx++) {
        maybeRestart();
        for (const c of scan) {
          for (let by = 0; by < c.v; by++) {
            for (let bx = 0; bx < c.h; bx++) {
              const row = my * c.v + by;
              const col = mx * c.h + bx;
              decodeBlock(c, (row * c.blocksPerLine + col) * 64);
            }
          }
        }
        sinceRestart++;
      }
    }
  }

  return reader.finish();
}

/** Dequantize and inverse-transform every block into 8-bit samples. */
function renderComponent(c: Component, q: Int32Array, outSize: number) {
  if (!q) throw new Error(`component ${c.id} has no quantization table`);
  const T = scaledCos(outSize);
  const n = outSize;
  // NOT n/2. The DC coefficient must land on the block mean F(0,0)/8 at every
  // scale, and both passes contribute C(0)=1/sqrt2 to it, so the two-pass sum
  // always carries F(0,0)/2 and the divisor is the constant 4 - the value the
  // 8-point path has always used. Dividing by n/2 agrees at n=8 by coincidence
  // and doubles the amplitude at n=4 (DC to F(0,0)/4), which clips the raster.
  const div = 4;
  c.lineWidth = c.blocksPerLine * n;
  c.pixels = new Uint8Array(c.lineWidth * c.blocksPerColumn * n);
  const block = new Float32Array(64);
  const tmp = new Float32Array(64);

  for (let by = 0; by < c.blocksPerColumn; by++) {
    for (let bx = 0; bx < c.blocksPerLine; bx++) {
      const off = (by * c.blocksPerLine + bx) * 64;
      // A block whose only non-zero coefficient is DC reconstructs to a flat
      // patch, and flat regions are common in a photographed receipt - the
      // background between glyphs, specular highlights, the white margin all
      // have no AC energy. EOB coding leaves the rest of the block at zero, so
      // proving it is a scan over just the coefficients this scale reads: at
      // 1/2 and 1/4 the higher frequencies are dropped anyway, and a block that
      // is flat at 1/2 need not be flat at 1/1.
      let flat = true;
      for (let y = 0; y < n && flat; y++) {
        for (let u = 0; u < n; u++) {
          if (y === 0 && u === 0) continue;
          if (c.coeffs[off + y * 8 + u]) {
            flat = false;
            break;
          }
        }
      }
      if (flat) {
        // Same arithmetic as the general path would do, not an algebraically
        // equivalent shortcut: the two passes each contribute C(0) = sqrt(1/2)
        // as a float32, so the sum is a float32-rounded product and the value
        // is not exactly block*q[0]/8. Rounding the shortcut instead would move
        // Math.round across its .5 boundary for a small share of blocks and
        // silently change bytes.
        const t = T[0];
        // bb is what the general path stores in block[0][0] (a Float32Array),
        // and its first pass writes fround(t * bb) into tmp. Both passes carry
        // the same factor, so the accumulator is t * fround(t * bb).
        const bb = Math.fround(c.coeffs[off] * q[0]);
        const s = t * Math.fround(t * bb);
        const v = Math.round(s / div + 128);
        const p = v < 0 ? 0 : v > 255 ? 255 : v;
        const ox = bx * n;
        const oy = by * n;
        for (let y = 0; y < n; y++) {
          const row = (oy + y) * c.lineWidth + ox;
          for (let x = 0; x < n; x++) c.pixels[row + x] = p;
        }
        continue;
      }
      if (n === 8) {
        for (let i = 0; i < 64; i++) block[i] = c.coeffs[off + i] * q[i];
      } else {
        // Gather the low-frequency n x n corner, not the first n*n entries of
        // the flat 64. Coeffcients are stored row-major as [vertical * 8 +
        // horizontal], so a flat cut at n*n would keep every horizontal
        // frequency of the first n/2 rows and drop the vertical ones that
        // matter - measured as 15.8 pp of character similarity before the fix.
        for (let y = 0; y < n; y++) {
          for (let u = 0; u < n; u++) {
            const i = y * 8 + u;
            block[y * n + u] = c.coeffs[off + i] * q[i];
          }
        }
      }

      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          let s = 0;
          for (let u = 0; u < n; u++) s += T[u * n + x] * block[y * n + u];
          tmp[y * n + x] = s;
        }
      }
      const ox = bx * n;
      const oy = by * n;
      for (let x = 0; x < n; x++) {
        for (let y = 0; y < n; y++) {
          let s = 0;
          for (let v = 0; v < n; v++) s += T[v * n + y] * tmp[v * n + x];
          const p = Math.round(s / div + 128);
          c.pixels[(oy + y) * c.lineWidth + ox + x] = p < 0 ? 0 : p > 255 ? 255 : p;
        }
      }
    }
  }
}

function toRgba(components: Component[], width: number, height: number, hMax: number, vMax: number): RGBA {
  const out = new Uint8Array(width * height * 4);
  const sample = (c: Component, x: number, y: number) => {
    // Nearest-neighbour chroma upsampling; subsampling already threw the
    // detail away, and text lives in luma.
    const sx = Math.min(c.lineWidth - 1, ((x * c.h) / hMax) | 0);
    const sy = ((y * c.v) / vMax) | 0;
    return c.pixels[sy * c.lineWidth + sx];
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const d = (y * width + x) * 4;
      if (components.length === 1) {
        const g = sample(components[0], x, y);
        out[d] = g;
        out[d + 1] = g;
        out[d + 2] = g;
      } else {
        const Y = sample(components[0], x, y);
        const cb = sample(components[1], x, y) - 128;
        const cr = sample(components[2], x, y) - 128;
        const r = Y + 1.402 * cr;
        const g = Y - 0.344136 * cb - 0.714136 * cr;
        const b = Y + 1.772 * cb;
        out[d] = r < 0 ? 0 : r > 255 ? 255 : r;
        out[d + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
        out[d + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
      }
      out[d + 3] = 255;
    }
  }
  return { width, height, data: out };
}

/** Dispatches on the file's magic bytes. */
/**
 * `downscale` is honoured for JPEG, where it comes free from the DCT; a PNG
 * has no cheap low-frequency path, so it decodes at full size and the caller
 * gets the larger image. A caller that must have the smaller raster can
 * resample the result.
 */
export async function decodeImage(buf: Uint8Array, downscale: 1 | 2 | 4 = 1): Promise<RGBA> {
  if (buf[0] === 0xff && buf[1] === 0xd8) return decodeJpeg(buf, downscale);
  const { decodePng } = await import("./png.ts");
  return decodePng(buf);
}
