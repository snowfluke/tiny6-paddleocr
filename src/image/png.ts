// PNG decoder. 8-bit, non-interlaced, colour types 0/2/4/6.
// Inflate comes from DecompressionStream, which every target runtime has,
// so this stays dependency-free.

export type RGBA = { width: number; height: number; data: Uint8Array };

const SIG = [137, 80, 78, 71, 13, 10, 26, 10];

const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 4: 2, 6: 4 };

export async function decodePng(buf: Uint8Array): Promise<RGBA> {
  for (let i = 0; i < 8; i++) {
    if (buf[i] !== SIG[i]) throw new Error("not a PNG");
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  let width = 0;
  let height = 0;
  let depth = 0;
  let colorType = 0;
  let palette: Uint8Array | null = null;
  const idat: Uint8Array[] = [];

  let o = 8;
  while (o < buf.length) {
    const len = dv.getUint32(o);
    const type = String.fromCharCode(buf[o + 4], buf[o + 5], buf[o + 6], buf[o + 7]);
    const start = o + 8;
    if (type === "IHDR") {
      width = dv.getUint32(start);
      height = dv.getUint32(start + 4);
      depth = buf[start + 8];
      colorType = buf[start + 9];
      if (buf[start + 12] !== 0) throw new Error("interlaced PNG not supported");
    } else if (type === "PLTE") {
      palette = buf.subarray(start, start + len);
    } else if (type === "IDAT") {
      idat.push(buf.subarray(start, start + len));
    } else if (type === "IEND") break;
    o = start + len + 4;
  }

  if (depth !== 8) throw new Error(`PNG bit depth ${depth} not supported`);
  const channels = colorType === 3 ? 1 : CHANNELS[colorType];
  if (!channels) throw new Error(`PNG colour type ${colorType} not supported`);
  if (colorType === 3 && !palette) throw new Error("indexed PNG without a palette");

  const raw = await inflate(concat(idat));
  const bpp = channels;
  const stride = width * bpp;
  const out = new Uint8Array(height * stride);

  // Filters reference the pixel to the left (a), above (b), and above-left (c).
  let prev = new Uint8Array(stride);
  let p = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[p++];
    const line = out.subarray(y * stride, (y + 1) * stride);
    line.set(raw.subarray(p, p + stride));
    p += stride;

    switch (filter) {
      case 0:
        break;
      case 1:
        for (let i = bpp; i < stride; i++) line[i] = (line[i] + line[i - bpp]) & 0xff;
        break;
      case 2:
        for (let i = 0; i < stride; i++) line[i] = (line[i] + prev[i]) & 0xff;
        break;
      case 3:
        for (let i = 0; i < stride; i++) {
          const a = i >= bpp ? line[i - bpp] : 0;
          line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xff;
        }
        break;
      case 4:
        for (let i = 0; i < stride; i++) {
          const a = i >= bpp ? line[i - bpp] : 0;
          const b = prev[i];
          const c = i >= bpp ? prev[i - bpp] : 0;
          line[i] = (line[i] + paeth(a, b, c)) & 0xff;
        }
        break;
      default:
        throw new Error(`unknown PNG filter ${filter} on row ${y}`);
    }
    prev = line;
  }

  return { width, height, data: toRgba(out, width, height, colorType, palette) };
}

function paeth(a: number, b: number, c: number): number {
  const pa = Math.abs(b - c);
  const pb = Math.abs(a - c);
  const pc = Math.abs(a + b - 2 * c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

function toRgba(
  src: Uint8Array,
  width: number,
  height: number,
  colorType: number,
  palette: Uint8Array | null,
): Uint8Array {
  const n = width * height;
  if (colorType === 6) return src;
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const d = i * 4;
    if (colorType === 0) {
      const g = src[i];
      out[d] = g;
      out[d + 1] = g;
      out[d + 2] = g;
      out[d + 3] = 255;
    } else if (colorType === 2) {
      const s = i * 3;
      out[d] = src[s];
      out[d + 1] = src[s + 1];
      out[d + 2] = src[s + 2];
      out[d + 3] = 255;
    } else if (colorType === 4) {
      const s = i * 2;
      out[d] = src[s];
      out[d + 1] = src[s];
      out[d + 2] = src[s];
      out[d + 3] = src[s + 1];
    } else {
      const s = src[i] * 3;
      out[d] = palette![s];
      out[d + 1] = palette![s + 1];
      out[d + 2] = palette![s + 2];
      out[d + 3] = 255;
    }
  }
  return out;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((a, b) => a + b.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** IDAT carries a zlib stream, which is what the "deflate" format means here. */
async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new DecompressionStream("deflate"));
  const chunks: Uint8Array[] = [];
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value as Uint8Array);
  }
  return concat(chunks);
}
