export type Tensor = { dims: number[]; data: Float32Array };

export const numel = (dims: number[]): number => dims.reduce((a, b) => a * b, 1);

export const make = (dims: number[]): Tensor => ({ dims, data: new Float32Array(numel(dims)) });

export const strides = (dims: number[]): number[] => {
  const s = new Array<number>(dims.length);
  let acc = 1;
  for (let i = dims.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= dims[i];
  }
  return s;
};

/** numpy broadcast of two shapes, right-aligned. */
export function broadcastShape(a: number[], b: number[]): number[] {
  const n = Math.max(a.length, b.length);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const x = a[a.length - n + i] ?? 1;
    const y = b[b.length - n + i] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new Error(`cannot broadcast [${a}] with [${b}]`);
    out[i] = Math.max(x, y);
  }
  return out;
}

/**
 * Elementwise op over two broadcast operands.
 * The common cases here are same-shape and per-channel bias, so those get
 * straight-line loops and everything else falls back to index arithmetic.
 */
export function binary(a: Tensor, b: Tensor, f: (x: number, y: number) => number): Tensor {
  const dims = broadcastShape(a.dims, b.dims);
  const out = make(dims);
  const n = out.data.length;

  if (a.data.length === n && b.data.length === n) {
    for (let i = 0; i < n; i++) out.data[i] = f(a.data[i], b.data[i]);
    return out;
  }
  if (b.data.length === 1) {
    const y = b.data[0];
    for (let i = 0; i < n; i++) out.data[i] = f(a.data[i], y);
    return out;
  }
  if (a.data.length === 1) {
    const x = a.data[0];
    for (let i = 0; i < n; i++) out.data[i] = f(x, b.data[i]);
    return out;
  }

  const rank = dims.length;
  const os = strides(dims);
  const pad = (d: number[]) => [...new Array(rank - d.length).fill(1), ...d];
  const ad = pad(a.dims);
  const bd = pad(b.dims);
  const as = strides(ad);
  const bs = strides(bd);
  const idx = new Array<number>(rank).fill(0);
  for (let i = 0; i < n; i++) {
    let rem = i;
    let ai = 0;
    let bi = 0;
    for (let d = 0; d < rank; d++) {
      const k = Math.floor(rem / os[d]);
      rem -= k * os[d];
      idx[d] = k;
      ai += (ad[d] === 1 ? 0 : k) * as[d];
      bi += (bd[d] === 1 ? 0 : k) * bs[d];
    }
    out.data[i] = f(a.data[ai], b.data[bi]);
  }
  return out;
}

export function unary(a: Tensor, f: (x: number) => number): Tensor {
  const out = make(a.dims.slice());
  for (let i = 0; i < a.data.length; i++) out.data[i] = f(a.data[i]);
  return out;
}
