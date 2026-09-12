import { make, numel, strides, type Tensor } from "../runtime/tensor.ts";

export function transpose(x: Tensor, perm: number[]): Tensor {
  const rank = x.dims.length;
  const outDims = perm.map((p) => x.dims[p]);
  const out = make(outDims);
  const xs = strides(x.dims);
  const os = strides(outDims);
  const n = out.data.length;
  for (let i = 0; i < n; i++) {
    let rem = i;
    let src = 0;
    for (let d = 0; d < rank; d++) {
      const k = Math.floor(rem / os[d]);
      rem -= k * os[d];
      src += k * xs[perm[d]];
    }
    out.data[i] = x.data[src];
  }
  return out;
}

const norm = (axes: number[], rank: number) => axes.map((a) => (a < 0 ? a + rank : a));

export function squeeze(x: Tensor, axes: number[]): Tensor {
  const drop = new Set(norm(axes, x.dims.length));
  const dims = x.dims.filter((_, i) => !drop.has(i));
  return { dims: dims.length ? dims : [1], data: x.data };
}

export function unsqueeze(x: Tensor, axes: number[]): Tensor {
  const rank = x.dims.length + axes.length;
  const add = new Set(norm(axes, rank));
  const dims: number[] = [];
  let j = 0;
  for (let i = 0; i < rank; i++) dims.push(add.has(i) ? 1 : x.dims[j++]);
  return { dims, data: x.data };
}

export function concat(xs: Tensor[], axis: number): Tensor {
  const rank = xs[0].dims.length;
  const ax = axis < 0 ? axis + rank : axis;
  const dims = xs[0].dims.slice();
  dims[ax] = xs.reduce((a, t) => a + t.dims[ax], 0);
  const out = make(dims);
  const outer = numel(dims.slice(0, ax));
  const outAxis = dims[ax];
  const inner = numel(dims.slice(ax + 1));
  let offset = 0;
  for (const t of xs) {
    const step = t.dims[ax] * inner;
    for (let o = 0; o < outer; o++) {
      out.data.set(t.data.subarray(o * step, (o + 1) * step), (o * outAxis + offset) * inner);
    }
    offset += t.dims[ax];
  }
  return out;
}

export function reduceMean(x: Tensor, axes: number[], keepdims: boolean): Tensor {
  const rank = x.dims.length;
  const red = new Set(norm(axes, rank));
  const outDims = x.dims.map((d, i) => (red.has(i) ? 1 : d));
  const out = make(outDims);
  const xs = strides(x.dims);
  const os = strides(outDims);
  let count = 1;
  for (const a of red) count *= x.dims[a];

  const n = x.data.length;
  for (let i = 0; i < n; i++) {
    let rem = i;
    let dst = 0;
    for (let d = 0; d < rank; d++) {
      const k = Math.floor(rem / xs[d]);
      rem -= k * xs[d];
      if (!red.has(d)) dst += k * os[d];
    }
    out.data[dst] += x.data[i];
  }
  for (let i = 0; i < out.data.length; i++) out.data[i] /= count;
  if (!keepdims) return { dims: x.dims.filter((_, i) => !red.has(i)), data: out.data };
  return out;
}

export function softmax(x: Tensor, axis: number): Tensor {
  const rank = x.dims.length;
  const ax = axis < 0 ? axis + rank : axis;
  const outer = numel(x.dims.slice(0, ax));
  const n = x.dims[ax];
  const inner = numel(x.dims.slice(ax + 1));
  const out = make(x.dims.slice());
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      const base = o * n * inner + i;
      let max = -Infinity;
      for (let k = 0; k < n; k++) {
        const v = x.data[base + k * inner];
        if (v > max) max = v;
      }
      let sum = 0;
      for (let k = 0; k < n; k++) {
        const e = Math.exp(x.data[base + k * inner] - max);
        out.data[base + k * inner] = e;
        sum += e;
      }
      for (let k = 0; k < n; k++) out.data[base + k * inner] /= sum;
    }
  }
  return out;
}

/**
 * Resize, nearest + asymmetric + floor only. Every Resize in these two models
 * uses that combination with integer scales, so it reduces to pixel replication.
 */
export function resizeNearest(x: Tensor, scales: number[]): Tensor {
  const rank = x.dims.length;
  const outDims = x.dims.map((d, i) => Math.floor(d * scales[i]));
  const out = make(outDims);
  const xs = strides(x.dims);
  const os = strides(outDims);
  const n = out.data.length;
  for (let i = 0; i < n; i++) {
    let rem = i;
    let src = 0;
    for (let d = 0; d < rank; d++) {
      const k = Math.floor(rem / os[d]);
      rem -= k * os[d];
      src += Math.floor(k / scales[d]) * xs[d];
    }
    out.data[i] = x.data[src];
  }
  return out;
}
