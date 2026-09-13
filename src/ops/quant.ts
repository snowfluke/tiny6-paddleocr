import { make, type Tensor } from "../runtime/tensor.ts";

/**
 * QuantizeLinear / DequantizeLinear with a per-tensor or per-axis scale and an
 * int8 zero point. The quantized tensor keeps its values in a Float32Array,
 * which is exact for int8, so the pair runs in fp32 as fake quantization:
 * the error an int8 kernel would introduce, without the kernel. Reference
 * path only; the resident runtime falls back to these.
 */
export function quantizeLinear(x: Tensor, scale: Tensor, zeroPoint: Tensor | null, axis = 1): Tensor {
  const out = make(x.dims);
  const ch = channelOf(x.dims, scale.data.length, axis);
  for (let i = 0; i < x.data.length; i++) {
    const c = ch(i);
    const q = roundHalfEven(x.data[i] / scale.data[c]) + (zeroPoint ? zeroPoint.data[c] : 0);
    out.data[i] = q < -128 ? -128 : q > 127 ? 127 : q;
  }
  return out;
}

export function dequantizeLinear(q: Tensor, scale: Tensor, zeroPoint: Tensor | null, axis = 1): Tensor {
  const out = make(q.dims);
  const ch = channelOf(q.dims, scale.data.length, axis);
  for (let i = 0; i < q.data.length; i++) {
    const c = ch(i);
    out.data[i] = (q.data[i] - (zeroPoint ? zeroPoint.data[c] : 0)) * scale.data[c];
  }
  return out;
}

/** ONNX rounds ties to even, as f32x4.nearest does. Math.round rounds them up. */
export function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Flat index -> scale index. One scale means per-tensor. */
function channelOf(dims: number[], scales: number, axis: number): (i: number) => number {
  if (scales === 1) return () => 0;
  if (dims[axis] !== scales) throw new Error(`${scales} scales for axis ${axis} of [${dims}]`);
  const inner = dims.slice(axis + 1).reduce((a, b) => a * b, 1);
  return (i) => Math.floor(i / inner) % scales;
}
