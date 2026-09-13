import { make, type Tensor } from "../runtime/tensor.ts";

/**
 * QuantizeLinear / DequantizeLinear with a per-tensor scale and an int8 zero
 * point. The quantized tensor keeps its values in a Float32Array, which is
 * exact for int8, so the pair runs in fp32 as fake quantization: the error
 * an int8 kernel would introduce, without the kernel. Reference path only;
 * the resident runtime falls back to these.
 */
export function quantizeLinear(x: Tensor, scale: Tensor, zeroPoint: Tensor | null): Tensor {
  const s = scalar(scale);
  const zp = zeroPoint ? scalar(zeroPoint) : 0;
  const out = make(x.dims);
  for (let i = 0; i < x.data.length; i++) {
    const q = roundHalfEven(x.data[i] / s) + zp;
    out.data[i] = q < -128 ? -128 : q > 127 ? 127 : q;
  }
  return out;
}

export function dequantizeLinear(q: Tensor, scale: Tensor, zeroPoint: Tensor | null): Tensor {
  const s = scalar(scale);
  const zp = zeroPoint ? scalar(zeroPoint) : 0;
  const out = make(q.dims);
  for (let i = 0; i < q.data.length; i++) out.data[i] = (q.data[i] - zp) * s;
  return out;
}

/** ONNX rounds ties to even, as f32x4.nearest does. Math.round rounds them up. */
export function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function scalar(t: Tensor): number {
  if (t.data.length !== 1) throw new Error(`per-axis quantization is not supported (got ${t.data.length} scales)`);
  return t.data[0];
}
