import { type Tensor, unary } from "../runtime/tensor.ts";

/** Abramowitz & Stegun 7.1.26. Max absolute error 1.5e-7, below f32 epsilon. */
export function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t * Math.exp(-a * a);
  return s * y;
}

export const relu = (a: Tensor): Tensor => unary(a, (x) => (x > 0 ? x : 0));
export const sigmoid = (a: Tensor): Tensor => unary(a, (x) => 1 / (1 + Math.exp(-x)));
export const erfOp = (a: Tensor): Tensor => unary(a, erf);

export const hardSigmoid = (a: Tensor, alpha: number, beta: number): Tensor =>
  unary(a, (x) => Math.min(1, Math.max(0, alpha * x + beta)));
