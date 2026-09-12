import { make, numel, type Tensor } from "../runtime/tensor.ts";

export type ConvAttrs = {
  kernel: [number, number];
  strides: [number, number];
  pads: [number, number, number, number]; // top left bottom right
  dilations: [number, number];
  group: number;
};

/** X[N,Cin,H,W] * W[Cout,Cin/group,kh,kw] + B[Cout]. Reference implementation. */
export function conv2d(x: Tensor, w: Tensor, b: Tensor | null, a: ConvAttrs): Tensor {
  const [N, Cin, H, W] = x.dims;
  const [Cout, CinPer, kh, kw] = w.dims;
  const [sy, sx] = a.strides;
  const [dy, dx] = a.dilations;
  const [pt, pl, pb, pr] = a.pads;
  const OH = Math.floor((H + pt + pb - (kh - 1) * dy - 1) / sy) + 1;
  const OW = Math.floor((W + pl + pr - (kw - 1) * dx - 1) / sx) + 1;
  const out = make([N, Cout, OH, OW]);
  const coutPerGroup = Cout / a.group;

  for (let n = 0; n < N; n++) {
    for (let oc = 0; oc < Cout; oc++) {
      const g = Math.floor(oc / coutPerGroup);
      const icBase = g * CinPer;
      const bias = b ? b.data[oc] : 0;
      for (let oy = 0; oy < OH; oy++) {
        const iy0 = oy * sy - pt;
        for (let ox = 0; ox < OW; ox++) {
          const ix0 = ox * sx - pl;
          let sum = bias;
          for (let c = 0; c < CinPer; c++) {
            const xPlane = ((n * Cin + icBase + c) * H) * W;
            const wPlane = ((oc * CinPer + c) * kh) * kw;
            for (let ky = 0; ky < kh; ky++) {
              const iy = iy0 + ky * dy;
              if (iy < 0 || iy >= H) continue;
              for (let kx = 0; kx < kw; kx++) {
                const ix = ix0 + kx * dx;
                if (ix < 0 || ix >= W) continue;
                sum += x.data[xPlane + iy * W + ix] * w.data[wPlane + ky * kw + kx];
              }
            }
          }
          out.data[((n * Cout + oc) * OH + oy) * OW + ox] = sum;
        }
      }
    }
  }
  return out;
}

/** X[N,Cin,H,W] * W[Cin,Cout/group,kh,kw]. Note the transposed weight layout. */
export function convTranspose2d(x: Tensor, w: Tensor, b: Tensor | null, a: ConvAttrs): Tensor {
  const [N, Cin, H, W] = x.dims;
  const [, CoutPer, kh, kw] = w.dims;
  const Cout = CoutPer * a.group;
  const [sy, sx] = a.strides;
  const [dy, dx] = a.dilations;
  const [pt, pl, pb, pr] = a.pads;
  const OH = (H - 1) * sy - pt - pb + (kh - 1) * dy + 1;
  const OW = (W - 1) * sx - pl - pr + (kw - 1) * dx + 1;
  const out = make([N, Cout, OH, OW]);
  const cinPerGroup = Cin / a.group;

  if (b) {
    for (let n = 0; n < N; n++) {
      for (let oc = 0; oc < Cout; oc++) out.data.fill(b.data[oc], ((n * Cout + oc) * OH) * OW, ((n * Cout + oc) * OH + OH) * OW);
    }
  }
  for (let n = 0; n < N; n++) {
    for (let ic = 0; ic < Cin; ic++) {
      const g = Math.floor(ic / cinPerGroup);
      for (let co = 0; co < CoutPer; co++) {
        const oc = g * CoutPer + co;
        for (let iy = 0; iy < H; iy++) {
          for (let ix = 0; ix < W; ix++) {
            const v = x.data[((n * Cin + ic) * H + iy) * W + ix];
            if (v === 0) continue;
            for (let ky = 0; ky < kh; ky++) {
              const oy = iy * sy + ky * dy - pt;
              if (oy < 0 || oy >= OH) continue;
              for (let kx = 0; kx < kw; kx++) {
                const ox = ix * sx + kx * dx - pl;
                if (ox < 0 || ox >= OW) continue;
                out.data[((n * Cout + oc) * OH + oy) * OW + ox] +=
                  v * w.data[((ic * CoutPer + co) * kh + ky) * kw + kx];
              }
            }
          }
        }
      }
    }
  }
  return out;
}

export type PoolAttrs = {
  kernel: [number, number];
  strides: [number, number];
  pads: [number, number, number, number];
  ceilMode: boolean;
  countIncludePad: boolean;
};

export function pool2d(x: Tensor, a: PoolAttrs, kind: "max" | "avg"): Tensor {
  const [N, C, H, W] = x.dims;
  const [kh, kw] = a.kernel;
  const [sy, sx] = a.strides;
  const [pt, pl, pb, pr] = a.pads;
  const round = a.ceilMode ? Math.ceil : Math.floor;
  const OH = round((H + pt + pb - kh) / sy) + 1;
  const OW = round((W + pl + pr - kw) / sx) + 1;
  const out = make([N, C, OH, OW]);

  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      const plane = (n * C + c) * H * W;
      for (let oy = 0; oy < OH; oy++) {
        for (let ox = 0; ox < OW; ox++) {
          let acc = kind === "max" ? -Infinity : 0;
          let count = 0;
          for (let ky = 0; ky < kh; ky++) {
            const iy = oy * sy - pt + ky;
            if (iy < 0 || iy >= H) continue;
            for (let kx = 0; kx < kw; kx++) {
              const ix = ox * sx - pl + kx;
              if (ix < 0 || ix >= W) continue;
              const v = x.data[plane + iy * W + ix];
              if (kind === "max") acc = v > acc ? v : acc;
              else acc += v;
              count++;
            }
          }
          const div = a.countIncludePad ? kh * kw : count;
          out.data[((n * C + c) * OH + oy) * OW + ox] = kind === "max" ? acc : acc / div;
        }
      }
    }
  }
  return out;
}

export function globalAveragePool(x: Tensor): Tensor {
  const [N, C, H, W] = x.dims;
  const out = make([N, C, 1, 1]);
  const hw = H * W;
  for (let i = 0; i < N * C; i++) {
    let s = 0;
    for (let j = 0; j < hw; j++) s += x.data[i * hw + j];
    out.data[i] = s / hw;
  }
  return out;
}

export function batchNorm(
  x: Tensor,
  scale: Tensor,
  bias: Tensor,
  mean: Tensor,
  varr: Tensor,
  epsilon: number,
): Tensor {
  const [N, C] = x.dims;
  const inner = numel(x.dims) / (N * C);
  const out = make(x.dims.slice());
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      const k = scale.data[c] / Math.sqrt(varr.data[c] + epsilon);
      const d = bias.data[c] - mean.data[c] * k;
      const base = (n * C + c) * inner;
      for (let i = 0; i < inner; i++) out.data[base + i] = x.data[base + i] * k + d;
    }
  }
  return out;
}

/** Batched matmul; batch dims of the smaller operand broadcast. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const ar = a.dims.length;
  const br = b.dims.length;
  const M = a.dims[ar - 2];
  const K = a.dims[ar - 1];
  const K2 = b.dims[br - 2];
  const N = b.dims[br - 1];
  if (K !== K2) throw new Error(`matmul shape mismatch [${a.dims}] x [${b.dims}]`);

  const aBatch = numel(a.dims.slice(0, ar - 2));
  const bBatch = numel(b.dims.slice(0, br - 2));
  const batch = Math.max(aBatch, bBatch);
  const outDims = (aBatch >= bBatch ? a.dims.slice(0, ar - 2) : b.dims.slice(0, br - 2)).concat([M, N]);
  const out = make(outDims);

  for (let z = 0; z < batch; z++) {
    const ao = (aBatch === 1 ? 0 : z) * M * K;
    const bo = (bBatch === 1 ? 0 : z) * K * N;
    const co = z * M * N;
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const av = a.data[ao + i * K + k];
        if (av === 0) continue;
        const brow = bo + k * N;
        const crow = co + i * N;
        for (let j = 0; j < N; j++) out.data[crow + j] += av * b.data[brow + j];
      }
    }
  }
  return out;
}
