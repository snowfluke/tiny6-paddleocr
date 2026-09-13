// Plans which nodes of a fused graph run on int8, and how.
//
// A calibration file gives every activation a per-channel range. From that
// the planner builds, once per session: the quantization parameters of each
// tensor in the int8 region, the folded and packed weights of each planned
// convolution, and a handler per planned node. At run time a handler takes
// resident tensors that are either fp32 (RT) or int8 NHWC (QT), quantizing
// an fp32 input on the way in; an unplanned node reading an int8 tensor gets
// it dequantized. The region's edges are therefore implicit.
//
// Planned today: 1x1 convolutions, depthwise convolutions, the two halves of
// squeeze-and-excite (ReduceMean, Mul by a per-channel factor), and a Gelu
// that follows a planned convolution, which folds into its epilogue.

import type { OnnxGraph, OnnxNode } from "../onnx/parse.ts";
import type { Resident, RT } from "./resident.ts";
import type { Tensor } from "./tensor.ts";
import { ACT_RELU } from "./fuse.ts";
import {
  ACT_GELU,
  allocQ,
  dequantizeFromQ,
  prepareQConv,
  qconv1x1,
  qdepthwise,
  type QParams,
  type QT,
  quantizeToQ,
  uploadDepthwise,
  uploadQConv,
  uploadQParams,
} from "../ops/qconv.ts";

/** Per-channel bounds of one activation, channel axis 1, from a calibration run. */
export type ChannelRange = { dims: number[]; min: number[]; max: number[] };
export type Calibration = Record<string, ChannelRange>;

export type QVal = RT | QT;
export const isQ = (v: QVal): v is QT => "q" in v;

/** Asymmetric int8 per channel. Zero stays representable, so padding is exact. */
export function chanParams(r: ChannelRange): { scale: Float32Array; zp: Int32Array } {
  const C = r.min.length;
  const scale = new Float32Array(C), zp = new Int32Array(C);
  for (let c = 0; c < C; c++) {
    const lo = Math.min(r.min[c], 0), hi = Math.max(r.max[c], 0);
    scale[c] = Math.max(hi - lo, 1e-6) / 255;
    zp[c] = Math.max(-128, Math.min(127, Math.round(-128 - lo / scale[c])));
  }
  return { scale, zp };
}

type Handler = (x: (QVal | null)[]) => QVal[];

/** Smallest plane worth quantizing; squeeze-and-excite's 1x1 tensors stay fp32. */
const MIN_PLANE = 16;

export class QPlan {
  readonly handlers = new Map<OnnxNode, Handler>();
  private readonly params = new Map<string, QParams>();

  constructor(
    private readonly r: Resident,
    private readonly calib: Calibration,
  ) {}

  /** Parameters of a tensor, uploaded on first use. */
  paramsOf(name: string): QParams {
    let p = this.params.get(name);
    if (!p) {
      const range = this.calib[name];
      if (!range) throw new Error(`no calibration for ${name}`);
      const { scale, zp } = chanParams(range);
      p = uploadQParams(this.r, scale, zp);
      this.params.set(name, p);
    }
    return p;
  }

  /** An input as int8, quantizing an fp32 tensor with the name's parameters. */
  asQ(name: string, v: QVal): QT {
    return isQ(v) ? v : quantizeToQ(this.r, v, this.paramsOf(name));
  }

  asRT(v: QVal): RT {
    return isQ(v) ? dequantizeFromQ(this.r, v) : v;
  }
}

/**
 * Builds the plan for a fused graph. Returns the graph with absorbed nodes
 * removed, since a Gelu folded into a convolution's epilogue is no longer a
 * node of its own.
 */
export function buildPlan(
  g: OnnxGraph,
  calib: Calibration,
  r: Resident,
  consts: Map<string, Tensor>,
): { graph: OnnxGraph; plan: QPlan } {
  const plan = new QPlan(r, calib);
  const readers = new Map<string, OnnxNode[]>();
  for (const n of g.nodes) for (const i of n.input) readers.set(i, [...(readers.get(i) ?? []), n]);
  const producer = new Map<string, OnnxNode>();
  for (const n of g.nodes) for (const o of n.output) producer.set(o, n);
  const outputs = new Set(g.outputs.map((o) => o.name));

  const plane = (name: string) => {
    const d = calib[name]?.dims;
    return d && d.length === 4 ? d[2] * d[3] : 0;
  };
  const attr = (n: OnnxNode, k: string) => n.attrs.get(k);
  const ints = (n: OnnxNode, k: string, d: number[]) => attr(n, k)?.ints ?? d;

  // Pass one: which nodes can run on int8 at all.
  const planned = new Set<OnnxNode>();
  const absorbed = new Map<OnnxNode, OnnxNode>(); // gelu -> conv
  const convKind = (n: OnnxNode): "1x1" | "dw" | null => {
    const w = consts.get(n.input[1]);
    if (!w || w.dims.length !== 4) return null;
    const [cout, cinPer, kh, kw] = w.dims;
    const group = attr(n, "group")?.i ?? 1;
    const strides = ints(n, "strides", [1, 1]);
    const pads = ints(n, "pads", [0, 0, 0, 0]);
    const dil = ints(n, "dilations", [1, 1]);
    if (dil[0] !== 1 || dil[1] !== 1) return null;
    if (!calib[n.input[0]] || plane(n.input[0]) < MIN_PLANE) return null;
    if (group === 1 && kh === 1 && kw === 1 && strides[0] === 1 && strides[1] === 1 && pads.every((p) => !p)) {
      return cinPer % 4 === 0 && cout % 8 === 0 ? "1x1" : null;
    }
    if (group === cout && cinPer === 1 && kh === kw && (kh === 3 || kh === 5) && strides[0] <= 2 && strides[1] <= 2) {
      return cout % 16 === 0 ? "dw" : null;
    }
    return null;
  };
  for (const n of g.nodes) {
    if (n.opType === "Conv" && convKind(n)) planned.add(n);
  }
  // Consumers of int8 tensors that can stay on int8.
  for (const n of g.nodes) {
    const p0 = producer.get(n.input[0]);
    if (!p0 || !planned.has(p0)) continue;
    if (n.opType === "ReduceMean") {
      const axes = ints(n, "axes", []);
      if (axes.length === 2 && axes[0] === 2 && axes[1] === 3) planned.add(n);
    } else if (n.opType === "Mul" && n.input.length === 2 && calib[n.output[0]]) {
      const f = calib[n.input[1]]?.dims;
      if (f && f.length === 4 && f[2] === 1 && f[3] === 1) planned.add(n);
    } else if (n.opType === "Gelu" && p0.opType === "Conv" && readers.get(n.input[0])?.length === 1 && !outputs.has(n.input[0])) {
      absorbed.set(n, p0);
    }
  }
  // A planned node's output becomes int8 when it is calibrated and every
  // reader stays on int8; otherwise the epilogue writes fp32 NCHW directly.
  const outName = (n: OnnxNode) => {
    const gelu = [...absorbed].find(([, conv]) => conv === n)?.[0];
    return gelu ? gelu.output[0] : n.output[0];
  };
  const emitsQ = (n: OnnxNode) => {
    const name = outName(n);
    if (!calib[name] || outputs.has(name)) return false;
    const rs = readers.get(name) ?? [];
    return rs.length > 0 && rs.every((m) => planned.has(m) && (m.opType !== "Conv" || m.input[0] === name || m.input[3] === name));
  };

  // Pass two: handlers.
  for (const n of planned) {
    const out = emitsQ(n) ? plan.paramsOf(outName(n)) : null;
    if (n.opType === "Conv") {
      const kind = convKind(n)!;
      const w = consts.get(n.input[1])!;
      const bias = n.input[2] ? consts.get(n.input[2])!.data : null;
      const pin = plan.paramsOf(n.input[0]);
      const gelu = [...absorbed].find(([, conv]) => conv === n)?.[0];
      let act = attr(n, "activation")?.i ?? 0;
      let p0 = 0, p1 = 0;
      if (gelu) {
        act = ACT_GELU;
        p0 = attr(gelu, "scale")!.f!;
        p1 = attr(gelu, "post")!.f!;
      }
      if (kind === "1x1") {
        const q = uploadQConv(r, prepareQConv(w, bias, pin.scale, pin.zp));
        const resName = n.input[3];
        // Nothing may upload once a run has started, so every parameter set
        // a handler could need is made here.
        if (resName) plan.paramsOf(resName);
        plan.handlers.set(n, (x) => {
          const xq = plan.asQ(n.input[0], x[0]!);
          const res = resName ? plan.asQ(resName, x[3]!) : null;
          return [qconv1x1(r, xq, q, act, p0, p1, res, out)];
        });
      } else {
        if (act === ACT_GELU) throw new Error("depthwise gelu epilogue is not implemented");
        const strides = ints(n, "strides", [1, 1]), pads = ints(n, "pads", [0, 0, 0, 0]);
        const dw = uploadDepthwise(r, w, bias, { sy: strides[0], sx: strides[1], pt: pads[0], pl: pads[1] }, pin.scale);
        plan.handlers.set(n, (x) => [qdepthwise(r, plan.asQ(n.input[0], x[0]!), dw, act, out)]);
      }
    } else if (n.opType === "ReduceMean") {
      plan.paramsOf(n.input[0]);
      plan.handlers.set(n, (x) => [qmean(r, plan.asQ(n.input[0], x[0]!))]);
    } else if (n.opType === "Mul") {
      plan.paramsOf(n.input[0]);
      const pout = out ?? plan.paramsOf(n.output[0]);
      plan.handlers.set(n, (x) => [qscale(r, plan.asQ(n.input[0], x[0]!), x[1] as RT, pout)]);
    }
  }

  // Absorbed gelus leave the graph; their convolution takes their output name.
  const nodes: OnnxNode[] = [];
  for (const n of g.nodes) {
    if (absorbed.has(n)) continue;
    const gelu = [...absorbed].find(([, conv]) => conv === n)?.[0];
    if (gelu) {
      const renamed = { ...n, output: gelu.output };
      const h = plan.handlers.get(n)!;
      plan.handlers.delete(n);
      plan.handlers.set(renamed, h);
      nodes.push(renamed);
    } else nodes.push(n);
  }
  return { graph: { ...g, nodes }, plan };
}

// ---- squeeze-and-excite ------------------------------------------------------

function qmean(r: Resident, x: QT): RT {
  const [N, C, H, W] = x.dims;
  const y = r.alloc([N, C, 1, 1]);
  for (let n = 0; n < N; n++) {
    r.ar.k.qmean_channels(C, H * W, x.ptr + n * C * H * W, x.q.ptr.zp, x.q.ptr.scale, y.ptr + n * C * 4);
  }
  return y;
}

function qscale(r: Resident, x: QT, factor: RT, out: QParams): QT {
  const [N, C, H, W] = x.dims;
  if (factor.len !== N * C) throw new Error(`qscale: factor has ${factor.len} values for ${N}x${C} channels`);
  const y = allocQ(r, x.dims, out);
  for (let n = 0; n < N; n++) {
    const off = n * C * H * W;
    r.ar.k.qscale_channels(C, x.ptr + off, x.q.ptr.zp, x.q.ptr.scale, factor.ptr + n * C * 4, y.ptr + off, out.ptr.inv, out.ptr.zp, 0, H * W);
  }
  return y;
}
