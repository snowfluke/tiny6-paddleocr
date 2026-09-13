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
import { convAttrs } from "./graph.ts";
import type { Resident, RT } from "./resident.ts";
import type { Tensor } from "./tensor.ts";
import { ACT_RELU } from "./fuse.ts";
import {
  ACT_GELU,
  allocQ,
  dequantizeFromQ,
  prepareQConv,
  qconv1x1,
  qconvDense,
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
  /** fp32 tensors quantized this run, so a tensor read by several planned nodes converts once. */
  private readonly quantized = new Map<number, QT>();

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

  /** Forget the run's boundary conversions; the arena recycles their memory. */
  beginRun() {
    this.quantized.clear();
  }

  /** An input as int8, quantizing an fp32 tensor with the name's parameters. */
  asQ(name: string, v: QVal, cs?: number): QT {
    if (isQ(v)) {
      if (cs !== undefined && v.cs !== cs) throw new Error(`${name}: channel stride ${v.cs}, wanted ${cs}`);
      return v;
    }
    const want = cs ?? v.dims[1];
    const hit = this.quantized.get(v.ptr);
    if (hit && hit.cs === want) return hit;
    const q = quantizeToQ(this.r, v, this.paramsOf(name), cs);
    this.quantized.set(v.ptr, q);
    return q;
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
  const convKind = (n: OnnxNode): "1x1" | "dense" | "dw" | null => {
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
    if (group === 1 && kh <= 5 && kw <= 5 && cout % 8 === 0) {
      // A channel count that is not a multiple of four is padded on the way
      // in, which only a boundary tensor (fp32 producer) can do.
      const src = producer.get(n.input[0]);
      return cinPer % 4 === 0 || !src || !planned.has(src) ? "dense" : null;
    }
    if (group === cout && cinPer === 1 && kh === kw && (kh === 3 || kh === 5) && strides[0] <= 2 && strides[1] <= 2) {
      return cout % 16 === 0 ? "dw" : null;
    }
    return null;
  };
  for (const n of g.nodes) {
    if (n.opType === "Conv" && convKind(n)) planned.add(n);
  }
  // Consumers of int8 tensors that can stay on int8. In graph order, so a
  // chain of them (Add -> Resize -> Add) plans through.
  const dims = (name: string) => calib[name]?.dims;
  const c16 = (name: string) => { const d = dims(name); return !!d && d.length === 4 && d[1] % 16 === 0; };
  for (const n of g.nodes) {
    const p0 = producer.get(n.input[0]);
    if (!p0 || !planned.has(p0)) continue;
    if (n.opType === "ReduceMean") {
      const axes = ints(n, "axes", []);
      if (axes.length === 2 && axes[0] === 2 && axes[1] === 3) planned.add(n);
    } else if (n.opType === "GlobalAveragePool") {
      planned.add(n);
    } else if (n.opType === "Add" && n.input.length === 2 && calib[n.output[0]] && c16(n.output[0])) {
      const p1 = producer.get(n.input[1]);
      const same = JSON.stringify(dims(n.input[0])) === JSON.stringify(dims(n.input[1]));
      if (p1 && planned.has(p1) && same) planned.add(n);
    } else if (n.opType === "Resize" && calib[n.output[0]]) {
      const sc = consts.get(n.input[2])?.data;
      const mode = attr(n, "mode")?.s ?? "nearest";
      if (sc && sc.length === 4 && sc[0] === 1 && sc[1] === 1 && sc[2] === 2 && sc[3] === 2 && mode === "nearest") planned.add(n);
    } else if (n.opType === "Concat" && (attr(n, "axis")?.i ?? 0) === 1 && calib[n.output[0]]) {
      if (n.input.every((i) => { const p = producer.get(i); return p && planned.has(p) && c16(i); })) planned.add(n);
    } else if (n.opType === "MaxPool" && calib[n.output[0]]) {
      const k = ints(n, "kernel_shape", []), st = ints(n, "strides", [1, 1]);
      if (k[0] === 2 && k[1] === 2 && st[0] === 1 && st[1] === 1 && attr(n, "auto_pad")?.s === "SAME_UPPER") planned.add(n);
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
      } else if (kind === "dense") {
        const q = uploadQConv(r, prepareQConv(w, bias, pin.scale, pin.zp));
        const cs = (w.dims[1] + 3) & ~3;
        plan.handlers.set(n, (x) => {
          // Pads may depend on the input size (auto_pad), so resolve per run.
          const at = convAttrs(n, x[0]!, w.dims.slice(2));
          const geom = { kh: w.dims[2], kw: w.dims[3], sy: at.strides[0], sx: at.strides[1], pt: at.pads[0], pl: at.pads[1], pb: at.pads[2], pr: at.pads[3] };
          return [qconvDense(r, plan.asQ(n.input[0], x[0]!, cs), q, geom, act, p0, p1, out)];
        });
      } else {
        if (act === ACT_GELU) throw new Error("depthwise gelu epilogue is not implemented");
        const dw = uploadDepthwise(r, w, bias, pin.scale);
        plan.handlers.set(n, (x) => {
          const at = convAttrs(n, x[0]!, w.dims.slice(2));
          const geom = { sy: at.strides[0], sx: at.strides[1], pt: at.pads[0], pl: at.pads[1], pb: at.pads[2], pr: at.pads[3] };
          return [qdepthwise(r, plan.asQ(n.input[0], x[0]!), dw, geom, act, out)];
        });
      }
    } else if (n.opType === "ReduceMean" || n.opType === "GlobalAveragePool") {
      plan.paramsOf(n.input[0]);
      plan.handlers.set(n, (x) => [qmean(r, plan.asQ(n.input[0], x[0]!))]);
    } else if (n.opType === "Add") {
      const pout = out ?? plan.paramsOf(n.output[0]);
      for (const i of n.input) plan.paramsOf(i);
      plan.handlers.set(n, (x) => [qadd(r, plan.asQ(n.input[0], x[0]!), plan.asQ(n.input[1], x[1]!), pout)]);
    } else if (n.opType === "Resize") {
      plan.paramsOf(n.input[0]);
      plan.handlers.set(n, (x) => [qresize2x(r, plan.asQ(n.input[0], x[0]!))]);
    } else if (n.opType === "Concat") {
      const pout = out ?? plan.paramsOf(n.output[0]);
      for (const i of n.input) plan.paramsOf(i);
      plan.handlers.set(n, (x) => [qconcat(r, n.input.map((i, k) => plan.asQ(i, x[k]!)), pout)]);
    } else if (n.opType === "MaxPool") {
      plan.paramsOf(n.input[0]);
      plan.handlers.set(n, (x) => [qmaxpool2x2same(r, plan.asQ(n.input[0], x[0]!))]);
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
    r.ar.pQScale(C, x.ptr + off, x.q.ptr.zp, x.q.ptr.scale, factor.ptr + n * C * 4, y.ptr + off, out.ptr.inv, out.ptr.zp, H * W);
  }
  return y;
}

// ---- glue between convolutions ------------------------------------------------

function qadd(r: Resident, a: QT, b: QT, out: QParams): QT {
  const [N, C, H, W] = a.dims;
  const y = allocQ(r, a.dims, out);
  for (let n = 0; n < N; n++) {
    const off = n * C * H * W;
    r.ar.pQAdd([C, a.ptr + off, a.q.ptr.zp, a.q.ptr.scale, b.ptr + off, b.q.ptr.zp, b.q.ptr.scale, y.ptr + off, out.ptr.inv, out.ptr.zp, H * W]);
  }
  return y;
}

/** Nearest 2x keeps the input's scale: every output byte is an input byte. */
function qresize2x(r: Resident, x: QT): QT {
  const [N, C, H, W] = x.dims;
  const y = allocQ(r, [N, C, 2 * H, 2 * W], x.q);
  for (let n = 0; n < N; n++) r.ar.pQResize2x(C, W, x.ptr + n * C * H * W, y.ptr + n * C * 4 * H * W, 2 * H);
  return y;
}

function qconcat(r: Resident, xs: QT[], out: QParams): QT {
  const [N, , H, W] = xs[0].dims;
  const C = xs.reduce((s, x) => s + x.dims[1], 0);
  const y = allocQ(r, [N, C, H, W], out);
  let off = 0;
  for (const x of xs) {
    for (let n = 0; n < N; n++) {
      r.ar.pQConcat([x.dims[1], C, off, x.ptr + n * x.dims[1] * H * W, x.q.ptr.zp, x.q.ptr.scale, y.ptr + n * C * H * W, out.ptr.inv, out.ptr.zp, H * W]);
    }
    off += x.dims[1];
  }
  return y;
}

/** Max commutes with dequantization, so the output keeps the input's parameters. */
function qmaxpool2x2same(r: Resident, x: QT): QT {
  const [N, C, H, W] = x.dims;
  const y = allocQ(r, x.dims, x.q);
  for (let n = 0; n < N; n++) r.ar.pQMaxPool(C, H, W, x.ptr + n * C * H * W, y.ptr + n * C * H * W);
  return y;
}
