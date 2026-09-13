import type { OnnxGraph, OnnxNode, OnnxTensor } from "../onnx/parse.ts";
import { binary, broadcastShape, type Tensor } from "./tensor.ts";

import {
  batchNorm,
  conv2d,
  type ConvAttrs,
  convTranspose2d,
  globalAveragePool,
  matmul,
  type PoolAttrs,
  pool2d,
} from "../ops/nn.ts";
import { concat, reduceMean, resizeNearest, softmax, squeeze, transpose, unsqueeze } from "../ops/shape.ts";
import { dequantizeLinear, quantizeLinear } from "../ops/quant.ts";
import { convResident, convTranspose2x2Resident, matmulResident } from "../ops/conv-wasm.ts";
import { BIN_OP, lastUseMap, Resident, UN_OP, type RT } from "./resident.ts";
import { ACT_RELU, dropIdentity, foldBatchNorm, fuseConvEpilogue, fuseGelu } from "./fuse.ts";
import { buildPlan, type Calibration, isQ, type QPlan, type QVal } from "./qplan.ts";
import { binaryFast, erfFast, geluFast, hardSigmoidFast, maxPool2x2Same, reduceMeanTrailing, reluFast, resizeNearestFast, sigmoidFast } from "../ops/fast.ts";
import type { Arena } from "../wasm/backend.ts";

function toTensor(t: OnnxTensor): Tensor {
  if (t.data instanceof Float32Array) return { dims: t.dims, data: t.data };
  const f = new Float32Array(t.data.length);
  for (let i = 0; i < f.length; i++) f[i] = Number((t.data as ArrayLike<number | bigint>)[i]);
  return { dims: t.dims, data: f };
}

const pair = (v: number[] | undefined, d: number): [number, number] =>
  v && v.length >= 2 ? [v[0], v[1]] : [d, d];

const quad = (v: number[] | undefined): [number, number, number, number] =>
  v && v.length >= 4 ? [v[0], v[1], v[2], v[3]] : [0, 0, 0, 0];

/**
 * auto_pad=SAME_* asks for an output the same size as the input divided by the
 * stride, so the padding depends on the input and cannot be baked into attrs.
 */
function resolvePads(
  n: OnnxNode,
  inHW: [number, number],
  k: [number, number],
  s: [number, number],
  d: [number, number],
): [number, number, number, number] {
  const autoPad = n.attrs.get("auto_pad")?.s ?? "NOTSET";
  if (autoPad === "NOTSET") return quad(n.attrs.get("pads")?.ints);
  if (autoPad === "VALID") return [0, 0, 0, 0];
  const axis = (i: number) => {
    const out = Math.ceil(inHW[i] / s[i]);
    const need = Math.max(0, (out - 1) * s[i] + (k[i] - 1) * d[i] + 1 - inHW[i]);
    const lo = Math.floor(need / 2);
    return autoPad === "SAME_UPPER" ? [lo, need - lo] : [need - lo, lo];
  };
  const [t, b] = axis(0);
  const [l, r] = axis(1);
  return [t, l, b, r];
}

/**
 * Attributes only depend on the input when auto_pad is SAME_*, which no Conv
 * in either model uses. Caching the rest keeps 120 object and array
 * allocations out of every graph run.
 */
const convAttrCache = new WeakMap<OnnxNode, ConvAttrs>();
const poolAttrCache = new WeakMap<OnnxNode, PoolAttrs>();

const padsDependOnInput = (n: OnnxNode) => {
  const ap = n.attrs.get("auto_pad")?.s ?? "NOTSET";
  return ap !== "NOTSET" && ap !== "VALID";
};

export function convAttrs(n: OnnxNode, x: { dims: number[] }, kernelFrom: number[]): ConvAttrs {
  const hit = convAttrCache.get(n);
  if (hit) return hit;
  const built = buildConvAttrs(n, x, kernelFrom);
  if (!padsDependOnInput(n)) convAttrCache.set(n, built);
  return built;
}

function poolAttrs(n: OnnxNode, x: { dims: number[] }): PoolAttrs {
  const hit = poolAttrCache.get(n);
  if (hit) return hit;
  const built = buildPoolAttrs(n, x);
  if (!padsDependOnInput(n)) poolAttrCache.set(n, built);
  return built;
}

function buildConvAttrs(n: OnnxNode, x: { dims: number[] }, kernelFrom: number[]): ConvAttrs {
  const ks = n.attrs.get("kernel_shape")?.ints ?? kernelFrom;
  const k: [number, number] = [ks[0], ks[1]];
  const dilations = pair(n.attrs.get("dilations")?.ints, 1);
  const strides = pair(n.attrs.get("strides")?.ints, 1);
  const inHW: [number, number] = [x.dims[2], x.dims[3]];
  return {
    kernel: k,
    strides,
    pads: resolvePads(n, inHW, k, strides, dilations),
    dilations,
    group: n.attrs.get("group")?.i ?? 1,
  };
}

function buildPoolAttrs(n: OnnxNode, x: { dims: number[] }): PoolAttrs {
  const ks = n.attrs.get("kernel_shape")?.ints ?? [1, 1];
  const k: [number, number] = [ks[0], ks[1]];
  const strides = pair(n.attrs.get("strides")?.ints, 1);
  return {
    kernel: k,
    strides,
    pads: resolvePads(n, [x.dims[2], x.dims[3]], k, strides, [1, 1]),
    ceilMode: (n.attrs.get("ceil_mode")?.i ?? 0) !== 0,
    countIncludePad: (n.attrs.get("count_include_pad")?.i ?? 0) !== 0,
  };
}

/** `axes` moved from attribute to input in opset 13; accept either. */
function axesOf(n: OnnxNode, x: (Tensor | null)[]): number[] {
  const attr = n.attrs.get("axes")?.ints;
  if (attr) return attr;
  return x[1] ? [...x[1].data] : [];
}

export type SessionOptions = {
  /** Default true. False runs the graph node for node as exported. */
  fuse?: boolean;
  /**
   * Per-channel activation ranges from tools/calibrate.ts. With them the
   * convolutions the planner can handle run on int8; needs an arena.
   */
  int8?: Calibration;
};

export type RunOptions = {
  /** Called after every node, for golden-diffing. Downloads every output. */
  onNode?: (node: OnnxNode, outputs: Tensor[]) => void;
  /**
   * Called after every node without materialising anything. Use this to time
   * nodes: onNode copies each output out of wasm memory, which on the resident
   * path costs more than most of the ops being measured.
   */
  onNodeDone?: (node: OnnxNode) => void;
};

export class Session {
  private readonly consts = new Map<string, Tensor>();
  private readonly res?: Resident;
  private readonly plan?: QPlan;
  private readonly residentConsts = new Map<string, RT>();
  private readonly lastUse: Map<string, number>;
  /**
   * Per node, the input names whose buffer dies here. Precomputed because the
   * run loop is hot: 219 nodes for recognition, and a crop takes about 13 ms,
   * so a Set allocation and a map lookup per node per input is real time.
   */
  private readonly releaseAt: string[][] = [];
  /** Reused across nodes so the input list is not reallocated 219 times. */
  private readonly scratchInputs: (QVal | null)[] = [];

  readonly graph: OnnxGraph;

  /**
   * Without an arena every op runs in TypeScript, which is the reference path.
   * Pass `fuse: false` to execute the graph exactly as exported, which is what
   * the golden check does so it still sees every intermediate tensor.
   */
  constructor(graph: OnnxGraph, arena?: Arena, opts: SessionOptions = {}) {
    graph = opts.fuse === false ? graph : fuseConvEpilogue(fuseGelu(foldBatchNorm(dropIdentity(graph).graph).graph).graph).graph;
    for (const [name, t] of graph.initializers) this.consts.set(name, toTensor(t));
    if (arena) {
      // Weights upload now, before any activation, so the arena can seal the
      // boundary between what persists and what a run may recycle.
      this.res = new Resident(arena);
      const mode = arena.dotMode;
      if (opts.int8 && mode !== "none") {
        const built = buildPlan(graph, opts.int8, this.res, this.consts, mode);
        graph = built.graph;
        this.plan = built.plan;
      }
      for (const [name, t] of this.consts) {
        this.residentConsts.set(name, this.res.constant(t.data, t.dims));
      }
    }
    this.graph = graph;
    this.lastUse = lastUseMap(graph.nodes, graph.outputs.map((o) => o.name));
    graph.nodes.forEach((n, i) => {
      const dying = [...new Set(n.input)].filter((name) => name && this.lastUse.get(name) === i);
      this.releaseAt.push(dying);
    });
  }

  /** How many nodes the int8 plan covers; zero without a calibration. */
  get int8Nodes(): number {
    return this.plan?.handlers.size ?? 0;
  }

  planned(n: OnnxNode): boolean {
    return this.plan?.handlers.has(n) ?? false;
  }

  run(feeds: Record<string, Tensor>, opts: RunOptions = {}): Map<string, Tensor> {
    if (this.res) return this.runResident(this.res, feeds, opts);
    return this.runTs(feeds, opts);
  }

  private runResident(r: Resident, feeds: Record<string, Tensor>, opts: RunOptions): Map<string, Tensor> {
    r.ar.beginRun();
    this.plan?.beginRun();
    const env = new Map<string, QVal>(this.residentConsts);
    for (const [k, v] of Object.entries(feeds)) {
      const rt = r.upload(v);
      r.ar.retain(rt.ptr);
      env.set(k, rt);
    }

    const nodes = this.graph.nodes;
    const inputs = this.scratchInputs;
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const handler = this.plan?.handlers.get(node);
      inputs.length = node.input.length;
      for (let j = 0; j < node.input.length; j++) {
        const name = node.input[j];
        if (name === "") {
          inputs[j] = null;
          continue;
        }
        let t = env.get(name);
        if (!t) throw new Error(`${node.opType} "${node.name}": missing input ${name}`);
        // An unplanned node reading an int8 tensor gets it back as fp32 once;
        // the copy takes the name so later readers and the release see it.
        if (isQ(t) && !handler) {
          t = this.plan!.asRT(t);
          env.set(name, t);
        }
        inputs[j] = t;
      }
      const outs = handler ? handler(inputs) : this.execResident(r, node, inputs as (RT | null)[]);
      for (let j = 0; j < node.output.length; j++) {
        if (!node.output[j]) continue;
        env.set(node.output[j], outs[j]);
        r.ar.retain(outs[j].ptr);
      }
      if (opts.onNode) opts.onNode(node, outs.map((o) => r.download(this.plan ? this.plan.asRT(o) : (o as RT))));
      if (opts.onNodeDone) opts.onNodeDone(node);
      // A buffer goes back on the free list once its last reader has run.
      for (const name of this.releaseAt[i]) {
        const t = env.get(name);
        if (t) r.ar.release(t.ptr);
      }
    }

    const result = new Map<string, Tensor>();
    for (const o of this.graph.outputs) {
      const t = env.get(o.name);
      if (t) result.set(o.name, r.download(this.plan ? this.plan.asRT(t) : (t as RT)));
    }
    return result;
  }

  private runTs(feeds: Record<string, Tensor>, opts: RunOptions = {}): Map<string, Tensor> {
    const env = new Map<string, Tensor>(this.consts);
    for (const [k, v] of Object.entries(feeds)) env.set(k, v);

    for (const node of this.graph.nodes) {
      const inputs = node.input.map((n) => {
        if (n === "") return null;
        const t = env.get(n);
        if (!t) throw new Error(`${node.opType} "${node.name}": missing input ${n}`);
        return t;
      });
      const outs = this.exec(node, inputs);
      for (let i = 0; i < node.output.length; i++) {
        if (node.output[i]) env.set(node.output[i], outs[i]);
      }
      opts.onNode?.(node, outs);
    }

    const result = new Map<string, Tensor>();
    for (const o of this.graph.outputs) {
      const t = env.get(o.name);
      if (t) result.set(o.name, t);
    }
    return result;
  }


  /**
   * Resident dispatch. Ops with a WASM kernel run on pointers; the rest fall
   * back to the TypeScript implementation, paying a download and upload of
   * every input. Keep hot ops off that path: Concat used to take it and cost
   * 5.5% of detection at 960x960.
   */
  private execResident(r: Resident, n: OnnxNode, x: (RT | null)[]): RT[] {
    const a = x[0]!;
    const fallback = (): RT[] => {
      const ts = x.map((i) => (i ? r.download(i) : null));
      return this.exec(n, ts).map((t) => r.upload(t));
    };
    const axes = (): number[] => {
      const attr = n.attrs.get("axes")?.ints;
      if (attr) return attr;
      return x[1] ? [...r.download(x[1]).data] : [];
    };

    switch (n.opType) {
      case "Conv":
        return [convResident(
          r,
          a,
          x[1]!,
          x[2] ?? null,
          convAttrs(n, a, x[1]!.dims.slice(2)),
          n.attrs.get("activation")?.i ?? 0,
          x[3] ?? null,
        )];
      case "ConvTranspose": {
        const at = convAttrs(n, a, x[1]!.dims.slice(2));
        const k2s2 = at.kernel[0] === 2 && at.kernel[1] === 2 && at.strides[0] === 2 &&
          at.strides[1] === 2 && at.group === 1 && at.pads.every((p) => p === 0);
        return k2s2 ? [convTranspose2x2Resident(r, a, x[1]!, x[2] ?? null)] : fallback();
      }
      case "MatMul":
        return [matmulResident(r, a, x[1]!)];
      case "QuantizeLinear":
      case "DequantizeLinear":
        return fallback();
      case "Add":
      case "Sub":
      case "Mul":
      case "Div": {
        const op = BIN_OP[n.opType.toLowerCase() as keyof typeof BIN_OP];
        const dims = broadcastShape(a.dims, x[1]!.dims);
        const out = r.binary(op, a, x[1]!, dims);
        return out ? [out] : fallback();
      }
      case "Relu":
        return [r.unary(UN_OP.relu, a)];
      case "Sigmoid":
        return [r.unary(UN_OP.sigmoid, a)];
      case "Erf":
        return [r.unary(UN_OP.erf, a)];
      case "HardSigmoid":
        return [r.unary(UN_OP.hardSigmoid, a, n.attrs.get("alpha")?.f ?? 0.2, n.attrs.get("beta")?.f ?? 0.5)];
      case "Gelu":
        return [r.unary(UN_OP.gelu, a, n.attrs.get("scale")!.f!, n.attrs.get("post")!.f!)];
      case "Identity":
        return [a];
      case "Squeeze": {
        const drop = new Set(axes().map((v) => (v < 0 ? v + a.dims.length : v)));
        const dims = a.dims.filter((_, i) => !drop.has(i));
        return [r.view(a, dims.length ? dims : [1])];
      }
      case "Unsqueeze": {
        const rank = a.dims.length + axes().length;
        const add = new Set(axes().map((v) => (v < 0 ? v + rank : v)));
        const dims: number[] = [];
        let j = 0;
        for (let i = 0; i < rank; i++) dims.push(add.has(i) ? 1 : a.dims[j++]);
        return [r.view(a, dims)];
      }
      case "MaxPool": {
        const at = poolAttrs(n, a);
        const same2x2 = at.kernel[0] === 2 && at.kernel[1] === 2 && at.strides[0] === 1 &&
          at.strides[1] === 1 && at.pads[0] === 0 && at.pads[1] === 0;
        return same2x2 ? [r.maxPool2x2(a)] : fallback();
      }
      case "GlobalAveragePool":
        return [r.reduceMean(a, [2, 3], true)!];
      case "ReduceMean": {
        const keep = (n.attrs.get("keepdims")?.i ?? 1) !== 0;
        return [r.reduceMean(a, axes(), keep) ?? fallback()[0]];
      }
      case "Concat": {
        const parts = x.filter((t): t is RT => t !== null);
        const rank = parts[0].dims.length;
        const ax = ((n.attrs.get("axis")?.i ?? 0) + rank) % rank;
        const dims = parts[0].dims.slice();
        dims[ax] = parts.reduce((sum, p) => sum + p.dims[ax], 0);
        return [r.concat(parts, ax, dims)];
      }
      case "BatchNormalization":
        return [r.batchNorm(a, x[1]!, x[2]!, x[3]!, x[4]!, n.attrs.get("epsilon")?.f ?? 1e-5)];
      case "Softmax": {
        const ax = n.attrs.get("axis")?.i ?? -1;
        const last = ax === -1 || ax === a.dims.length - 1;
        return last ? [r.softmaxLast(a)] : fallback();
      }
      case "Transpose": {
        const perm = n.attrs.get("perm")?.ints ?? a.dims.map((_, i) => a.dims.length - 1 - i);
        return [r.transpose(a, perm) ?? fallback()[0]];
      }
      case "Resize": {
        const scales = x[2] ? [...r.download(x[2]).data] : [];
        const ok = scales.length === 4 && scales[0] === 1 && scales[1] === 1 &&
          Number.isInteger(scales[2]) && Number.isInteger(scales[3]);
        return ok ? [r.resizeNearest(a, scales[2], scales[3])] : fallback();
      }
      default:
        return fallback();
    }
  }

  private exec(n: OnnxNode, x: (Tensor | null)[]): Tensor[] {
    const a = x[0]!;
    switch (n.opType) {
      case "Conv": {
        const at = convAttrs(n, a, x[1]!.dims.slice(2));
        const y0 = conv2d(a, x[1]!, x[2] ?? null, at);
        const y = x[3] ? binaryFast(y0, x[3], "add") : y0;
        return [n.attrs.get("activation")?.i === ACT_RELU ? reluFast(y) : y];
      }
      case "ConvTranspose":
        return [convTranspose2d(a, x[1]!, x[2] ?? null, convAttrs(n, a, x[1]!.dims.slice(2)))];
      case "MatMul":
        return [matmul(a, x[1]!)];
      case "QuantizeLinear":
        return [quantizeLinear(a, x[1]!, x[2] ?? null, n.attrs.get("axis")?.i ?? 1)];
      case "DequantizeLinear":
        return [dequantizeLinear(a, x[1]!, x[2] ?? null, n.attrs.get("axis")?.i ?? 1)];
      case "Add":
        return [binaryFast(a, x[1]!, "add")];
      case "Sub":
        return [binaryFast(a, x[1]!, "sub")];
      case "Mul":
        return [binaryFast(a, x[1]!, "mul")];
      case "Div":
        return [binaryFast(a, x[1]!, "div")];
      case "Relu":
        return [reluFast(a)];
      case "Sigmoid":
        return [sigmoidFast(a)];
      case "Erf":
        return [erfFast(a)];
      case "HardSigmoid":
        return [hardSigmoidFast(a, n.attrs.get("alpha")?.f ?? 0.2, n.attrs.get("beta")?.f ?? 0.5)];
      case "Gelu":
        return [geluFast(a, n.attrs.get("scale")!.f!, n.attrs.get("post")!.f!)];
      case "Identity":
        return [a];
      case "MaxPool": {
        const at = poolAttrs(n, a);
        const same2x2 = at.kernel[0] === 2 && at.kernel[1] === 2 && at.strides[0] === 1 &&
          at.strides[1] === 1 && at.pads[0] === 0 && at.pads[1] === 0;
        return [(same2x2 ? maxPool2x2Same(a) : null) ?? pool2d(a, at, "max")];
      }
      case "AveragePool":
        return [pool2d(a, poolAttrs(n, a), "avg")];
      case "GlobalAveragePool":
        return [globalAveragePool(a)];
      case "BatchNormalization":
        return [batchNorm(a, x[1]!, x[2]!, x[3]!, x[4]!, n.attrs.get("epsilon")?.f ?? 1e-5)];
      case "Concat":
        return [concat(x as Tensor[], n.attrs.get("axis")?.i ?? 0)];
      case "Transpose":
        return [transpose(a, n.attrs.get("perm")?.ints ?? a.dims.map((_, i) => a.dims.length - 1 - i))];
      case "Squeeze":
        return [squeeze(a, axesOf(n, x))];
      case "Unsqueeze":
        return [unsqueeze(a, axesOf(n, x))];
      case "ReduceMean": {
        const ax = axesOf(n, x);
        const keep = (n.attrs.get("keepdims")?.i ?? 1) !== 0;
        return [reduceMeanTrailing(a, ax, keep) ?? reduceMean(a, ax, keep)];
      }
      case "Softmax":
        return [softmax(a, n.attrs.get("axis")?.i ?? -1)];
      case "Resize":
        return [resizeNearestFast(a, [...(x[2]?.data ?? [])]) ?? resizeNearest(a, [...(x[2]?.data ?? [])])];
      default:
        throw new Error(`unimplemented op ${n.opType}`);
    }
  }
}
