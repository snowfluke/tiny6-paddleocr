// Folds the five-node Gelu that paddle2onnx emits into one node.
//
// Gelu exports as Div -> Erf -> Add -> Mul -> Mul. Every one of those is a
// full pass over a tensor that is megabytes wide at detection resolution, and
// elementwise ops are bandwidth-bound: measured on four threads, Mul gains
// 1.31x and Add 1.20x where Conv gains 2.46x. Threads cannot fix a pass that
// is waiting on memory, so the fix is to stop making five of them.
//
// Detection has 13 of these, recognition 4.

import type { OnnxGraph, OnnxNode, OnnxTensor } from "../onnx/parse.ts";

/** Matches ACT_RELU in kernels.rs. */
export const ACT_RELU = 1;

/** The lone value of a scalar initializer, or null if it is not one. */
function scalar(g: OnnxGraph, name: string): number | null {
  const t: OnnxTensor | undefined = g.initializers.get(name);
  if (!t || !(t.data instanceof Float32Array) || t.data.length !== 1) return null;
  return t.data[0];
}

export function fuseGelu(g: OnnxGraph): { graph: OnnxGraph; fused: number } {
  // A tensor may only be folded away if nothing outside the pattern reads it.
  const readers = new Map<string, number>();
  const bump = (n: string) => readers.set(n, (readers.get(n) ?? 0) + 1);
  for (const n of g.nodes) for (const i of n.input) bump(i);
  for (const o of g.outputs) bump(o.name);

  const producer = new Map<string, OnnxNode>();
  for (const n of g.nodes) for (const o of n.output) producer.set(o, n);

  const only = (name: string, op: string): OnnxNode | null => {
    if (readers.get(name) !== 1) return null;
    const c = g.nodes.find((n) => n.input.includes(name));
    return c && c.opType === op ? c : null;
  };

  const drop = new Set<OnnxNode>();
  const replace = new Map<OnnxNode, OnnxNode>();

  for (const div of g.nodes) {
    if (div.opType !== "Div" || drop.has(div)) continue;
    const x = div.input[0];
    const c1 = scalar(g, div.input[1]);
    if (c1 === null || c1 === 0) continue;

    const erf = only(div.output[0], "Erf");
    if (!erf) continue;
    const add = only(erf.output[0], "Add");
    if (!add || scalar(g, add.input[1]) !== 1) continue;
    const mul1 = only(add.output[0], "Mul");
    // The surviving multiply must take the very tensor the Div consumed.
    if (!mul1 || mul1.input[0] !== x) continue;
    const mul2 = only(mul1.output[0], "Mul");
    if (!mul2) continue;
    const c3 = scalar(g, mul2.input[1]);
    if (c3 === null) continue;

    for (const n of [div, erf, add, mul1]) drop.add(n);
    replace.set(mul2, {
      name: mul2.name,
      opType: "Gelu",
      input: [x],
      output: mul2.output,
      // gelu(x) = c3 * x * (1 + erf(x / c1)); the reciprocal folds the divide.
      attrs: new Map([
        ["scale", { name: "scale", type: 1, f: 1 / c1 }],
        ["post", { name: "post", type: 1, f: c3 }],
      ]),
    });
  }

  if (!drop.size) return { graph: g, fused: 0 };
  const nodes = g.nodes.filter((n) => !drop.has(n)).map((n) => replace.get(n) ?? n);
  return { graph: { ...g, nodes }, fused: replace.size };
}

/**
 * Drops Identity nodes by pointing their readers at the source tensor. The
 * recognition export puts one after every convolution, so without this the
 * Conv -> Add and Conv -> Gelu patterns below never see each other: an
 * Identity in between left recognition with no residual fusions at all.
 * An Identity that produces a graph output stays.
 */
export function dropIdentity(g: OnnxGraph): { graph: OnnxGraph; dropped: number } {
  const outputs = new Set(g.outputs.map((o) => o.name));
  const alias = new Map<string, string>();
  const nodes: OnnxNode[] = [];
  for (const n of g.nodes) {
    const input = n.input.map((i) => alias.get(i) ?? i);
    if (n.opType === "Identity" && !outputs.has(n.output[0])) {
      alias.set(n.output[0], input[0]);
      continue;
    }
    nodes.push(input.some((i, k) => i !== n.input[k]) ? { ...n, input } : n);
  }
  return alias.size ? { graph: { ...g, nodes }, dropped: alias.size } : { graph: g, dropped: 0 };
}

/**
 * Folds a BatchNormalization that is the sole reader of a convolution into
 * that convolution's weights and bias: w' = w * g / sqrt(v + eps),
 * b' = (b - mean) * g / sqrt(v + eps) + beta, per output channel. The
 * recognition export has two, on its first two convolutions, each a full
 * pass over the widest activations in the model.
 */
export function foldBatchNorm(g: OnnxGraph): { graph: OnnxGraph; folded: number } {
  const readers = readerCount(g);
  const initializers = new Map(g.initializers);
  const drop = new Set<OnnxNode>();
  const rewrite = new Map<OnnxNode, OnnxNode>();
  const f32 = (name: string): (OnnxTensor & { data: Float32Array }) | null => {
    const t = initializers.get(name);
    return t && t.data instanceof Float32Array ? (t as OnnxTensor & { data: Float32Array }) : null;
  };
  for (const conv of g.nodes) {
    if (conv.opType !== "Conv") continue;
    const bn = soleReader(g, readers, conv.output[0], "BatchNormalization");
    const w = f32(conv.input[1]);
    if (!bn || !w) continue;
    const [gamma, beta, mean, varr] = bn.input.slice(1).map(f32);
    if (!gamma || !beta || !mean || !varr) continue;
    const eps = bn.attrs.get("epsilon")?.f ?? 1e-5;
    const cout = w.dims[0];
    const per = w.data.length / cout;
    const wd = new Float32Array(w.data.length);
    const bd = new Float32Array(cout);
    const b0 = conv.input[2] ? f32(conv.input[2]) : null;
    for (let c = 0; c < cout; c++) {
      const k = gamma.data[c] / Math.sqrt(varr.data[c] + eps);
      for (let i = 0; i < per; i++) wd[c * per + i] = w.data[c * per + i] * k;
      bd[c] = ((b0 ? b0.data[c] : 0) - mean.data[c]) * k + beta.data[c];
    }
    const wn = `${w.name}_bn`, bname = `${conv.output[0]}_bn_bias`;
    initializers.set(wn, { name: wn, dims: w.dims, dataType: w.dataType, data: wd });
    initializers.set(bname, { name: bname, dims: [cout], dataType: 1, data: bd });
    rewrite.set(conv, { ...conv, input: [conv.input[0], wn, bname], output: bn.output });
    drop.add(bn);
  }
  if (!drop.size) return { graph: g, folded: 0 };
  const nodes = g.nodes.filter((n) => !drop.has(n)).map((n) => rewrite.get(n) ?? n);
  return { graph: { ...g, nodes, initializers }, folded: drop.size };
}

/** How many times each tensor is read, counting graph outputs as a reader. */
function readerCount(g: OnnxGraph): Map<string, number> {
  const readers = new Map<string, number>();
  const bump = (n: string) => readers.set(n, (readers.get(n) ?? 0) + 1);
  for (const n of g.nodes) for (const i of n.input) bump(i);
  for (const o of g.outputs) bump(o.name);
  return readers;
}

/** The single node reading `name`, if there is exactly one and it is `op`. */
function soleReader(g: OnnxGraph, readers: Map<string, number>, name: string, op: string) {
  if (readers.get(name) !== 1) return null;
  const c = g.nodes.find((n) => n.input.includes(name));
  return c && c.opType === op ? c : null;
}

/**
 * Folds a per-channel Add after a convolution into the convolution's bias, and
 * a following Relu into its epilogue.
 *
 * paddle2onnx gives the recognition model no conv biases at all: all 37 of its
 * convolutions emit a bare product and a separate Add of a [1, C, 1, 1]
 * constant, 33 of which are a sole reader. The GEMM applies bias in the same
 * pass that writes its output, so each of those Adds is a full read and write
 * of the activation for nothing. Relu is the same argument.
 *
 * A residual Add, where both inputs are activations, is not this and is left
 * alone; detection's 10 post-conv Adds are all that kind.
 */
export function fuseConvEpilogue(g: OnnxGraph): { graph: OnnxGraph; bias: number; act: number; residual: number } {
  const readers = readerCount(g);
  const drop = new Set<OnnxNode>();
  const rewrite = new Map<OnnxNode, OnnxNode>();
  const initializers = new Map(g.initializers);
  let bias = 0;
  let act = 0;
  let residual = 0;

  for (const conv of g.nodes) {
    if (conv.opType !== "Conv") continue;
    const cout = g.initializers.get(conv.input[1])?.dims[0];
    if (cout === undefined) continue;

    let node = conv;
    const add = soleReader(g, readers, node.output[0], "Add");
    if (add && node.input.length <= 2) {
      const other = add.input.find((i) => i !== node.output[0])!;
      const t = initializers.get(other);
      // A bias is one value per output channel, whatever rank it is shaped as.
      if (t && t.data instanceof Float32Array && t.data.length === cout) {
        const name = `${other}_as_bias`;
        initializers.set(name, { name, dims: [cout], dataType: t.dataType, data: t.data });
        node = { ...node, input: [...node.input, name], output: add.output };
        drop.add(add);
        bias++;
      }
    }

    // A residual Add, both inputs activations, rides along as a fourth input
    // and is summed in the GEMM epilogue. onnxruntime calls this Conv Add
    // Activation fusion. Only 1x1 convolutions get it here; the runtime
    // throws if the shapes disagree rather than silently misreading.
    const w = g.initializers.get(node.input[1]);
    const resAdd = soleReader(g, readers, node.output[0], "Add");
    if (resAdd && w && w.dims[2] === 1 && w.dims[3] === 1) {
      const other = resAdd.input.find((i) => i !== node.output[0])!;
      if (!initializers.has(other) && g.nodes.findIndex((m) => m.output.includes(other)) < g.nodes.indexOf(conv)) {
        node = { ...node, input: [node.input[0], node.input[1], node.input[2] ?? "", other], output: resAdd.output };
        drop.add(resAdd);
        residual++;
      }
    }

    const relu = soleReader(g, readers, node.output[0], "Relu");
    if (relu) {
      node = {
        ...node,
        output: relu.output,
        attrs: new Map(node.attrs).set("activation", { name: "activation", type: 2, i: ACT_RELU }),
      };
      drop.add(relu);
      act++;
    }

    if (node !== conv) rewrite.set(conv, node);
  }

  if (!drop.size) return { graph: g, bias, act, residual };
  const nodes = g.nodes.filter((n) => !drop.has(n)).map((n) => rewrite.get(n) ?? n);
  return { graph: { ...g, nodes, initializers }, bias, act, residual };
}
