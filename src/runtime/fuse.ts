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
export function fuseConvEpilogue(g: OnnxGraph): { graph: OnnxGraph; bias: number; act: number } {
  const readers = readerCount(g);
  const drop = new Set<OnnxNode>();
  const rewrite = new Map<OnnxNode, OnnxNode>();
  const initializers = new Map(g.initializers);
  let bias = 0;
  let act = 0;

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

  if (!drop.size) return { graph: g, bias, act };
  const nodes = g.nodes.filter((n) => !drop.has(n)).map((n) => rewrite.get(n) ?? n);
  return { graph: { ...g, nodes, initializers }, bias, act };
}
