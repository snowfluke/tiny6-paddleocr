// Rewrite an ONNX model so every intermediate tensor is also a graph output.
// Lets ORT dump per-node reference values we can diff our runtime against.

import { Reader } from "../src/onnx/reader.ts";
import { parseOnnx } from "../src/onnx/parse.ts";

const vi = (n: number): number[] => {
  const o: number[] = [];
  while (n > 127) {
    o.push((n & 0x7f) | 0x80);
    n >>>= 7;
  }
  o.push(n);
  return o;
};
const lenDelim = (field: number, body: ArrayLike<number>): number[] => [
  ...vi((field << 3) | 2),
  ...vi(body.length),
  ...body,
];

/** A ValueInfoProto carrying only a name. ORT accepts these as outputs. */
const namedOutput = (name: string) => lenDelim(12, lenDelim(1, [...new TextEncoder().encode(name)]));

export function exposeAllTensors(buf: Uint8Array): { model: Uint8Array; names: string[] } {
  const r = new Reader(buf);
  const graph = parseOnnx(buf);

  let head: number[] = [];
  let graphBody: Uint8Array | null = null;
  for (const f of r.fields(0, buf.length)) {
    if (f.no === 7) graphBody = buf.subarray(f.start, f.end);
    else if (f.no === 1) head.push(...vi((1 << 3) | 0), ...buf.subarray(f.start, f.end));
    else if (f.no === 8) head.push(...lenDelim(8, buf.subarray(f.start, f.end)));
  }
  if (!graphBody) throw new Error("no graph in model");

  const already = new Set(graph.outputs.map((o) => o.name));
  const names: string[] = [];
  const extra: number[] = [];
  for (const n of graph.nodes) {
    for (const out of n.output) {
      if (!out || already.has(out)) continue;
      already.add(out);
      names.push(out);
      extra.push(...namedOutput(out));
    }
  }

  const body = new Uint8Array(graphBody.length + extra.length);
  body.set(graphBody, 0);
  body.set(extra, graphBody.length);
  return { model: new Uint8Array([...head, ...lenDelim(7, body)]), names };
}

if (import.meta.main) {
  const src = process.argv[2];
  const dst = process.argv[3];
  const { model, names } = exposeAllTensors(new Uint8Array(await Bun.file(src).arrayBuffer()));
  await Bun.write(dst, model);
  console.log(`${dst}: +${names.length} exposed tensors, ${(model.length / 1e6).toFixed(2)} MB`);
}
