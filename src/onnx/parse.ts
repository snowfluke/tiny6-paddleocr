// ONNX ModelProto -> a graph shape the runtime can execute.
// Field numbers come from onnx.proto3.

import { Reader, type Field } from "./reader.ts";

export type DataType = 1 | 6 | 7 | 9 | 11; // FLOAT INT32 INT64 BOOL DOUBLE

export type OnnxTensor = {
  name: string;
  dims: number[];
  dataType: number;
  data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
};

export type Attr = {
  name: string;
  type: number;
  f?: number;
  i?: number;
  s?: string;
  floats?: Float32Array;
  ints?: number[];
  t?: OnnxTensor;
};

export type OnnxNode = {
  name: string;
  opType: string;
  input: string[];
  output: string[];
  attrs: Map<string, Attr>;
};

export type OnnxGraph = {
  nodes: OnnxNode[];
  initializers: Map<string, OnnxTensor>;
  inputs: { name: string; dims: (number | string)[] }[];
  outputs: { name: string; dims: (number | string)[] }[];
};

const ATTR_FLOAT = 1;
const ATTR_INT = 2;
const ATTR_STRING = 3;
const ATTR_TENSOR = 4;
const ATTR_FLOATS = 6;
const ATTR_INTS = 7;

function parseTensor(r: Reader, f: Field): OnnxTensor {
  const dims: number[] = [];
  let dataType = 0;
  let name = "";
  let raw: Uint8Array | null = null;
  let floatData: Float32Array | null = null;
  let int64Data: number[] | null = null;
  let int32Data: number[] | null = null;

  for (const g of r.fields(f.start, f.end)) {
    if (g.no === 1) g.wire === 2 ? dims.push(...r.packedInts(g)) : dims.push(r.int(g));
    else if (g.no === 2) dataType = r.int(g);
    else if (g.no === 4) floatData = r.packedF32(g);
    else if (g.no === 5) int32Data = [...(int32Data ?? []), ...r.packedInts(g)];
    else if (g.no === 7) int64Data = [...(int64Data ?? []), ...(g.wire === 2 ? r.packedInts(g) : [r.int(g)])];
    else if (g.no === 8) name = r.str(g);
    else if (g.no === 9) raw = r.bytes(g);
    else if (g.no === 13) throw new Error(`tensor ${name} uses external data`);
  }

  const count = dims.reduce((a, b) => a * b, 1);
  let data: OnnxTensor["data"];
  if (raw) {
    // raw_data is little-endian and not guaranteed aligned, so copy.
    const copy = raw.slice();
    if (dataType === 1) data = new Float32Array(copy.buffer, copy.byteOffset, count);
    else if (dataType === 7) data = new BigInt64Array(copy.buffer, copy.byteOffset, count);
    else if (dataType === 6) data = new Int32Array(copy.buffer, copy.byteOffset, count);
    else data = copy;
  } else if (floatData) data = floatData;
  else if (int64Data) data = BigInt64Array.from(int64Data.map(BigInt));
  else if (int32Data) data = Int32Array.from(int32Data);
  else data = new Float32Array(0);

  return { name, dims, dataType, data };
}

function parseAttr(r: Reader, f: Field): Attr {
  const a: Attr = { name: "", type: 0 };
  const ints: number[] = [];
  const floats: number[] = [];
  for (const g of r.fields(f.start, f.end)) {
    if (g.no === 1) a.name = r.str(g);
    else if (g.no === 20) a.type = r.int(g);
    else if (g.no === 2) a.f = r.f32(g);
    else if (g.no === 3) a.i = r.int(g);
    else if (g.no === 4) a.s = r.str(g);
    else if (g.no === 5) a.t = parseTensor(r, g);
    // proto3 lets a producer send repeated scalars packed or one field at a
    // time. paddle2onnx sends them unpacked, so both forms have to accumulate.
    else if (g.no === 7) g.wire === 2 ? floats.push(...r.packedF32(g)) : floats.push(r.f32(g));
    else if (g.no === 8) g.wire === 2 ? ints.push(...r.packedInts(g)) : ints.push(r.int(g));
  }
  if (ints.length) a.ints = ints;
  if (floats.length) a.floats = Float32Array.from(floats);
  // Some producers omit `type`; infer it from which value showed up.
  if (!a.type) {
    if (a.ints) a.type = ATTR_INTS;
    else if (a.floats) a.type = ATTR_FLOATS;
    else if (a.t) a.type = ATTR_TENSOR;
    else if (a.s !== undefined) a.type = ATTR_STRING;
    else if (a.i !== undefined) a.type = ATTR_INT;
    else if (a.f !== undefined) a.type = ATTR_FLOAT;
  }
  return a;
}

function parseNode(r: Reader, f: Field): OnnxNode {
  const n: OnnxNode = { name: "", opType: "", input: [], output: [], attrs: new Map() };
  for (const g of r.fields(f.start, f.end)) {
    if (g.no === 1) n.input.push(r.str(g));
    else if (g.no === 2) n.output.push(r.str(g));
    else if (g.no === 3) n.name = r.str(g);
    else if (g.no === 4) n.opType = r.str(g);
    else if (g.no === 5) {
      const a = parseAttr(r, g);
      n.attrs.set(a.name, a);
    }
  }
  return n;
}

function parseValueInfo(r: Reader, f: Field) {
  let name = "";
  const dims: (number | string)[] = [];
  for (const g of r.fields(f.start, f.end)) {
    if (g.no === 1) name = r.str(g);
    else if (g.no === 2) {
      for (const t of r.fields(g.start, g.end)) {
        if (t.no !== 1) continue; // tensor_type
        for (const tt of r.fields(t.start, t.end)) {
          if (tt.no !== 2) continue; // shape
          for (const sh of r.fields(tt.start, tt.end)) {
            if (sh.no !== 1) continue; // dim
            for (const d of r.fields(sh.start, sh.end)) {
              if (d.no === 1) dims.push(r.int(d));
              else if (d.no === 2) dims.push(r.str(d));
            }
          }
        }
      }
    }
  }
  return { name, dims };
}

export function parseOnnx(buf: Uint8Array): OnnxGraph {
  const r = new Reader(buf);
  const graph: OnnxGraph = { nodes: [], initializers: new Map(), inputs: [], outputs: [] };

  for (const top of r.fields(0, buf.length)) {
    if (top.no !== 7) continue; // ModelProto.graph
    for (const g of r.fields(top.start, top.end)) {
      if (g.no === 1) graph.nodes.push(parseNode(r, g));
      else if (g.no === 5) {
        const t = parseTensor(r, g);
        graph.initializers.set(t.name, t);
      } else if (g.no === 11) graph.inputs.push(parseValueInfo(r, g));
      else if (g.no === 12) graph.outputs.push(parseValueInfo(r, g));
    }
  }

  // Graph inputs also list initializers in some exports; keep only real feeds.
  graph.inputs = graph.inputs.filter((i) => !graph.initializers.has(i.name));
  return graph;
}
