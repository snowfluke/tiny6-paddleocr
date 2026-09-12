// Dump ORT reference values for every intermediate tensor, so the hand-written
// runtime can be diffed node by node. onnxruntime-node is a dev dependency and
// never ships; this runs once and the output is committed.

import * as ort from "onnxruntime-node";

/** Deterministic input so goldens are reproducible without an image decoder. */
export function seededInput(n: number, seed = 1): Float32Array {
  const a = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s / 0xffffffff) * 2 - 1;
  }
  return a;
}

type Dump = { shape: number[]; data: Float32Array };

export async function dump(modelPath: string, feeds: Record<string, ort.Tensor>) {
  const s = await ort.InferenceSession.create(modelPath, { graphOptimizationLevel: "disabled" });
  const out = await s.run(feeds);
  const result = new Map<string, Dump>();
  for (const [name, t] of Object.entries(out)) {
    if (t.type !== "float32") continue;
    result.set(name, { shape: t.dims as number[], data: t.data as Float32Array });
  }
  return result;
}

/** One flat file: [count][name,shape,data]... so the test side needs no deps. */
export function pack(m: Map<string, Dump>): Uint8Array {
  const parts: Uint8Array[] = [];
  const head = new Uint8Array(4);
  new DataView(head.buffer).setUint32(0, m.size, true);
  parts.push(head);
  for (const [name, d] of m) {
    const nameBytes = new TextEncoder().encode(name);
    const meta = new Uint8Array(4 + nameBytes.length + 4 + d.shape.length * 4);
    const dv = new DataView(meta.buffer);
    dv.setUint32(0, nameBytes.length, true);
    meta.set(nameBytes, 4);
    let o = 4 + nameBytes.length;
    dv.setUint32(o, d.shape.length, true);
    o += 4;
    for (const s of d.shape) {
      dv.setInt32(o, s, true);
      o += 4;
    }
    parts.push(meta, new Uint8Array(d.data.buffer, d.data.byteOffset, d.data.byteLength));
  }
  const total = parts.reduce((a, b) => a + b.length, 0);
  const buf = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    buf.set(p, o);
    o += p.length;
  }
  return buf;
}

if (import.meta.main) {
  const cases = [
    { name: "det", model: "test/golden/det.exposed.onnx", input: "x", dims: [1, 3, 256, 256] },
    { name: "rec", model: "test/golden/rec.exposed.onnx", input: "x", dims: [1, 3, 48, 320] },
  ];
  for (const c of cases) {
    const n = c.dims.reduce((a, b) => a * b, 1);
    const t = new ort.Tensor("float32", seededInput(n), c.dims);
    const m = await dump(c.model, { [c.input]: t });
    await Bun.write(`test/golden/${c.name}.golden.bin`, pack(m));
    await Bun.write(`test/golden/${c.name}.input.bin`, new Uint8Array(seededInput(n).buffer));
    const bytes = [...m.values()].reduce((a, d) => a + d.data.byteLength, 0);
    console.log(`${c.name}: ${m.size} tensors, ${(bytes / 1e6).toFixed(1)} MB, dims ${c.dims}`);
  }
}
