// One recognition worker: owns a complete rec Session and processes whole
// crops. This is task parallelism, not the data parallelism in wasm/pool.ts.
//
// A crop is small enough that splitting one graph across threads barely pays
// (measured 1.07x at the median crop width) while still dispatching 219 jobs.
// Handing a worker the entire crop removes the dispatch entirely.

import { parseOnnx } from "../onnx/parse.ts";
import { Session } from "../runtime/graph.ts";
import { loadKernels } from "../wasm/backend.ts";
import { parseDictionary, recognizeCrop } from "../pipeline/recognize.ts";
import type { RGBA } from "../image/png.ts";

export type RecInit = { kind: "init"; rec: Uint8Array; wasm: Uint8Array; dict: string };
export type RecTask = { kind: "task"; id: number; width: number; height: number; data: Uint8Array };
export type RecDone = { id: number; text: string; confidence: number };

let session: Session | null = null;
let dict: string[] = [];

declare const self: Worker;

self.onmessage = async (e: MessageEvent<RecInit | RecTask>) => {
  const msg = e.data;
  if (msg.kind === "init") {
    const arena = await loadKernels(msg.wasm);
    session = new Session(parseOnnx(msg.rec), arena);
    dict = parseDictionary(msg.dict);
    postMessage("ready");
    return;
  }
  const img: RGBA = { width: msg.width, height: msg.height, data: msg.data };
  const { text, confidence } = recognizeCrop(session!, img, dict);
  postMessage({ id: msg.id, text, confidence } satisfies RecDone);
};
