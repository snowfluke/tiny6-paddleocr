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

export type RecInit = { kind: "init"; rec: Uint8Array; wasm: Uint8Array; dict: string; calib?: string };
export type RecTask = { kind: "task"; id: number; width: number; height: number; data: Uint8Array };
export type RecDone = { id: number; text: string; confidence: number; ms: number };

let session: Session | null = null;
let dict: string[] = [];

declare const self: Worker;

self.onmessage = async (e: MessageEvent<RecInit | RecTask>) => {
  const msg = e.data;
  if (msg.kind === "init") {
    const arena = await loadKernels(msg.wasm);
    // The pool hands these bytes over a SharedArrayBuffer, which is what keeps
    // the main thread from cloning 4.46 MB once per worker. Anything that wants
    // a private copy has to make it here instead: the ONNX reader runs a
    // TextDecoder over the buffer and one rejects a shared view outright
    // ("The provided ArrayBufferView value must not be shared"), which threw
    // inside this async handler where nothing could see it. The copy is a few
    // milliseconds, and it happens in parallel across the workers rather than
    // in series on the thread that is trying to paint.
    const rec = typeof SharedArrayBuffer !== "undefined" && msg.rec.buffer instanceof SharedArrayBuffer
      ? msg.rec.slice()
      : msg.rec;
    session = new Session(parseOnnx(rec), arena, msg.calib ? { int8: JSON.parse(msg.calib) } : {});
    dict = parseDictionary(msg.dict);
    postMessage("ready");
    return;
  }
  const img: RGBA = { width: msg.width, height: msg.height, data: msg.data };
  const t = performance.now();
  const { text, confidence } = recognizeCrop(session!, img, dict);
  // The time inside the worker, so a caller can tell compute from queueing.
  postMessage({ id: msg.id, text, confidence, ms: performance.now() - t } satisfies RecDone);
};
