// Task-parallel recognition: each worker owns a complete rec Session and is
// handed whole crops.
//
// The data-parallel pool in wasm/pool.ts splits one kernel across threads,
// which suits detection's large tensors. A recognition crop is small - 48
// pixels tall, usually under 200 wide - so splitting its kernels barely pays
// (1.07x at the median width) while the graph still dispatches 219 jobs.
// Whole crops per worker dispatch nothing.
//
// Each worker keeps its own copy of the weights, ~4.3 MB, because they run
// on separate non-shared memories. That is the price of the simpler design.

import type { RGBA } from "../image/png.ts";
import type { Decoded } from "./recognize.ts";

export type RecPoolAssets = { rec: Uint8Array; wasm: Uint8Array; dict: string; calib?: string };

/** Supplied by the entry point: Node points at the file, the browser at a blob. */
export type WorkerFactory = () => Worker;

type Pending = { resolve: (d: Decoded) => void; reject: (e: Error) => void };

/**
 * Backs the bytes with a SharedArrayBuffer so a postMessage transfers a view
 * instead of a copy. Returns them unchanged where SharedArrayBuffer is absent
 * - a page without cross-origin isolation - or for a buffer that already is
 * one, which is what a second pool on the same weights would hand it.
 */
function shareable(bytes: Uint8Array): Uint8Array {
  if (typeof SharedArrayBuffer === "undefined" || bytes.buffer instanceof SharedArrayBuffer) return bytes;
  const sab = new SharedArrayBuffer(bytes.length);
  new Uint8Array(sab).set(bytes);
  return new Uint8Array(sab);
}

export class RecPool {
  private readonly idle: Worker[] = [];
  private readonly queue: { crop: RGBA; pending: Pending }[] = [];
  private readonly inflight = new Map<Worker, Pending>();
  private nextId = 1;

  private constructor(private readonly workers: Worker[]) {
    this.idle.push(...workers);
  }

  get size(): number {
    return this.workers.length;
  }

  static async create(
    make: WorkerFactory,
    assets: RecPoolAssets,
    count: number,
  ): Promise<RecPool> {
    // One shared copy rather than one clone per worker. A plain Uint8Array in
    // a postMessage is structurally cloned into every worker: the 4.46 MB of
    // rec weights cost 20.7-21.8 ms of main-thread time to hand to six of them,
    // and a view onto a SharedArrayBuffer costs 0, because the bytes are never
    // copied at all. The workers only read them.
    const rec = shareable(assets.rec);
    const workers = await Promise.all(
      Array.from({ length: count }, () =>
        new Promise<Worker>((resolve, reject) => {
          const w = make();
          w.onmessage = () => resolve(w);
          w.onerror = (e) => reject(new Error(`rec worker failed: ${(e as ErrorEvent).message ?? e}`));
          w.postMessage({ kind: "init", rec, wasm: assets.wasm, dict: assets.dict, calib: assets.calib });
        })),
    );
    const pool = new RecPool(workers);
    for (const w of workers) {
      w.onmessage = (e: MessageEvent) => pool.finish(w, e.data as Decoded);
      w.onerror = (e) => pool.fail(w, new Error((e as ErrorEvent).message ?? "rec worker error"));
    }
    return pool;
  }

  /** Results come back index-aligned with `crops`, whatever order they finish. */
  async run(crops: RGBA[]): Promise<Decoded[]> {
    return Promise.all(crops.map((crop) =>
      new Promise<Decoded>((resolve, reject) => {
        this.queue.push({ crop, pending: { resolve, reject } });
        this.pump();
      })
    ));
  }

  private pump() {
    while (this.idle.length && this.queue.length) {
      const w = this.idle.pop()!;
      const job = this.queue.shift()!;
      this.inflight.set(w, job.pending);
      const { width, height, data } = job.crop;
      w.postMessage({ kind: "task", id: this.nextId++, width, height, data }, [data.buffer]);
    }
  }

  private finish(w: Worker, result: Decoded) {
    const pending = this.inflight.get(w);
    this.inflight.delete(w);
    this.idle.push(w);
    pending?.resolve(result);
    this.pump();
  }

  private fail(w: Worker, err: Error) {
    const pending = this.inflight.get(w);
    this.inflight.delete(w);
    this.idle.push(w);
    pending?.reject(err);
    this.pump();
  }

  destroy() {
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
    this.idle.length = 0;
  }
}
