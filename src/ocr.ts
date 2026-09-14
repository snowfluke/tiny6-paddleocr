import { parseOnnx } from "./onnx/parse.ts";
import { Session } from "./runtime/graph.ts";
import { loadKernels, loadKernelsThreaded, type Arena } from "./wasm/backend.ts";
import { decodePng, type RGBA } from "./image/png.ts";
import { DEFAULT_DETECT, detect, type Box, type DetectOptions } from "./pipeline/detect.ts";
import { defaultThreads } from "./wasm/pool.ts";
import { cropForBox, parseDictionary, recognizeBox } from "./pipeline/recognize.ts";
import { RecPool, type WorkerFactory } from "./pipeline/rec-pool.ts";

export type OcrWord = { text: string; confidence: number; box: Box };

export type LineFilter = {
  /** Mean per-character CTC confidence a line must reach. Default 0.5. */
  minConfidence?: number;
  /** Drop lines made only of rule characters. Default true. */
  dropSeparators?: boolean;
};
export type OcrLine = { text: string; confidence: number; words: OcrWord[] };

export type Assets = {
  det: Uint8Array;
  rec: Uint8Array;
  dict: string;
  /** The plain kernels, used when threads resolve to 1. */
  wasm: Uint8Array;
  /** The shared-memory kernels. Without it the runtime stays single-threaded. */
  wasmShared?: Uint8Array;
  /** Defaults to half the cores, capped at four. Pass 1 to disable workers. */
  threads?: number;
  /**
   * Supplying this starts a task-parallel recognition pool: each worker owns
   * a rec Session and takes whole crops. Without it `recognizeAsync` falls
   * back to running crops in order on this thread.
   */
  makeRecWorker?: WorkerFactory;
  /**
   * Calibration JSON from tools/calibrate.ts. With it the convolutions run
   * on int8 where the engine supports the relaxed dot product.
   */
  detCalib?: string;
  recCalib?: string;
  /** Recognition workers. Defaults to the same count as `threads`. */
  recWorkers?: number;
};

/**
 * Lines below this mean per-character CTC confidence are dropped. Matches
 * ppu-paddle-ocr's default: the detector fires on logos, barcodes and
 * rules, and the recognizer reports those crops as low confidence.
 */
export const DEFAULT_MIN_CONFIDENCE = 0.5;

/**
 * A line of nothing but rule characters is a printed separator, not text.
 * Receipts and forms are full of them and the recogniser reads them
 * confidently, so a confidence threshold never catches them.
 */
export function isSeparator(text: string): boolean {
  const t = text.replace(/\s+/g, "");
  return t.length > 0 && !/[\p{L}\p{N}]/u.test(t) && /^[-_=+~*.:|/\\<>^'"`,;!?()[\]{}]+$/u.test(t);
}

/**
 * Six on an eight-core big.LITTLE machine, measured on the receipt: four
 * workers 93 ms, six 83 ms, eight 85 ms. Recognition is task-parallel with no
 * barrier, so an efficiency core can help here where it cannot in the
 * detection pool. Each worker holds its own copy of the weights, 4.3 MB.
 */
export function defaultRecWorkers(cores?: number): number {
  const n = cores ?? (typeof navigator !== "undefined" ? (navigator.hardwareConcurrency ?? 4) : 4);
  // Two workers measured slower than the calling thread (0.93x in Chrome):
  // not enough parallelism to cover the crops, still a message per crop.
  return n < 3 ? 0 : Math.min(6, n);
}

export class Ocr {
  private constructor(
    private readonly det: Session,
    private readonly rec: Session,
    private readonly dict: string[],
    readonly arena: Arena,
    private readonly recPool: RecPool | null,
  ) {}

  get recWorkers(): number {
    return this.recPool?.size ?? 0;
  }

  /** Whether the convolutions run on int8: calibrated, and the engine's dot product is signed. */
  get int8(): boolean {
    return this.det.int8Nodes > 0 || this.rec.int8Nodes > 0;
  }

  /** Exposed for callers driving recognition directly, such as batching. */
  get recSession(): Session {
    return this.rec;
  }

  get dictionary(): string[] {
    return this.dict;
  }

  static async create(a: Assets): Promise<Ocr> {
    const threads = a.threads ?? (a.wasmShared ? defaultThreads() : 1);
    const threaded = threads > 1 && !!a.wasmShared;
    // The pool starts after the Sessions, not before them: the constructor
    // uploads every weight into the arena, each upload can grow the shared
    // memory, and a pool that is already spinning makes every one of those
    // grows several times more expensive. Measured on this machine, 8 threads
    // against 1, the two Sessions cost 155.9 ms with the pool first and
    // 50.2 ms with it deferred - see backend.loadKernelsThreaded.
    const arena = threaded
      ? await loadKernelsThreaded(a.wasmShared!, threads, 64, false)
      : await loadKernels(a.wasm);

    // The recognition pool is started here, before the Sessions, so its work
    // overlaps them instead of following them. It shares nothing with the
    // constructor but arena.dotMode, and spawning the workers, cloning 4.4 MB
    // of weights into each and having each one parse the model is the single
    // largest cost in a browser start: measured in Chrome on this page, six
    // workers take 8 ms to spawn, 27 ms of main-thread clone and 38 ms to
    // initialise, against 55 ms for both Sessions. In series that is 155 ms
    // end to end; overlapped it is the longer of the two plus the clone.
    //
    // The synchronous part of RecPool.create - the spawn and every postMessage
    // - still runs before the constructor does, since JS is single threaded up
    // to its first await. That is the 35 ms that cannot be hidden, and it is
    // cheaper than the 38 ms of initialisation it buys.
    const recCount = a.recWorkers ?? defaultRecWorkers();
    const recPoolPromise = a.makeRecWorker && recCount > 1
      ? RecPool.create(a.makeRecWorker, { rec: a.rec, wasm: a.wasm, dict: a.dict, calib: arena.dotMode !== "none" ? a.recCalib : undefined }, recCount)
      : null;
    // The await below is separated from this by another await, so a rejection
    // arriving in between would be reported as unhandled before anyone is
    // listening. This keeps the original promise awaitable and still rejects.
    recPoolPromise?.catch(() => {});

    // An engine whose dot product lowering the probe does not recognise
    // stays on fp32 rather than being silently wrong.
    const int8 = (json?: string) => (json && arena.dotMode !== "none" ? { int8: JSON.parse(json) } : {});
    const det = new Session(parseOnnx(a.det), arena, int8(a.detCalib));
    const rec = new Session(parseOnnx(a.rec), arena, int8(a.recCalib));

    if (threaded) await arena.startPool(a.wasmShared!, threads);
    const recPool = recPoolPromise ? await recPoolPromise : null;

    return new Ocr(det, rec, parseDictionary(a.dict), arena, recPool);
  }

  /** Stops both worker pools. The instance is unusable afterwards. */
  destroy() {
    this.arena.destroy();
    this.recPool?.destroy();
  }

  /** Exposed so callers can spread it and override one field. */
  readonly detectDefaults = DEFAULT_DETECT;

  detect(img: RGBA, opts: DetectOptions = DEFAULT_DETECT) {
    return detect(this.det, img, opts);
  }

  /** Boxes sharing a row are joined into one line, as the per-line strategy does. */
  recognize(img: RGBA, opts: DetectOptions = DEFAULT_DETECT, filter: LineFilter = {}): OcrLine[] {
    return this.recognizeBoxes(img, this.detect(img, opts).boxes, filter);
  }

  /** Recognition only, for callers that already ran detection. */
  recognizeBoxes(img: RGBA, boxes: Box[], filter: LineFilter = {}): OcrLine[] {
    const { minConfidence = DEFAULT_MIN_CONFIDENCE, dropSeparators = true } = filter;
    const lines: OcrLine[] = [];
    for (const row of readingOrder(boxes)) {
      const words: OcrWord[] = [];
      for (const box of row) {
        const { text, confidence } = recognizeBox(this.rec, img, box, this.dict);
        if (text.trim()) words.push({ text, confidence, box });
      }
      if (!words.length) continue;
      const confidence = words.reduce((a, w) => a + w.confidence, 0) / words.length;
      if (confidence < minConfidence) continue;
      const text = words.map((w) => w.text).join(" ");
      if (dropSeparators && isSeparator(text)) continue;
      lines.push({ text, confidence, words });
    }
    return lines;
  }

  /** Same as `recognize`, but spreads the crops across the recognition pool. */
  async recognizeAsync(
    img: RGBA,
    opts: DetectOptions = DEFAULT_DETECT,
    filter: LineFilter = {},
  ): Promise<OcrLine[]> {
    return this.recognizeBoxesAsync(img, this.detect(img, opts).boxes, filter);
  }

  async recognizeBoxesAsync(
    img: RGBA,
    boxes: Box[],
    filter: LineFilter = {},
  ): Promise<OcrLine[]> {
    // Below the crossover the shared arena wins: while a pool worker runs a
    // whole crop on one thread, the synchronous path gives each crop all of
    // the arena's. Measured in Chrome on the receipt, synchronous against
    // six workers: 3 crops 19.0 ms vs 22.4, 4 crops 25.3 vs 20.7. Single
    // threaded, one worker already beats the calling thread, so the bar is
    // lower there.
    if (!this.recPool || boxes.length < (this.arena.threads > 1 ? 4 : 2)) {
      return this.recognizeBoxes(img, boxes, filter);
    }
    const { minConfidence = DEFAULT_MIN_CONFIDENCE, dropSeparators = true } = filter;

    const rows = readingOrder(boxes);
    // Widest first. Each idle worker takes the next crop, so a wide one that
    // starts late runs alone at the end while the other workers sit idle; a
    // 424 px crop costs 25 ms against 5 for a narrow one. Cropping stays
    // here: it is cheap next to the graph, and the pixels cross to the worker
    // either way.
    const flat = rows.flat().sort((a, b) => b.width / b.height - a.width / a.height);
    const decoded = await this.recPool.run(flat.map((b) => cropForBox(img, b)));

    const byBox = new Map(flat.map((b, i) => [b, decoded[i]]));
    const lines: OcrLine[] = [];
    for (const row of rows) {
      const words: OcrWord[] = [];
      for (const box of row) {
        const d = byBox.get(box)!;
        if (d.text.trim()) words.push({ text: d.text, confidence: d.confidence, box });
      }
      if (!words.length) continue;
      const confidence = words.reduce((a, w) => a + w.confidence, 0) / words.length;
      if (confidence < minConfidence) continue;
      const text = words.map((w) => w.text).join(" ");
      if (dropSeparators && isSeparator(text)) continue;
      lines.push({ text, confidence, words });
    }
    return lines;
  }

  text(img: RGBA, opts: DetectOptions = DEFAULT_DETECT, filter: LineFilter = {}): string {
    return this.recognize(img, opts, filter).map((l) => l.text).join("\n");
  }
}

/** Group boxes into rows by vertical overlap, then read each row left to right. */
export function readingOrder(boxes: Box[]): Box[][] {
  const sorted = [...boxes].sort((a, b) => a.y - b.y);
  const rows: Box[][] = [];
  for (const b of sorted) {
    const row = rows.find((r) => {
      const ref = r[0];
      const overlap = Math.min(ref.y + ref.height, b.y + b.height) - Math.max(ref.y, b.y);
      return overlap > Math.min(ref.height, b.height) * 0.5;
    });
    if (row) row.push(b);
    else rows.push([b]);
  }
  for (const r of rows) r.sort((a, b) => a.x - b.x);
  return rows;
}

export { decodePng };
