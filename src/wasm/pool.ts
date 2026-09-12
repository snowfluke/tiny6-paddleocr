// A fixed pool of workers sharing one WASM linear memory.
//
// Every parallel kernel splits an index range, and each share writes to a
// disjoint slice of the output, so the workers need no locks: a job is one
// sequence-number bump plus a barrier. The main thread takes share 0 and
// spins for the rest rather than calling Atomics.wait, which browsers forbid
// on the main thread.

export const JOB = {
  gemm: 1,
  depthwise: 2,
  unary: 4,
  binary: 5,
  scatter2x2: 6,
  softmax: 7,
  affine: 8,
  im2colStrip: 9,
} as const;

/** Int32 slots in the control block: 0 sequence, 1 completions, 2 op, 3+ args. */
const CTRL_SLOTS = 32;
export const CTRL_BYTES = CTRL_SLOTS * 4;
const SEQ = 0;
const DONE = 1;
const OP = 2;
const ARG0 = 3;

/**
 * Runs one share of the current job. Shared verbatim by the workers and the
 * main thread, so both compute their slice the same way.
 */
const RUN_SHARE = `
function share(total, index, count) {
  const per = Math.ceil(total / count);
  const lo = Math.min(total, index * per);
  return [lo, Math.min(total, lo + per)];
}
// The GEMM's micro-kernel is eight columns wide, so a share whose width is
// not a multiple of eight drops its remainder into the edge path, which has
// one accumulator instead of sixteen and runs 6-8x slower per flop. Rounding
// the share up gives 64,64,64,48 where the even split gave 60,60,60,60: more
// work on three of the threads and no edge column anywhere but the last.
//
// Only the column split may round. A job sharded over channels must not, or
// 24 channels across four threads becomes 8,8,8,0.
function shareBy8(total, index, count) {
  const per = (Math.ceil(total / count) + 7) & ~7;
  const lo = Math.min(total, index * per);
  return [lo, Math.min(total, lo + per)];
}
function runShare(k, c, index, count) {
  const op = c[2];
  const a = 3;
  if (op === ${JOB.gemm}) {
    const [lo, hi] = shareBy8(c[a + 2], index, count);
    if (lo < hi) k.gemm_range(c[a], c[a+1], c[a+2], c[a+3], c[a+4], c[a+5], c[a+6], c[a+7], c[a+8], c[a+9], lo, hi);
  } else if (op === ${JOB.depthwise}) {
    const [lo, hi] = share(c[a], index, count);
    if (lo < hi) k.depthwise(c[a], c[a+1], c[a+2], c[a+3], c[a+4], c[a+5], c[a+6], c[a+7], c[a+8], c[a+9], c[a+10], c[a+11], c[a+12], c[a+13], c[a+14], c[a+15], lo, hi);
  } else if (op === ${JOB.im2colStrip}) {
    const [lo, hi] = share(c[a], index, count);
    if (lo < hi) k.im2col_strip(c[a+1], c[a+2], c[a+3], c[a+4], c[a+5], c[a+6], c[a+7], c[a+8], c[a+9], c[a+10], c[a+11], c[a+12], c[a+13], c[a+14], c[a+15], lo, hi);
    } else if (op === ${JOB.unary}) {
    const [lo, hi] = share(c[a + 1], index, count);
    if (lo < hi) k.unary(c[a], hi - lo, c[a + 2] + lo * 4, c[a + 3] + lo * 4, f[a + 4], f[a + 5]);
  } else if (op === ${JOB.binary}) {
    const [lo, hi] = share(c[a + 2], index, count);
    if (lo < hi) k.binary(c[a], 0, hi - lo, 0, 0, c[a + 3] + lo * 4, c[a + 4] + lo * 4, c[a + 5] + lo * 4);
  } else if (op === ${JOB.affine}) {
    const [lo, hi] = share(c[a], index, count);
    if (lo < hi) k.affine_channels(c[a + 1], c[a + 2], c[a], c[a + 3], c[a + 4], c[a + 5], c[a + 6], lo, hi);
  } else if (op === ${JOB.softmax}) {
    const [lo, hi] = share(c[a], index, count);
    if (lo < hi) k.softmax_rows(c[a + 1], c[a + 2], c[a + 3], lo, hi);
  } else if (op === ${JOB.scatter2x2}) {
    const [lo, hi] = share(c[a], index, count);
    if (lo < hi) k.scatter2x2(c[a + 1], c[a + 2], c[a + 3], c[a + 4], c[a + 5], c[a + 6], lo, hi);
  }
}
`;

const WORKER_SRC = `
let k = null, c = null, f = null, index = 0, count = 1;
${RUN_SHARE}
onmessage = async (e) => {
  const d = e.data;
  const { instance } = await WebAssembly.instantiate(d.bytes, { env: { memory: d.memory } });
  k = instance.exports;
  c = new Int32Array(d.memory.buffer, d.ctrl, ${CTRL_SLOTS});
  f = new Float32Array(d.memory.buffer, d.ctrl, ${CTRL_SLOTS});
  index = d.index;
  count = d.count;
  // Snapshot the sequence BEFORE reporting ready. Reading it after lets a job
  // dispatched in that window set seen to its own value, so the wait below
  // blocks on a change that already happened and never wakes.
  let seen = Atomics.load(c, ${SEQ});
  postMessage("ready");
  for (;;) {
    Atomics.wait(c, ${SEQ}, seen);
    const seq = Atomics.load(c, ${SEQ});
    if (seq < 0) return;
    seen = seq;
    runShare(k, c, index, count);
    Atomics.add(c, ${DONE}, 1);
  }
};
`;

type ShareFn = (k: unknown, c: Int32Array, index: number, count: number) => void;

export class Pool {
  private readonly workers: Worker[] = [];
  private readonly ctrl: Int32Array;
  private readonly ctrlF: Float32Array;
  private readonly runShare: ShareFn;
  private url: string | null = null;

  /** `count` includes the calling thread, which always takes share 0. */
  private constructor(
    readonly count: number,
    private readonly k: unknown,
    memory: WebAssembly.Memory,
    ctrlPtr: number,
  ) {
    this.ctrl = new Int32Array(memory.buffer, ctrlPtr, CTRL_SLOTS);
    this.ctrlF = new Float32Array(memory.buffer, ctrlPtr, CTRL_SLOTS);
    this.runShare = new Function("f", `${RUN_SHARE}; return runShare;`)(this.ctrlF) as ShareFn;
  }

  static async create(
    bytes: Uint8Array,
    memory: WebAssembly.Memory,
    kernels: unknown,
    ctrlPtr: number,
    threads: number,
  ): Promise<Pool> {
    const pool = new Pool(threads, kernels, memory, ctrlPtr);
    if (threads <= 1) return pool;

    const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
    pool.url = url;
    await Promise.all(
      Array.from({ length: threads - 1 }, (_, i) => {
        const w = new Worker(url, { type: "module" });
        pool.workers.push(w);
        return new Promise<void>((resolve, reject) => {
          w.onmessage = () => resolve();
          w.onerror = (e) => reject(new Error(`worker failed: ${(e as ErrorEvent).message ?? e}`));
          w.postMessage({ bytes, memory, ctrl: ctrlPtr, index: i + 1, count: threads });
        });
      }),
    );
    return pool;
  }

  /** `args` are int32 slots; `floats` overlay the same slots as f32. */
  dispatch(op: number, args: number[], floats?: Record<number, number>) {
    if (this.count <= 1) {
      this.ctrl[OP] = op;
      for (let i = 0; i < args.length; i++) this.ctrl[ARG0 + i] = args[i];
      if (floats) for (const [k, v] of Object.entries(floats)) this.ctrlF[ARG0 + Number(k)] = v;
      this.runShare(this.k, this.ctrl, 0, 1);
      return;
    }

    this.ctrl[OP] = op;
    for (let i = 0; i < args.length; i++) this.ctrl[ARG0 + i] = args[i];
    if (floats) for (const [k, v] of Object.entries(floats)) this.ctrlF[ARG0 + Number(k)] = v;

    Atomics.store(this.ctrl, DONE, 0);
    Atomics.add(this.ctrl, SEQ, 1);
    Atomics.notify(this.ctrl, SEQ);

    this.runShare(this.k, this.ctrl, 0, this.count);

    // Spin rather than Atomics.wait, which a browser main thread may not call.
    // Shares finish within a millisecond of each other; the deadline only
    // exists so a lost worker surfaces as an error instead of a frozen tab.
    const want = this.count - 1;
    const deadline = Date.now() + 10_000;
    while (Atomics.load(this.ctrl, DONE) < want) {
      if (Date.now() > deadline) {
        throw new Error(`worker pool stalled: ${Atomics.load(this.ctrl, DONE)}/${want} shares done`);
      }
    }
  }

  destroy() {
    if (!this.workers.length) return;
    Atomics.store(this.ctrl, SEQ, -1);
    Atomics.notify(this.ctrl, SEQ);
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
    if (this.url) URL.revokeObjectURL(this.url);
  }
}

/**
 * Below this many output elements a job is not worth splitting. Measured on
 * the detection graph at 960x960: 4096 beat 16k, 32k and 128k, because the
 * later layers are small and shrinking the threshold keeps them parallel.
 */
export const PARALLEL_MIN = 4096;

/**
 * Half the reported cores, capped at four.
 *
 * Every job ends at a barrier, so the slowest share sets the pace. On a
 * big.LITTLE machine (any Apple silicon, most recent phones) a share landing
 * on an efficiency core holds up the rest: measured on the detection graph at
 * 960x960, four threads ran 164 ms and eight ran 180 ms. Half the core count
 * keeps the work on the fast cores.
 */
export function defaultThreads(): number {
  const n = typeof navigator !== "undefined" ? (navigator.hardwareConcurrency ?? 4) : 4;
  return Math.max(1, Math.min(4, Math.floor(n / 2)));
}
