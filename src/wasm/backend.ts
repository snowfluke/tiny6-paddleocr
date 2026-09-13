import { CTRL_BYTES, defaultThreads, JOB, PARALLEL_MIN, Pool } from "./pool.ts";

// Linear-memory arena over the WASM kernels.
//
// Weights are uploaded once and kept (persistent, grows up from heap_base).
// Activations are scratch: every call takes a mark and rewinds to it on exit.

/** Shadow stack per worker. The kernels spill at most a few kilobytes. */
const WORKER_STACK_BYTES = 256 * 1024;

export type Kernels = {
  memory?: WebAssembly.Memory;
  heap_base(): number;
  gemm_range(
    m: number,
    k: number,
    n: number,
    ldb: number,
    ldc: number,
    a: number,
    b: number,
    c: number,
    bias: number,
    act: number,
    res: number,
    lo: number,
    hi: number,
  ): void;
  binary(
    op: number, mode: number, n: number, inner: number, channels: number,
    a: number, b: number, out: number,
  ): void;
  unary(op: number, n: number, a: number, out: number, p0: number, p1: number): void;
  reduce_mean(outer: number, inner: number, a: number, out: number): void;
  maxpool2x2(planes: number, h: number, w: number, a: number, out: number, lo: number, hi: number): void;
  affine_channels(
    n: number,
    inner: number,
    channels: number,
    a: number,
    s: number,
    t: number,
    out: number,
    cLo: number,
    cHi: number,
  ): void;
  im2col_strip(
    ih: number,
    iw: number,
    ow: number,
    kh: number,
    kw: number,
    sy: number,
    sx: number,
    pt: number,
    pl: number,
    dy: number,
    dx: number,
    p0: number,
    width: number,
    x: number,
    col: number,
    cLo: number,
    cHi: number,
  ): void;
  softmax_rows(cols: number, a: number, out: number, rLo: number, rHi: number): void;
  transpose4(
    d0: number,
    d1: number,
    d2: number,
    d3: number,
    s0: number,
    s1: number,
    s2: number,
    s3: number,
    a: number,
    out: number,
  ): void;
  scatter2x2(
    h: number,
    w: number,
    ky: number,
    kx: number,
    src: number,
    dst: number,
    pLo: number,
    pHi: number,
  ): void;
  resize_nearest(
    planes: number, h: number, w: number, sh: number, sw: number,
    a: number, out: number, lo: number, hi: number,
  ): void;
  gemm(
    m: number,
    k: number,
    n: number,
    ldb: number,
    ldc: number,
    a: number,
    b: number,
    c: number,
    bias: number,
    act: number,
    res: number,
  ): void;
  depthwise(
    channels: number,
    ih: number,
    iw: number,
    oh: number,
    ow: number,
    kh: number,
    kw: number,
    sy: number,
    sx: number,
    pt: number,
    pl: number,
    x: number,
    w: number,
    bias: number,
    y: number,
    act: number,
    chLo: number,
    chHi: number,
  ): void;
};

const PAGE = 65536;

export class Arena {
  readonly k: Kernels;
  private top: number;
  private readonly uploads = new WeakMap<Float32Array, number>();

  /**
   * Weights are persisted below this mark and activations above it, so a run
   * can reclaim every activation by rewinding to it. Sealing fixes the
   * boundary once all graphs have uploaded their weights.
   */
  private scratchBase = -1;
  private readonly free = new Map<number, number[]>();
  private readonly live = new Map<number, { len: number; refs: number }>();

  private pool: Pool | null = null;
  private readonly memory: WebAssembly.Memory;
  /** Fixed low allocation the worker protocol lives in; never recycled. */
  readonly ctrlPtr: number;
  constructor(exports: WebAssembly.Exports, memory?: WebAssembly.Memory) {
    this.k = exports as unknown as Kernels;
    this.memory = memory ?? (this.k.memory as WebAssembly.Memory);
    this.top = this.k.heap_base();
    this.ctrlPtr = this.alloc(CTRL_BYTES / 4);
  }

  get threads(): number {
    return this.pool?.count ?? 1;
  }

  /** Dispatches where a share ran over a millisecond late, and the time spent waiting for them. */
  get stalls(): { count: number; ms: number } {
    return { count: this.pool?.stalls ?? 0, ms: this.pool?.stallMs ?? 0 };
  }

  async startPool(bytes: Uint8Array, threads: number) {
    if (threads <= 1) return;
    // Every worker instantiates the same module on the same shared memory,
    // so without this they all place their shadow stack at the same address
    // and any kernel that spills a local races with the other three. It
    // stayed hidden until the 5x5 depthwise kernel spilled its 25 taps. Each
    // worker gets its own region here, below the seal so it is never reused.
    const stackBase = this.alloc(((threads - 1) * WORKER_STACK_BYTES) / 4);
    this.pool = await Pool.create(bytes, this.memory, this.k, this.ctrlPtr, threads, stackBase, WORKER_STACK_BYTES);
  }

  destroy() {
    this.pool?.destroy();
    this.pool = null;
  }

  // ---- parallel entry points ------------------------------------------
  // Each falls back to the single-threaded kernel when there is no pool or
  // the tensor is too small for the dispatch to pay for itself.

  /** `ldb` and `ldc` default to `n`; a strip passes its own B width and the full C width. */
  pGemm(
    m: number,
    k: number,
    n: number,
    a: number,
    b: number,
    c: number,
    bias: number,
    act: number,
    ldb = n,
    ldc = n,
    res = 0,
  ) {
    if (this.pool && m * n >= PARALLEL_MIN) {
      this.pool.dispatch(JOB.gemm, [m, k, n, ldb, ldc, a, b, c, bias, act, res]);
    } else {
      this.k.gemm(m, k, n, ldb, ldc, a, b, c, bias, act, res);
    }
  }

  /** im2col and GEMM for one strip as a single job; see JOB.convStrip. */
  pConvStrip(args: number[]) {
    const [cin, ih, iw, ow, kh, kw, sy, sx, pt, pl, dy, dx, p0, width, x, col, m, k, w, cbase, bias, act, ldc] = args;
    if (this.pool && m * width >= PARALLEL_MIN) {
      this.pool.dispatch(JOB.convStrip, args);
    } else {
      this.k.im2col_strip(ih, iw, ow, kh, kw, sy, sx, pt, pl, dy, dx, p0, width, x, col, 0, cin);
      this.k.gemm_range(m, k, width, width, ldc, w, col, cbase, bias, act, 0, 0, width);
    }
  }

  pDepthwise(args: number[]) {
    const channels = args[0];
    const outElems = channels * args[3] * args[4];
    if (this.pool && outElems >= PARALLEL_MIN) this.pool.dispatch(JOB.depthwise, args);
    else (this.k.depthwise as (...a: number[]) => void)(...args, 0, channels);
  }

  pUnary(op: number, n: number, a: number, out: number, p0: number, p1: number) {
    if (this.pool && n >= PARALLEL_MIN) {
      this.pool.dispatch(JOB.unary, [op, n, a, out], { 4: p0, 5: p1 });
    } else {
      this.k.unary(op, n, a, out, p0, p1);
    }
  }

  pBinarySame(op: number, n: number, a: number, b: number, out: number) {
    if (this.pool && n >= PARALLEL_MIN) this.pool.dispatch(JOB.binary, [op, 0, n, a, b, out]);
    else this.k.binary(op, 0, n, 0, 0, a, b, out);
  }

  pAffineChannels(n: number, inner: number, channels: number, a: number, s: number, t: number, out: number) {
    if (this.pool && n >= PARALLEL_MIN && channels > 1) {
      this.pool.dispatch(JOB.affine, [channels, n, inner, a, s, t, out]);
    } else {
      this.k.affine_channels(n, inner, channels, a, s, t, out, 0, channels);
    }
  }

  pSoftmaxRows(rows: number, cols: number, a: number, out: number) {
    if (this.pool && rows * cols >= PARALLEL_MIN && rows > 1) {
      this.pool.dispatch(JOB.softmax, [rows, cols, a, out]);
    } else {
      this.k.softmax_rows(cols, a, out, 0, rows);
    }
  }

  pScatter2x2(planes: number, h: number, w: number, ky: number, kx: number, src: number, dst: number) {
    if (this.pool && planes * h * w >= PARALLEL_MIN) {
      this.pool.dispatch(JOB.scatter2x2, [planes, h, w, ky, kx, src, dst]);
    } else {
      this.k.scatter2x2(h, w, ky, kx, src, dst, 0, planes);
    }
  }

  seal() {
    this.scratchBase = this.top;
  }

  /** Drop every activation from the previous run. */
  beginRun() {
    if (this.scratchBase < 0) this.seal();
    this.top = this.scratchBase;
    this.free.clear();
    this.live.clear();
  }

  /**
   * Exact-size free list. A static graph reuses the same handful of shapes,
   * so exact matching recycles nearly everything without a splitting policy.
   */
  allocScratch(len: number): number {
    const pool = this.free.get(len);
    const ptr = pool?.length ? pool.pop()! : this.alloc(len);
    this.live.set(ptr, { len, refs: 0 });
    return ptr;
  }

  retain(ptr: number) {
    const e = this.live.get(ptr);
    if (e) e.refs++;
  }

  release(ptr: number) {
    const e = this.live.get(ptr);
    if (!e || --e.refs > 0) return;
    this.live.delete(ptr);
    const pool = this.free.get(e.len);
    if (pool) pool.push(ptr);
    else this.free.set(e.len, [ptr]);
  }

  private get mem(): ArrayBuffer {
    return this.memory.buffer;
  }

  private ensure(end: number) {
    const have = this.mem.byteLength;
    if (end <= have) return;
    this.memory.grow(Math.ceil((end - have) / PAGE));
  }

  /** 16-byte aligned so every allocation is safe for v128 loads. */
  private alloc(floats: number): number {
    const ptr = (this.top + 15) & ~15;
    this.top = ptr + floats * 4;
    this.ensure(this.top);
    return ptr;
  }


  /** Cached by array identity, so graph weights upload exactly once. */
  persist(a: Float32Array): number {
    const hit = this.uploads.get(a);
    if (hit !== undefined) return hit;
    if (this.scratchBase >= 0) throw new Error("persist() after seal(): weights must upload first");
    const ptr = this.alloc(a.length);
    this.write(ptr, a);
    this.uploads.set(a, ptr);
    return ptr;
  }

  write(ptr: number, a: Float32Array) {
    new Float32Array(this.mem, ptr, a.length).set(a);
  }

  /**
   * Copy straight into the caller's array. A returned view would dangle after
   * the next memory.grow, and slicing first would copy the data twice.
   */
  /** Moves f32s inside the arena. Never crosses into JavaScript. */
  move(dst: number, src: number, n: number) {
    new Float32Array(this.mem).copyWithin(dst >> 2, src >> 2, (src >> 2) + n);
  }

  readInto(ptr: number, dst: Float32Array) {
    dst.set(new Float32Array(this.mem, ptr, dst.length));
  }

  zero(ptr: number, n: number) {
    new Float32Array(this.mem, ptr, n).fill(0);
  }
}

/**
 * Instantiating from bytes resolves to { instance, module } per the spec.
 * @types/bun declares the return as Instance, hence the cast.
 */
function instanceOf(result: unknown): WebAssembly.Instance {
  return (result as { instance: WebAssembly.Instance }).instance;
}

export async function loadKernels(bytes: Uint8Array): Promise<Arena> {
  const result = await WebAssembly.instantiate(bytes as BufferSource, {});
  return new Arena(instanceOf(result).exports);
}

/**
 * Loads the shared-memory build and attaches a worker pool. `bytes` must be
 * the module whose memory is imported (see tools/share-memory.ts).
 */
export async function loadKernelsThreaded(
  bytes: Uint8Array,
  threads = defaultThreads(),
  initialPages = 64,
): Promise<Arena> {
  const memory = new WebAssembly.Memory({ initial: initialPages, maximum: 32768, shared: true });
  const result = await WebAssembly.instantiate(bytes as BufferSource, { env: { memory } });
  const arena = new Arena(instanceOf(result).exports, memory);
  await arena.startPool(bytes, threads);
  return arena;
}
