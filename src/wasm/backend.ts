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
  qgemm(
    m: number, k: number, n: number, a: number, b: number, c: number,
    sw: number, bias: number, comp: number, act: number,
    res: number, rs: number, rzp: number, oinv: number, ozp: number, outI8: number,
    p0: number, p1: number, lo: number, hi: number,
  ): void;
  qdepthwise(
    c: number, ih: number, iw: number, oh: number, ow: number, kh: number, kw: number, sy: number, sx: number,
    pt: number, pl: number, x: number, xzp: number, w: number, sw: number, bias: number, act: number,
    out: number, oinv: number, ozp: number, outI8: number, lo: number, hi: number,
  ): void;
  qmean_channels(c: number, pixels: number, x: number, zp: number, scale: number, out: number): void;
  dot_probe(): number;
  qscale_channels(c: number, x: number, zp: number, scale: number, factor: number, out: number, oinv: number, ozp: number, lo: number, hi: number): void;
  quantize_nhwc(channels: number, cs: number, pixels: number, x: number, out: number, inv: number, zp: number, lo: number, hi: number): void;
  qim2col(
    ih: number, iw: number, ow: number, kh: number, kw: number, sy: number, sx: number, pt: number, pl: number,
    cs: number, x: number, zp: number, col: number, lo: number, hi: number,
  ): void;
  dequantize_nchw(channels: number, pixels: number, q: number, out: number, scale: number, zp: number, lo: number, hi: number): void;
  transpose_f32(rows: number, cols: number, a: number, out: number, lo: number, hi: number): void;
  qadd(c: number, a: number, azp: number, ascale: number, b: number, bzp: number, bscale: number, out: number, oinv: number, ozp: number, lo: number, hi: number): void;
  qresize2x(c: number, iw: number, x: number, out: number, lo: number, hi: number): void;
  qconcat_in(cin: number, cout: number, off: number, x: number, zx: number, sx: number, out: number, oinv: number, ozp: number, lo: number, hi: number): void;
  qmaxpool2x2same(c: number, h: number, w: number, x: number, out: number, lo: number, hi: number): void;
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

  /** Whether the engine's relaxed dot product is signed on both operands, which the int8 kernels need. */
  get signedDot(): boolean {
    return this.k.dot_probe() === -512;
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

  /** Rows of the int8 GEMM; see JOB.qgemm for the argument order. */
  pQGemm(args: number[], p0: number, p1: number) {
    const [m, , n] = args;
    if (this.pool && m * n >= PARALLEL_MIN) {
      this.pool.dispatch(JOB.qgemm, args, { 16: p0, 17: p1 });
    } else {
      (this.k.qgemm as (...a: number[]) => void)(...args, p0, p1, 0, m);
    }
  }

  // The int8 region's edges and its byte-wide passes, split by pixel.
  pQuantize(channels: number, cs: number, pixels: number, x: number, out: number, inv: number, zp: number) {
    if (this.pool && channels * pixels >= PARALLEL_MIN) this.pool.dispatch(JOB.quantize, [channels, cs, pixels, x, out, inv, zp]);
    else this.k.quantize_nhwc(channels, cs, pixels, x, out, inv, zp, 0, pixels);
  }

  pDequantize(channels: number, pixels: number, q: number, out: number, scale: number, zp: number) {
    if (this.pool && channels * pixels >= PARALLEL_MIN) this.pool.dispatch(JOB.dequantize, [channels, pixels, q, out, scale, zp]);
    else this.k.dequantize_nchw(channels, pixels, q, out, scale, zp, 0, pixels);
  }

  pQScale(c: number, x: number, zp: number, scale: number, factor: number, out: number, oinv: number, ozp: number, pixels: number) {
    if (this.pool && c * pixels >= PARALLEL_MIN) this.pool.dispatch(JOB.qscale, [c, x, zp, scale, factor, out, oinv, ozp, pixels]);
    else this.k.qscale_channels(c, x, zp, scale, factor, out, oinv, ozp, 0, pixels);
  }

  pQAdd(args: number[]) {
    const [c, , , , , , , , , , pixels] = args;
    if (this.pool && c * pixels >= PARALLEL_MIN) this.pool.dispatch(JOB.qadd, args);
    else (this.k.qadd as (...a: number[]) => void)(...args.slice(0, 10), 0, pixels);
  }

  pQResize2x(c: number, iw: number, x: number, out: number, oh: number) {
    if (this.pool && c * iw * oh >= PARALLEL_MIN) this.pool.dispatch(JOB.qresize2x, [c, iw, x, out, oh]);
    else this.k.qresize2x(c, iw, x, out, 0, oh);
  }

  pQConcat(args: number[]) {
    const [cin, , , , , , , , , pixels] = args;
    if (this.pool && cin * pixels >= PARALLEL_MIN) this.pool.dispatch(JOB.qconcat, args);
    else (this.k.qconcat_in as (...a: number[]) => void)(...args.slice(0, 9), 0, pixels);
  }

  pQMaxPool(c: number, h: number, w: number, x: number, out: number) {
    if (this.pool && c * h * w >= PARALLEL_MIN) this.pool.dispatch(JOB.qmaxpool, [c, h, w, x, out]);
    else this.k.qmaxpool2x2same(c, h, w, x, out, 0, h);
  }

  pTranspose(rows: number, cols: number, a: number, out: number) {
    if (this.pool && rows * cols >= PARALLEL_MIN) this.pool.dispatch(JOB.transpose, [rows, cols, a, out]);
    else this.k.transpose_f32(rows, cols, a, out, 0, rows);
  }

  /** im2col and int8 GEMM for a dense convolution, split by output pixel; args as in JOB.qconvDense. */
  pQConvDense(args: number[], p0: number, p1: number) {
    const [, , oh, ow, kh, kw, , , , , cs, x, zp, col, n, b, c, sw, bias, comp, act, oinv, ozp, outI8] = args;
    const m = oh * ow, k = kh * kw * cs;
    if (this.pool && m * n >= PARALLEL_MIN) {
      this.pool.dispatch(JOB.qconvDense, args, { 24: p0, 25: p1 });
    } else {
      this.k.qim2col(args[0], args[1], ow, kh, kw, args[6], args[7], args[8], args[9], cs, x, zp, col, 0, m);
      this.k.qgemm(m, k, n, col, b, c, sw, bias, comp, act, 0, 0, 0, oinv, ozp, outI8, p0, p1, 0, m);
    }
  }

  /** Output rows of the int8 depthwise convolution; args as in JOB.qdepthwise. */
  pQDepthwise(args: number[]) {
    const [c, , , oh, ow] = args;
    if (this.pool && c * oh * ow >= PARALLEL_MIN) this.pool.dispatch(JOB.qdepthwise, args);
    else (this.k.qdepthwise as (...a: number[]) => void)(...args, 0, oh);
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

  /** Byte-level twins of persist and write for the int8 path. */
  persistBytes(a: Uint8Array | Int8Array | Int32Array): number {
    if (this.scratchBase >= 0) throw new Error("persist() after seal(): weights must upload first");
    const ptr = this.alloc((a.byteLength + 3) >> 2);
    this.writeBytes(ptr, a);
    return ptr;
  }

  writeBytes(ptr: number, a: Uint8Array | Int8Array | Int32Array) {
    new Uint8Array(this.mem, ptr, a.byteLength).set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  }

  readBytesInto(ptr: number, a: Uint8Array | Int8Array) {
    a.set(new Uint8Array(this.mem, ptr, a.byteLength) as never);
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
