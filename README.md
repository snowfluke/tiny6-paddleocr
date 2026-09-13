# tiny6-paddleocr

PP-OCRv6 tiny with **zero runtime dependencies**. A hand-written ONNX
interpreter plus WASM SIMD kernels replace onnxruntime and OpenCV.

Proof of concept. Runs on Node/Bun and in the browser.

## Size

| part | size |
|---|---|
| runtime, browser bundle | **126 KB** (48 KB gzipped) |
| of which WASM kernels | 28,589 bytes, twice (plain + shared memory) |
| detection model | 1.70 MB |
| recognition model | 4.26 MB |
| dictionary (6,906 entries) | 0.03 MB |
| **total** | **6.12 MB** on the wire, 6.18 MB unpacked |

Peak wasm memory for detection at 960x960 is 168 MB. It was 422 MB before the
im2col matrix was built a strip at a time; a heap that size fails on a phone.

For comparison, `onnxruntime-web`'s smallest WASM binary is 14 MB and
`onnxruntime-node` unpacks to 85 MB per platform. OpenCV.js adds 15 MB.

## Speed

darwin arm64. ORT uses every core; the "4 threads" column is ours with the
worker pool on. Each figure is measured in its own process, because both
engines hold cores and whichever ran second would look slow.

There are two different comparisons and they give very different answers.
onnxruntime-web is WebAssembly like us and hits the same engine ceiling;
onnxruntime-node is native ARM64 and does not.

| model | input | ours 1t | ours 4t | ort-web 1t | ort-web 4t | ort native |
|---|---|---|---|---|---|---|
| det | 1x3x960x960 | 272 ms | **95 ms** | 246 ms | 77 ms | 55 ms |
| rec | 1x3x48x320 | 13.3 ms | **6.6 ms** | 11.8 ms | 3.6 ms | 3 ms |

Against onnxruntime-web the gap is 1.1x on one thread and 1.2x on four for
detection. ort-web scales 3.2x on four threads where we get 2.9x; what is
left is not the split (see Threads), it is that every thread here slows by
about a quarter once all cores are loaded, and ort-web's kernels move fewer
bytes per flop.

End to end on a 720x1280 receipt, both warm, minimum of six, measured in the
same minute:

| | idle machine | loaded machine |
|---|---|---|
| ppu-paddle-ocr (onnxruntime-node + OpenCV) | **110** | 151 |
| tiny6-paddleocr (this) | 148 (detect 66 + recognize 80) | **149** |

Under load it is a tie. Idle, the native reference wins by 1.35x, and on
fp32 the arithmetic says it keeps winning: native ORT does 18.7 us per
recognition column against our 42, and its detection graph runs in 29 ms
against our 56. The fp32 ceiling is the engine (see below), not the kernel.

So the convolutions run on int8. The relaxed dot product
(`i32x4.relaxed_dot_i8x16_i7x16_add_s`) does sixteen multiply-adds in one
instruction where an FMA does four, and the engine lowers it to one ARM
`sdot`. Measured on the same receipt, same minute, warm minimum of six, on
a machine with about a quarter of its cores busy:

| pipeline, 720x1280 receipt | ms | vs native |
|---|---|---|
| ppu-paddle-ocr (onnxruntime-node, native ARM64) | 121 | 1.00x |
| tiny6-paddleocr fp32 | 142 | 0.85x |
| **tiny6-paddleocr int8** | **86** (detect 41 + recognize 48) | **1.41x** |

Single-threaded per model, int8 against fp32: recognition 14 -> 8 ms,
detection at 960x544 171 -> 120 ms. The GEMM alone measures 3.1-5.5x on
the recognition shapes (`i32` accumulate, per-channel `fp32` epilogue).
What is left in detection is the fp32 tail (transposed convolutions, the
head) and the byte-wide passes between convolutions, which are memory-bound.

Absolute numbers on a developer machine are not worth much: the same build
measured 116 ms and 193 ms for the same work depending on what else was
running. `bun run ab <rev> <model> <dims> <threads>` exports that revision,
builds it, and alternates which side runs first each round, reporting the
minimum of each. Every ratio quoted here comes from that.

Detection uses every core, capped at eight. It used to be four: with static
shares every job ended at a barrier, so a share landing on an efficiency core
held up the rest and eight threads measured slower than four. Work is claimed
in blocks now and a slow core just takes fewer of them; interleaved at
960x960, four threads 95.7 ms and eight 88.0, and the receipt pipeline
154 -> 145. Recognition workers default to six: on the receipt four workers
recognise in 93 ms, six in 83, eight in 85.

Brave reports a random core count per site (a fingerprinting defence), which
would leave the pool on a fraction of the machine. The cores themselves are
not hidden, so on Brave `createOcr` measures them: the same loop on 1, 2, 4
and 8 workers at once, summing single-time over each-time as core
equivalents, stopping when more workers stop adding throughput. About 30 ms
once per origin, cached (`src/cores.ts`). This M1 with four performance and
four efficiency cores measures 8.

Detection at 960x960, as the work landed:

| | ms |
|---|---|
| first working graph, all TypeScript | 3391 |
| convolutions in WASM | 1170 |
| activations resident in WASM memory | 445 |
| worker pool | 214 |
| 8x8 GEMM micro-kernel | 144 |
| gelu fused, concat and transposed conv kept in wasm | 118 |
| im2col in strips, one job per strip, depthwise specialised | 96 |
| residual add in the gemm epilogue, work claimed in blocks | **95** (loaded machine) |

Recognition on the receipt, 28 crops, wall:

| | ms |
|---|---|
| four workers, reading order | 101 |
| eleven ops no longer falling back to JavaScript | 100 |
| crop widths padded to eight, widest first | 93 |
| six workers | 83 |
| depthwise specialised | **80** |

A crop costs about 1.0 ms plus 42 us per pixel of width on one thread. The
28 crops sum to ~320 ms of compute, and six workers finish in 80 with every
worker busy within 2-6 ms of the wall. There is nothing left in scheduling.

## Accuracy

Checked against `ppu-paddle-ocr` (onnxruntime + OpenCV) on the same file.

- Every intermediate tensor of both models matches ORT to 2e-3, on the
  TypeScript path and the WASM path. 242 tensors for detection, 219 for
  recognition. Every node of a threaded run is bit-identical to the
  single-threaded one, checked over six runs because a race only shows on
  some schedules, and the recognition pool returns exactly what the serial
  path does.
- Receipt: 13 of 18 lines byte-identical with filtering off. Two of the five
  that differ are barcode noise both implementations read as garbage; three
  differ by one character because the crops differ by a pixel or two.
- The same receipt as a progressive JPEG, decoded here, yields identical text
  to the PNG.
- int8, on 60 SROIE receipts (33,761 ground-truth characters) with the
  calibration taken from 16 other receipts, scored by order-independent
  token F1 because the SROIE transcripts are uppercased and split
  inconsistently: fp32 79.13%, int8 81.00% (20 receipts worse, 39 better).
  The difference is inside the run-to-run band, so read it as no loss, not
  as a gain. Per-tensor activation scales lost about a point; per-channel
  scales, which fold into the next layer's weights for free, do not. A
  fake-quantization harness (`tools/fakequant.ts`, `tools/sroie.ts`)
  predicted 81.01% before the kernels existed, and the kernels are pinned
  to an integer reference bit for bit (`test/qgemm.test.ts`).

## Run it

```sh
bun install            # onnxruntime-node, dev only, for goldens and benchmarks
bun run build          # kernels.wasm, kernels.shared.wasm (committed), demo/tiny6.js
bun run build:web      # the bundle only; no Rust needed
bun test               # goldens, geometry, threading, JPEG, end-to-end OCR
bun run typecheck
bun run ocr test/images/receipt.jpg
bun run ocr test/images/tilted.png --rotated
bun run bench
bun run bench:gemm
bun run ab HEAD models/det.onnx 1x3x960x960 4   # interleaved A/B vs a revision
bun run demo           # copies models into demo/, serves http://localhost:8099
```

`bun test` regenerates the ORT goldens on first run if they are absent
(~1 minute, ~100 MB, gitignored) and builds the kernels if needed.

```ts
import { decodeImage, makeRecWorker, Ocr } from "tiny6-paddleocr";

const ocr = await Ocr.create({ det, rec, dict, wasm, wasmShared, makeRecWorker });
const lines = await ocr.recognizeAsync(await decodeImage(bytes));
ocr.destroy();
```

## How it works

```
PNG/JPEG -> RGBA -> resize to 32-grid -> normalize -> det graph
    -> probability map -> threshold -> connected components -> boxes -> merge
    -> crops -> 48px tall, /127.5-1 -> rec graph -> CTC greedy decode -> text
```

- `src/onnx/` parses the ONNX protobuf directly. No generated code.
- `src/runtime/fuse.ts` rewrites the graph before it runs. paddle2onnx exports
  gelu as five nodes and gives the recognition model no convolution bias at
  all, emitting a separate Add for each one. Folding those, plus Relu, takes
  detection from 242 nodes to 171 and recognition from 219 to 146. These are
  the same rewrites onnxruntime calls Gelu Fusion, Conv Add Fusion and Conv
  Activation Fusion.
- `src/runtime/graph.ts` executes the 22 ops the two models use. It has two
  paths: pure TypeScript, which stays as the reference, and a resident path
  where every activation is a pointer into WASM memory and nothing crosses the
  boundary until the graph output. Buffers are refcounted and recycled the
  moment their last reader has run.
- `src/wasm/kernels.rs` is the only native code: GEMM, depthwise conv, im2col,
  elementwise, reductions, pooling, resize. Both models are MobileNet-shaped,
  so 86 of 122 convolutions are 1x1 and need no im2col. `exp` and `erf` are
  implemented there too; `no_std` has no libm.
- `src/image/` decodes PNG with `DecompressionStream` and JPEG from scratch
  (baseline and progressive). The browser entry uses `createImageBitmap`.
- `src/pipeline/components.ts` replaces `cv.findContours` with union-find
  connected components; `boxes.ts` replaces `cv.minAreaRect` with a convex
  hull and rotating calipers.

### Falling back to JavaScript is the expensive mistake

An op with no resident kernel downloads its inputs out of wasm memory, runs in
TypeScript and uploads the result. That cost more than every other
inefficiency in the recognition model put together: eleven nodes per run took
that path and they were a third of its time.

| | before | after |
|---|---|---|
| Add | 454 us/node | 11 us |
| Softmax | 1615 us | 624 us |
| rec, 4 threads | 16.5 ms | **8.2 ms** |

Only AveragePool still falls back. The lesson generalises: measure what leaves
wasm memory before tuning what happens inside it.

### Two kinds of parallelism

Detection and recognition need opposite shapes, which the measurements forced:

| | detection | recognition |
|---|---|---|
| tensors | megabytes | a few hundred KB |
| data-parallel speedup | 2.3x | **1.07x** |
| what runs in parallel | one kernel, split | one whole crop per worker |

The same size difference decides which optimisations pay at all. Folding the
five-node gelu into one kernel is worth 1.12x on detection at 960x960 and
nothing measurable on recognition, because a recognition tensor fits in cache
and the extra passes over it were nearly free.

`src/wasm/pool.ts` splits a single kernel across threads over one shared
linear memory. A job's range is cut into four blocks per thread and every
thread, the main one included, claims the next block with one atomic add;
each block writes a disjoint slice, so a job is one sequence bump, a claim
counter and a barrier, no locks. Static equal shares came first and measured
1.12x slower: the barrier waited on whichever thread finished last, and with
four threads one always did. That suits detection.

A recognition crop is too small for a split to pay while the graph still
dispatches 146 jobs, so `src/pipeline/rec-pool.ts` gives each worker a
complete rec Session and hands it whole crops instead, widest first. The cost
is one copy of the weights per worker, about 4.3 MB. An odd crop width makes
every downstream column count ragged and the GEMM's ragged tail is scalar, 16
to 17 px measured 2.0 to 2.9 ms, so widths pad up to a multiple of eight with
mid-grey, which is what PaddleOCR trains with.

### The GEMM

Sixteen accumulators, an 8 row by 8 column micro-kernel. That is what the
kernel is limited by, measured on a 1x1 convolution at 240x240:

| accumulators | GFLOP/s |
|---|---|
| 4 (4x4) | 19.6 |
| 8 (4x8) | 25.4 |
| 16 (8x8) | **35.9** |

Packing B into a contiguous panel was tried first and measured 0.99-1.05x, so
locality was not the constraint; the dependency chains were. The production
kernel reaches 33-36 GFLOP/s.

### The engine is the ceiling

ORT's 79.6 GFLOP/s single-threaded is not reachable from WebAssembly on this
machine, and the gap is not the kernel.

Twenty-four independent `f32x4.relaxed_madd` chains with no memory traffic at
all, which is the most favourable shape a FLOP benchmark can have, measure
**56.4 GFLOP/s** under Bun. That is the ceiling, and it sits below ORT's 79.6.
Native code is not reachable from here however good the kernel gets.

| chains | GFLOP/s |
|---|---|
| 8 | 30.5 |
| 16 | 54.8 |
| 24 | **56.4** |

The kernel at 33-36 is about 62% of that ceiling, so a perfect GEMM is worth
roughly 1.6x, not the 2.3x the raw ORT ratio suggests.

JavaScriptCore does not fuse `relaxed_madd`: an accumulator sweep with two
loads per twelve ops tops out at 2.19 instructions per cycle where the int8
dot product reaches 3.84, which is the signature of a two-instruction
lowering (multiply then add). That closes the question. fp32 on this engine
is four multiply-adds on two instructions; the relaxed dot product is
sixteen on one, and that ratio is where the int8 speed comes from.

### int8

Tensors inside the convolution backbone are int8 NHWC with a per-channel
scale and zero point, calibrated once (`bun tools/calibrate.ts <images>`
writes `models/*.calib.json`, which `Ocr.create` takes as `detCalib` and
`recCalib`). A planner (`src/runtime/qplan.ts`) walks the fused graph and
keeps whole regions on int8: 1x1, dense and depthwise convolutions with
relu or gelu in the epilogue, the residual add, squeeze-and-excite, the
FPN's add, 2x resize, concat and max-pool. Anything else gets fp32 back at
the boundary, so the head of each model, the recognition MatMuls and the
transposed convolutions run exactly as before. Weights are folded with the
input channel's scale, quantized per output channel and packed
`[K/4][N][4]` so one 16-byte load feeds the dot product; the zero points
fold into a per-channel compensation, so the kernel is a plain dot product
on raw bytes.

The relaxed dot product only promises a 7-bit second operand. ARM engines
lower it to `sdot` and read both operands signed; x86 engines lower it to
`pmaddubsw` and read the second as unsigned. `Arena.signedDot` probes this
at start-up, the way MLAS and XNNPACK do, and an engine that fails the
probe stays on fp32 rather than being silently wrong.

### Threads

The pool needs `SharedArrayBuffer`, so a page must be cross-origin isolated
(`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`; `tools/serve.ts` sends both).
Without them `canUseThreads()` is false and the runtime stays on one thread.

The default is every reported core capped at eight. Under static shares it
was four: every job ends at a barrier, and a share landing on an efficiency
core held up the rest, 164 ms on four threads against 180 on eight. Block
claiming removed that; see above.

Instrumenting every dispatch at four threads showed the main thread, which
runs share 0 and every serial op between kernels, finishing last on 99% of
GEMMs with workers idle about 35 ms a run. Scaling its share down did not
help: 1.0x a worker's share 121 ms, 0.5x 122, 0 136. Three workers doing all
the work took 40% longer per unit than one thread alone does. The loss is not
the split; every thread slows once four run. Column blocking of the GEMM was
neutral for the same reason, each thread's slice of B already fits L2.

`rust-lld` will not emit a shared memory, because the precompiled `libcore`
for this target was not built with the atomics feature and rebuilding it needs
nightly. The kernels do not want atomic instructions anyway, only the memory's
shared flag, so `tools/share-memory.ts` rewrites that one byte of the binary
and turns the defined memory into an import.

### Rotated text

Off by default (`rotated: true` to enable). Each region gets a minimum-area
rectangle and the crop is straightened before recognition. On tilted text that
is the difference between `Iam signty ite he` at 0.52 confidence and
`I am slightly tilted hehe` at 0.95. On upright pages it trades: it fixes
`Kembal ian` but turns `SMS/WA` into `SMS/HA`, so upright stays the default.

### Line filtering

`recognize(img, detectOpts, { minConfidence, dropSeparators })`. By default a
line needs 0.5 mean CTC confidence, which drops barcodes and logos, and a line
of nothing but rule characters (`+-`, `----`) is dropped as a printed
separator - those score high, so confidence alone never catches them.

## Verification

`tools/expose.ts` rewrites a model so every intermediate tensor is also a
graph output, `tools/golden.ts` dumps what ORT computes for them once, and
`tools/check.ts` diffs the runtime node by node and names the first node that
drifts. It caught two silent bugs that produce plausible output but wrong
pixels:

- paddle2onnx writes repeated attribute ints unpacked, so a naive reader keeps
  only the last. Stride `[2,2]` became `[2]`, which fell back to stride 1, and
  pads `[1,1,1,1]` became `[1]`, which fell back to 0.
- `auto_pad=SAME_UPPER` on MaxPool was ignored, giving 127x127 where the model
  wanted 128x128.

It also caught the worker pool's startup race, where a worker read the job
sequence number after reporting ready and so slept through the first job.

The tensor-level threading test caught a second race that the text-level one
had passed for a week: every worker instantiates the same module on the same
shared memory, and a module's `__stack_pointer` starts at the same address in
every instance, so all workers ran their shadow stacks on the same bytes. No
kernel spilled a local until the specialised 5x5 depthwise kernel put 25 taps
in play; after that, threaded detection differed from single-threaded on
about one run in three, by a few activations in one layer, and the text was
usually still right. Each worker now gets its own stack region from the arena.

## Measured and rejected

- **Batching recognition crops.** 0.87x at the tightest grouping, 0.72x at the
  loosest. A batch makes every intermediate N times larger, and losing cache
  locality costs more than the per-run overhead it saves. `recognizeBatch` is
  kept, correct and tested, but is not the default.
- **Packing the GEMM's B operand.** 0.99-1.05x. Removed.
- **Column blocking the GEMM.** Neutral at four threads from 128 columns to
  none, 107.3-108.0 ms. Each thread's slice of B already fits L2.
- **Spinning before parking a worker.** Detection 28% slower: a quarter of
  the graph is serial and three spinning workers steal cores from the thread
  doing it. Separating the sequence and completion counters onto different
  cache lines only mattered while spinning; also reverted.
- **Weighting the main thread's share.** See Threads. 1.0x was best.
- **Crop width alignment above eight.** 16 and 32 measured no faster end to
  end; the wall is set by the widest crops, which lose only 8% to raggedness.
- **Relaxed SIMD.** `f32x4.relaxed_madd` is in the kernels and is worth 15% on
  3x3 im2col and nothing on the 1x1 shapes. It needs Chrome 114, Safari 18 or
  Node 20; there is no fallback build.
- **`v128.load32_splat` for the GEMM's A operand.** One instruction instead of
  a scalar load plus a splat, and 0.90x measured. JavaScriptCore lowers it
  worse than the pair.
- **More than four detection threads, under static shares.** Slower on
  big.LITTLE. Reversed by block claiming; eight is the default now. Recognition
  workers stay at six: eight did not beat six.

## Not done

- **int8 on x86.** The kernels want a signed dot product; on x86 engines
  the probe fails and everything runs fp32. An `u8 x s8` variant with the
  weights offset by 128, which is what MLAS ships, would cover it.
- **Calibration breadth.** 16 SROIE receipts. The 6906-class recognition
  head is fp32, but the backbone's ranges come from receipts, and other
  documents may want their own calibration or a wider set.
- **The recognition pool in the browser.** The demo gets detection threads but
  runs recognition serially: `src/browser.ts` has no `makeRecWorker`, which
  needs `rec-worker.ts` bundled into a second inlined blob. Node and Bun get
  both pools.
- **AveragePool.** The one op still without a resident kernel, 451 us a run in
  recognition.
- **The 5x5 depthwise still spills.** 25 taps plus ten accumulator chains do
  not fit the register file; it runs at 3.4 GB/s where 3x3 runs at 9. Reloading
  five taps per kernel row would fit. About 1 ms at four threads.
- **The other models.** Only v6 tiny. Other exports may use ops not
  implemented here; the runtime throws by name if so.
- **Arithmetic-coded and 12-bit JPEG**, which nothing produces in practice.
