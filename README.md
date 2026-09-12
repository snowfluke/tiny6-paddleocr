# tiny6-paddleocr

PP-OCRv6 tiny with **zero runtime dependencies**. A hand-written ONNX
interpreter plus WASM SIMD kernels replace onnxruntime and OpenCV.

Proof of concept. Runs on Node/Bun and in the browser.

## Size

| part | size |
|---|---|
| runtime, browser bundle | **90 KB** (26.7 KB gzipped) |
| of which WASM kernels | 17,544 bytes, twice (plain + shared memory) |
| detection model | 1.70 MB |
| recognition model | 4.26 MB |
| dictionary (6,906 entries) | 0.03 MB |
| **total** | **6.02 MB** on the wire, 6.08 MB unpacked |

For comparison, `onnxruntime-web`'s smallest WASM binary is 14 MB and
`onnxruntime-node` unpacks to 85 MB per platform. OpenCV.js adds 15 MB.

## Speed

darwin arm64. ORT uses every core; the "4 threads" column is ours with the
worker pool on. Each figure is measured in its own process, because both
engines hold cores and whichever ran second would look slow.

| model | input | 1 thread | 4 threads | ORT | vs ORT |
|---|---|---|---|---|---|
| det | 1x3x960x960 | 302 ms | **121 ms** | 55 ms | 2.2x |
| det | 1x3x256x256 | 25 ms | **12 ms** | 4 ms | 3.1x |
| rec | 1x3x48x320 | 20 ms | **14 ms** | 3 ms | 4.7x |

End to end on a 720x1280 receipt: detection ~140 ms, recognition ~245 ms for
28 crops across four workers, **~385 ms total**.

Four threads is the default because this machine has four performance cores.
Measured at 960x960: 2 threads 180 ms, 4 threads 121 ms, 5 threads 140 ms,
6 threads 131 ms, 8 threads 129 ms. Every job ends at a barrier, so a share
that lands on an efficiency core holds up the other three.

Detection at 960x960, as the work landed:

| | ms |
|---|---|
| first working graph, all TypeScript | 3391 |
| convolutions in WASM | 1170 |
| activations resident in WASM memory | 445 |
| worker pool | 214 |
| 8x8 GEMM micro-kernel | 144 |
| gelu fused, concat and transposed conv kept in wasm | **121** |

## Accuracy

Checked against `ppu-paddle-ocr` (onnxruntime + OpenCV) on the same file.

- Every intermediate tensor of both models matches ORT to 2e-3, on the
  TypeScript path and the WASM path. 242 tensors for detection, 219 for
  recognition. Worker threads produce bit-identical output to one thread, and
  the recognition pool returns exactly what the serial path does.
- Receipt: 13 of 18 lines byte-identical with filtering off. Two of the five
  that differ are barcode noise both implementations read as garbage; three
  differ by one character because the crops differ by a pixel or two.
- The same receipt as a progressive JPEG, decoded here, yields identical text
  to the PNG.

## Run it

```sh
bun install            # onnxruntime-node, dev only, for goldens and benchmarks
bun run build          # kernels.wasm, kernels.shared.wasm, demo/tiny6.js
bun test               # goldens, geometry, threading, JPEG, end-to-end OCR
bun run typecheck
bun run ocr test/images/receipt.jpg
bun run ocr test/images/tilted.png --rotated
bun run bench
bun run bench:gemm
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
linear memory: each share writes a disjoint slice, so a job is one sequence
bump plus a barrier, no locks. That suits detection.

A recognition crop is too small for a split to pay while the graph still
dispatches 219 jobs, so `src/pipeline/rec-pool.ts` gives each worker a
complete rec Session and hands it whole crops instead: 3.6x on a 28-crop
receipt. The cost is one copy of the weights per worker, about 4.3 MB.

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

Whether an engine turns `relaxed_madd` into one hardware instruction is not
settled here, and the obvious microbenchmarks do not answer it: writing the
same loop as a separate multiply and add lets the compiler hoist the
loop-invariant multiply out, so the two are not doing equal work. What is
measured is the kernel itself, where the fused form is neutral on the 1x1
shapes and worth 15% on 3x3 im2col.

### Threads

The pool needs `SharedArrayBuffer`, so a page must be cross-origin isolated
(`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`; `tools/serve.ts` sends both).
Without them `canUseThreads()` is false and the runtime stays on one thread.

The default is half the reported cores capped at four. Every job ends at a
barrier, so on a big.LITTLE machine a share landing on an efficiency core
holds up the rest: four threads ran 164 ms where eight ran 180 ms.

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

## Measured and rejected

- **Batching recognition crops.** 0.87x at the tightest grouping, 0.72x at the
  loosest. A batch makes every intermediate N times larger, and losing cache
  locality costs more than the per-run overhead it saves. `recognizeBatch` is
  kept, correct and tested, but is not the default.
- **Packing the GEMM's B operand.** 0.99-1.05x. Removed.
- **Relaxed SIMD.** `f32x4.relaxed_madd` is in the kernels and is worth 15% on
  3x3 im2col and nothing on the 1x1 shapes. It needs Chrome 114, Safari 18 or
  Node 20; there is no fallback build.
- **`v128.load32_splat` for the GEMM's A operand.** One instruction instead of
  a scalar load plus a splat, and 0.90x measured. JavaScriptCore lowers it
  worse than the pair.
- **More than four threads.** Slower on big.LITTLE. See above.

## Not done

- **A better GEMM still.** 33-36 GFLOP/s against the engine's 56.4 ceiling.
  Worth about 1.6x on the convolutions if it were perfect, and convolutions
  are 58% of threaded detection, so call it 1.3x end to end. Cache-tiled loops
  with proper MR x NR blocking are the next step.
- **The recognition pool in the browser.** The demo gets detection threads but
  runs recognition serially: `src/browser.ts` has no `makeRecWorker`, which
  needs `rec-worker.ts` bundled into a second inlined blob. Node and Bun get
  both pools.
- **The other models.** Only v6 tiny. Other exports may use ops not
  implemented here; the runtime throws by name if so.
- **Arithmetic-coded and 12-bit JPEG**, which nothing produces in practice.
