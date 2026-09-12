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
| det | 1x3x960x960 | 336 ms | **144 ms** | 55 ms | 2.6x |
| det | 1x3x256x256 | 27 ms | **16 ms** | 4 ms | 4.4x |
| rec | 1x3x48x320 | 20 ms | **16 ms** | 3 ms | 5.5x |

End to end on a 720x1280 receipt: detection ~160 ms, recognition ~250 ms for
28 crops across four workers, **~400 ms total**.

Detection at 960x960, as the work landed:

| | ms |
|---|---|
| first working graph, all TypeScript | 3391 |
| convolutions in WASM | 1170 |
| activations resident in WASM memory | 445 |
| worker pool | 214 |
| 8x8 GEMM micro-kernel | **144** |

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
kernel reaches 33-40 GFLOP/s against ORT's 79.6 single-threaded.

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
- **More than four threads.** Slower on big.LITTLE. See above.

## Not done

- **A better GEMM still.** 33-40 GFLOP/s against ORT's 79.6 single-threaded is
  most of the remaining gap. Real packing with a proper MR x NR blocking and
  cache-tiled loops is the next step.
- **The recognition pool in the browser.** The demo gets detection threads but
  runs recognition serially: `src/browser.ts` has no `makeRecWorker`, which
  needs `rec-worker.ts` bundled into a second inlined blob. Node and Bun get
  both pools.
- **The other models.** Only v6 tiny. Other exports may use ops not
  implemented here; the runtime throws by name if so.
- **Arithmetic-coded and 12-bit JPEG**, which nothing produces in practice.
