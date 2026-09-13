# tiny6-paddleocr

PP-OCRv6 tiny text detection and recognition with zero runtime dependencies.
A hand-written ONNX interpreter in TypeScript and hand-written WebAssembly
SIMD kernels in Rust replace onnxruntime and OpenCV. Runs on Bun, Node and
in the browser from one 186 KB bundle plus the 6 MB of models.

Research code. The question it answers: can WebAssembly beat native
onnxruntime on this model? Yes, on ARM, with int8. See Benchmarks.

## Highlights

- No onnxruntime, no OpenCV, no npm dependencies at runtime. The ONNX
  files are parsed and executed here; the image decoders are here too.
- 41 KB of WASM kernels: GEMM, depthwise, im2col, elementwise, softmax,
  resize, and int8 versions of the convolution backbone.
- Multi-threaded through a worker pool on shared memory, plus a task-parallel
  recognition pool.
- int8 inference on the `relaxed_dot` instruction: 1.4x faster than native
  onnxruntime end to end on Apple Silicon, no measured accuracy loss on 60
  receipts.
- Every intermediate tensor checked against onnxruntime to 2e-3; kernels
  pinned to integer references bit for bit; threaded runs bit-identical to
  single-threaded ones.

## Quick start

```sh
bun install                    # dev only: onnxruntime for goldens and benchmarks
bun run build:web              # demo/tiny6.js; no Rust needed, kernels are committed
bun run ocr test/images/receipt.jpg
bun run demo                   # http://localhost:8099
bun test
```

Bun or Node:

```ts
import { readFile } from "node:fs/promises";
import { decodeImage, makeRecWorker, Ocr } from "./src/node.ts";

const ocr = await Ocr.create({
  det: await readFile("models/det.onnx"),
  rec: await readFile("models/rec.onnx"),
  dict: await readFile("models/dict.txt", "utf8"),
  wasm: await readFile("src/wasm/kernels.wasm"),
  wasmShared: await readFile("src/wasm/kernels.shared.wasm"),
  detCalib: await readFile("models/det.calib.json", "utf8"), // optional: int8
  recCalib: await readFile("models/rec.calib.json", "utf8"),
  makeRecWorker,                                              // optional: rec pool
});
const lines = await ocr.recognizeAsync(await decodeImage(bytes));
// [{ text, confidence, words: [{ text, confidence, box }] }, ...]
ocr.destroy();
```

Browser, from the bundle (kernels inlined, threads when the page is
cross-origin isolated):

```ts
import { createOcr, decodeImage } from "./tiny6.js";

const ocr = await createOcr({
  detUrl: "models/det.onnx", recUrl: "models/rec.onnx", dictUrl: "models/dict.txt",
  detCalibUrl: "models/det.calib.json", recCalibUrl: "models/rec.calib.json",
});
const lines = ocr.recognize(await decodeImage(file));
```

Chrome 114+, Firefox 145+, Node 21+ and Bun get the relaxed-SIMD kernels;
Safari, which keeps relaxed SIMD behind a flag, gets a basic build of the
same kernels (fp32 only, about 15% slower), chosen at load. Threads need `Cross-Origin-Opener-Policy:
same-origin` and `Cross-Origin-Embedder-Policy: require-corp`
(`demo/_headers`, `tools/serve.ts`).

## Benchmarks

720x1280 receipt, 28 text boxes, warm minimum of six runs, same minute,
Apple M1 (4P+4E), about a quarter of the machine busy with other work.

| pipeline | ms | vs native |
|---|---|---|
| ppu-paddle-ocr (onnxruntime-node native ARM64 + OpenCV) | 121 | 1.00x |
| tiny6-paddleocr fp32, 8 threads + 6 rec workers | 142 | 0.85x |
| **tiny6-paddleocr int8**, same | **86** (detect 41 + recognize 48) | **1.41x** |

Per model, one thread:

| model | input | fp32 | int8 | onnxruntime-web fp32 |
|---|---|---|---|---|
| detection | 1x3x960x544 | 171 ms | 120 ms | - |
| detection | 1x3x960x960 | 272 ms | - | 246 ms |
| recognition | 1x3x48x320 | 14 ms | 8 ms | 11.8 ms |

Numbers on a developer machine drift by 60% with load; the ratios come from
interleaved A/B runs (`bun run ab`). Peak WASM memory for detection at
960x960 is 168 MB.

Accuracy, 60 SROIE receipts (33,761 ground-truth characters), calibration
from 16 other receipts, scored by order-independent token F1 because the
SROIE transcripts are uppercased and split inconsistently:

| | token F1 | CER |
|---|---|---|
| fp32 | 79.13% | 13.60% |
| int8 | 81.00% | 13.45% |

The difference is inside the run-to-run band: no loss. Against
ppu-paddle-ocr on the test receipt, 13 of 18 lines are byte-identical with
filtering off; the rest differ by a character where the crops differ by a
pixel, or are barcode noise both read as garbage.

## Models

PP-OCRv6 tiny, exported by paddle2onnx, unchanged. Other exports may use ops
not implemented here; the runtime throws by name.

| file | size | notes |
|---|---|---|
| `models/det.onnx` | 1.70 MB | DB text detection, 242 nodes |
| `models/rec.onnx` | 4.26 MB | CTC recognition, 219 nodes, 6,906 classes |
| `models/dict.txt` | 0.03 MB | character dictionary |
| `models/*.calib.json` | 0.1 MB | per-channel activation ranges for int8 |

`bun tools/calibrate.ts <images...>` rewrites the calibration files from
your own images.

## How it works

- **Interpreter.** `src/onnx/parse.ts` reads the protobuf; `src/runtime/graph.ts`
  executes it. Every activation lives in WASM linear memory and ops run on
  pointers; a node without a kernel falls back to TypeScript at the cost of
  a download and upload, and only AveragePool still does.
- **Fusion** (`src/runtime/fuse.ts`): Identity removal, BatchNorm folded
  into weights, the five-node Gelu into one, conv bias, relu and the
  residual add into the GEMM epilogue. Detection 242 -> 161 nodes,
  recognition 219 -> 78.
- **Kernels** (`src/wasm/kernels.rs`, `no_std`): an 8x8 fp32 micro-kernel
  with sixteen FMA chains, im2col in cache-sized strips, depthwise
  specialised per kernel shape with stride-2 lane gathers.
- **Threads** (`src/wasm/pool.ts`): one shared memory, a control block, a
  sequence counter; each job is cut into blocks that threads claim with an
  atomic add, so a slow efficiency core takes fewer. Each worker has its own
  shadow stack. Recognition runs crops in a separate task-parallel pool.
- **int8** (`src/runtime/qplan.ts`): a planner keeps whole regions of the
  graph on int8 NHWC tensors with per-channel scale and zero point:
  convolutions with relu or gelu in the epilogue, residual add,
  squeeze-and-excite, the FPN's add, resize, concat, max-pool. Per-channel
  activation scales fold into the next layer's weights, zero points into a
  per-channel bias term, so the kernel is a plain dot product on bytes.
  Weights are packed `[K/4][N][4]` for `i32x4.relaxed_dot_i8x16_i7x16_add_s`:
  sixteen multiply-adds per instruction against four for an FMA, and the
  engine does not even fuse the FMA.
- **Portability probe.** The dot product only promises a 7-bit second
  operand. ARM engines lower it to `sdot` (signed); x86 engines to
  `pmaddubsw` (unsigned). `Arena.signedDot` probes at start-up and an engine
  that fails stays on fp32 instead of being silently wrong. Brave reports a
  random core count per site; `createOcr` measures the real one there.

## Tools

| command | what |
|---|---|
| `bun run build` | kernels (needs the pinned Rust toolchain) and bundle |
| `bun run check` | every tensor of both models against onnxruntime |
| `bun run ab <rev> <model> <dims> <threads>` | interleaved A/B against a git revision |
| `bun run bench:pipeline [--int8]` | warm end-to-end timing |
| `bun tools/profile.ts <det\|rec> <dims> --wasm [--int8] [--threads N]` | time per op |
| `bun tools/calibrate.ts <images...>` | write the int8 calibration |
| `bun tools/fakequant.ts`, `bun tools/sroie.ts` | quantization accuracy harness |
| `bun run deploy:demo` | Cloudflare static Worker over `demo/` |

## Status

- CI runs the suite on Linux x86-64, Linux ARM64, macOS ARM64 and Windows
  x86-64, plus the basic kernel build; verified by hand on macOS in Chrome
  and Brave.
- int8 needs the signed dot product, so x86 runs fp32 today. A 7-bit weight
  variant would lift that.
- Safari runs the basic kernels, fp32 only; not yet tried on a real Safari.
- The int8 path is batch-1; `recognizeBatch` with a calibration throws.
- Calibration covers receipts.
- Determinism is checked every run under load; one 60-image run under
  memory pressure once produced different fp32 text and has not reproduced.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for the models.
- [ppu-paddle-ocr](https://github.com/snowfluke/ppu-paddle-ocr), the
  onnxruntime + OpenCV pipeline this one is checked against.
- onnxruntime, whose graph fusions and MLAS kernels set the shapes here,
  and whose outputs are the goldens.
- [SROIE](https://rrc.cvc.uab.es/?ch=13) for the receipts.
