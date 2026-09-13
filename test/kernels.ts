// Which kernel build the tests load. TINY6_KERNELS=basic runs the whole
// suite on the build without relaxed SIMD, which is what Safari gets.
import { makeMemoryShared } from "../src/wasm/share-memory.ts";

export const KERNELS = process.env.TINY6_KERNELS === "basic" ? "src/wasm/kernels.basic.wasm" : "src/wasm/kernels.wasm";
export const wasm = new Uint8Array(await Bun.file(KERNELS).arrayBuffer());
export const sharedWasm = makeMemoryShared(wasm, 32768);
