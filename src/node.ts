// Node and Bun entry point. Adds the recognition-worker factory, which needs
// a module URL the bundler cannot provide.

export { Ocr, readingOrder, DEFAULT_MIN_CONFIDENCE } from "./ocr.ts";
export type { Assets, OcrLine, OcrWord } from "./ocr.ts";
export { decodePng } from "./image/png.ts";
/** Reads PNG or JPEG, dispatching on the magic bytes. */
export { decodeImage, decodeJpeg } from "./image/jpeg.ts";
export type { RGBA } from "./image/png.ts";
export { DEFAULT_DETECT, type Box, type DetectOptions } from "./pipeline/detect.ts";

/** Spawns the recognition worker from source; Bun and Node both load it directly. */
export function makeRecWorker(): Worker {
  return new Worker(new URL("./workers/rec-worker.ts", import.meta.url).href, { type: "module" });
}
