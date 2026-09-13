// Browser entry. The kernels are inlined as base64 at build time, so the page
// loads one script plus the two model files and nothing else.

import { Ocr, type OcrLine } from "./ocr.ts";
import type { RGBA } from "./image/png.ts";
import { decodePng } from "./image/png.ts";

/** Replaced by tools/build-web.ts with the real base64 payloads. */
declare const __KERNELS_B64__: string;
declare const __KERNELS_BASIC_B64__: string;

/**
 * Worker threads need SharedArrayBuffer, which a page only gets when it is
 * cross-origin isolated (COOP: same-origin plus COEP: require-corp). Without
 * those headers this returns false and the runtime stays single-threaded.
 */
export function canUseThreads(): boolean {
  return typeof SharedArrayBuffer !== "undefined" &&
    (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
}

/**
 * One function taking three v128s and returning f32x4.relaxed_madd of them.
 * The kernels are built with relaxed SIMD, so an engine that cannot validate
 * this cannot run them: Chrome 114, Firefox 145 and Node 21 are the floors;
 * Safari keeps relaxed SIMD behind a flag.
 */
const RELAXED_SIMD_PROBE = Uint8Array.from([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x08, 0x01, 0x60, 0x03, 0x7b, 0x7b, 0x7b, 0x01, 0x7b,
  0x03, 0x02, 0x01, 0x00,
  0x0a, 0x0d, 0x01, 0x0b, 0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0xfd, 0x85, 0x02, 0x0b,
]);

/** False on a browser too old for the kernels. Check before createOcr. */
export function canRunKernels(): boolean {
  return typeof WebAssembly !== "undefined" && WebAssembly.validate(RELAXED_SIMD_PROBE);
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Decodes anything the browser can decode, which includes JPEG and WebP.
 * The Node path only reads PNG; here the platform does the work.
 */
export async function decodeImage(blob: Blob): Promise<RGBA> {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return { width: img.width, height: img.height, data: new Uint8Array(img.data.buffer) };
}

import { measureCores, reportsFakeCores } from "./cores.ts";
import { makeMemoryShared } from "./wasm/share-memory.ts";

export type WebAssets = {
  detUrl: string;
  recUrl: string;
  dictUrl: string;
  /** Calibration JSON from tools/calibrate.ts; with both, the convolutions run on int8 where the engine allows. */
  detCalibUrl?: string;
  recCalibUrl?: string;
  /** Defaults to half the cores when the page is cross-origin isolated. */
  threads?: number;
};

export async function createOcr(
  assets: WebAssets,
  onProgress?: (what: string) => void,
): Promise<Ocr> {
  // Relaxed SIMD (Chrome 114, Firefox 145, Node 21) gets the fast kernels;
  // Safari gets the basic build: same math, no int8, about 15% slower.
  const relaxed = canRunKernels();
  const fetchBytes = async (url: string) => {
    onProgress?.(url.split("/").pop() ?? url);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const text = async (url?: string) => (url ? new TextDecoder().decode(await fetchBytes(url)) : undefined);
  const [det, rec, dict, detCalib, recCalib] = await Promise.all([
    fetchBytes(assets.detUrl),
    fetchBytes(assets.recUrl),
    fetchBytes(assets.dictUrl),
    text(assets.detCalibUrl),
    text(assets.recCalibUrl),
  ]);
  onProgress?.("kernels");
  const threaded = canUseThreads();
  // Brave reports a random core count per site; measure instead (src/cores.ts).
  const threads = assets.threads ?? (threaded && reportsFakeCores() ? await measureCores() : undefined);
  const wasm = base64ToBytes(relaxed ? __KERNELS_B64__ : __KERNELS_BASIC_B64__);
  return Ocr.create({
    det,
    rec,
    dict: new TextDecoder().decode(dict),
    wasm,
    wasmShared: threaded ? makeMemoryShared(wasm, 32768) : undefined,
    threads: threaded ? threads : 1,
    detCalib,
    recCalib,
  });
}

export { decodePng, measureCores, Ocr };
export type { OcrLine, RGBA };
