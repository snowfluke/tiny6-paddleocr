// Browser entry. The kernels are inlined as base64 at build time, so the page
// loads one script plus the two model files and nothing else.

import { Ocr, type OcrLine } from "./ocr.ts";
import type { RGBA } from "./image/png.ts";
import { decodePng } from "./image/png.ts";

/** Replaced by tools/build-web.ts with the real base64 payloads. */
declare const __KERNELS_B64__: string;
declare const __KERNELS_SHARED_B64__: string;

/**
 * Worker threads need SharedArrayBuffer, which a page only gets when it is
 * cross-origin isolated (COOP: same-origin plus COEP: require-corp). Without
 * those headers this returns false and the runtime stays single-threaded.
 */
export function canUseThreads(): boolean {
  return typeof SharedArrayBuffer !== "undefined" &&
    (typeof crossOriginIsolated === "undefined" || crossOriginIsolated);
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

export type WebAssets = {
  detUrl: string;
  recUrl: string;
  dictUrl: string;
  /** Defaults to half the cores when the page is cross-origin isolated. */
  threads?: number;
};

export async function createOcr(
  assets: WebAssets,
  onProgress?: (what: string) => void,
): Promise<Ocr> {
  const fetchBytes = async (url: string) => {
    onProgress?.(url.split("/").pop() ?? url);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const [det, rec, dict] = await Promise.all([
    fetchBytes(assets.detUrl),
    fetchBytes(assets.recUrl),
    fetchBytes(assets.dictUrl),
  ]);
  onProgress?.("kernels");
  const threaded = canUseThreads();
  return Ocr.create({
    det,
    rec,
    dict: new TextDecoder().decode(dict),
    wasm: base64ToBytes(__KERNELS_B64__),
    wasmShared: threaded ? base64ToBytes(__KERNELS_SHARED_B64__) : undefined,
    threads: assets.threads ?? (threaded ? undefined : 1),
  });
}

export { decodePng, Ocr };
export type { OcrLine, RGBA };
