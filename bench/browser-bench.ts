// Times the browser path in a real Chrome, which the other benches cannot
// reach: they run on Bun or Node, where the kernels are the same but the
// worker pool, the cross-origin-isolation check and decodeImage are not.
//
//   bun run demo                                   # in one terminal
//   bun bench/browser-bench.ts                     # in another
//
// The page must be the bundle that `bun run build` and tools/serve.ts stage.
// Spawns its own headless Chrome with a private profile, so it never touches a
// browser you already have open. Env: CHROME, BASE, CDP_PORT.
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/** Chrome's own location, or whatever CHROME points at. */
function chromePath(): string {
  if (process.env.CHROME) return process.env.CHROME;
  if (process.platform === "win32") return "C:/Program Files/Google/Chrome/Application/chrome.exe";
  if (process.platform === "darwin") return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  return "google-chrome";
}
const CHROME = chromePath();
// BASE lets the same harness point at a server that sends no COOP/COEP, which
// is how the worker pool is checked on a page that is not cross-origin isolated.
const BASE = process.env.BASE ?? "http://localhost:8099";
const PAGE = `${BASE}/`;
const PORT = Number(process.env.CDP_PORT ?? 9222);
const profile = mkdtempSync(join(tmpdir(), "tiny6-chrome-"));

const chrome = spawn(CHROME, [
  "--headless=new",
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  "--remote-allow-origins=*",
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-extensions",
  "--disable-background-networking",
  "about:blank",
], { stdio: "ignore" });

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const cleanup = () => { chrome.kill(); try { rmSync(profile, { recursive: true, force: true }); } catch {} };

async function waitForPort() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (r.ok) return await r.json();
    } catch {}
    await sleep(250);
  }
  throw new Error("chrome did not open its debugging port");
}

const version = await waitForPort();
console.log(`chrome ${version.Browser}`);

// Open the page as its own target; connect straight to that target's socket so
// there is no session-attach plumbing.
await fetch(`http://127.0.0.1:${PORT}/json/new?${PAGE}`, { method: "PUT" });
await sleep(1500);

const tabs: any[] = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const tab = tabs.find((t) => t.type === "page" && t.url.startsWith(BASE));
if (!tab) throw new Error(`no debug target for ${BASE} (saw ${tabs.map((t) => t.url).join(", ")})`);

const BENCH = `(async () => {
  const { createOcr, decodeImage } = await import("./tiny6.js");
  const out = [];
  out.push("crossOriginIsolated=" + crossOriginIsolated +
           "  SharedArrayBuffer=" + (typeof SharedArrayBuffer !== "undefined") +
           "  cores=" + navigator.hardwareConcurrency);
  const ocr = await createOcr({
    detUrl: "models/det.onnx", recUrl: "models/rec.onnx", dictUrl: "models/dict.txt",
    detCalibUrl: "models/det.calib.json", recCalibUrl: "models/rec.calib.json",
  });
  const blob = await (await fetch("receipt.png")).blob();
  const time = (n, fn) => { fn(); const t = performance.now(); for (let i = 0; i < n; i++) fn(); return (performance.now() - t) / n; };
  const timeAsync = async (n, fn) => { await fn(); const t = performance.now(); for (let i = 0; i < n; i++) await fn(); return (performance.now() - t) / n; };
  const dec = await timeAsync(5, () => decodeImage(blob));
  const img = await decodeImage(blob);
  out.push("threads=" + ocr.arena.threads + "  dotMode=" + ocr.arena.dotMode +
           "  int8=" + ocr.int8 + "  recWorkers=" + ocr.recWorkers +
           "  image=" + img.width + "x" + img.height);
  const opts = ocr.detectDefaults;
  const det = time(5, () => ocr.detect(img, opts));
  const boxes = ocr.detect(img, opts).boxes;
  out.push("boxes=" + boxes.length);
  const recSync = time(5, () => ocr.recognizeBoxes(img, boxes));
  const recAsync = await timeAsync(5, () => ocr.recognizeBoxesAsync(img, boxes));
  // Whole pipeline, both ways: old = detect + recognize synchronously on the
  // shared arena; new = detect + whole crops on the worker pool.
  const pipeOld = await timeAsync(5, async () => { const b = ocr.detect(img, opts).boxes; ocr.recognizeBoxes(img, b); });
  const pipeNew = await timeAsync(5, async () => { const b = ocr.detect(img, opts).boxes; await ocr.recognizeBoxesAsync(img, b); });
  out.push("decode        " + dec.toFixed(1) + " ms   (createImageBitmap + drawImage + getImageData)");
  out.push("detect        " + det.toFixed(1) + " ms   (resize+normalize, det graph, DBNet boxes)");
  out.push("recognize     " + recSync.toFixed(1) + " ms   (" + boxes.length + " crops, sync/calling thread)");
  out.push("recognize xN  " + recAsync.toFixed(1) + " ms   (" + boxes.length + " crops, worker pool)");
  out.push("  rec speedup " + (recSync / recAsync).toFixed(2) + "x");
  out.push("pipeline old  " + pipeOld.toFixed(1) + " ms   (detect + sync recognize)");
  out.push("pipeline new  " + pipeNew.toFixed(1) + " ms   (detect + worker pool)");
  out.push("  pipeline    " + (pipeOld / pipeNew).toFixed(2) + "x");
  return out.join("\\n");
})()`;

const ws = new WebSocket(tab.webSocketDebuggerUrl);
const reply = await new Promise<string>((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error("CDP evaluate timed out")), 180000);
  ws.onopen = () => ws.send(JSON.stringify({
    id: 1,
    method: "Runtime.evaluate",
    params: { expression: BENCH, awaitPromise: true, returnByValue: true },
  }));
  ws.onerror = (e) => { clearTimeout(timer); reject(new Error(`ws error: ${e}`)); };
  ws.onmessage = (m) => {
    const msg = JSON.parse(String(m.data));
    if (msg.id !== 1) return;
    clearTimeout(timer);
    if (msg.result?.exceptionDetails) {
      reject(new Error(JSON.stringify(msg.result.exceptionDetails.exception?.description ?? msg.result.exceptionDetails)));
    } else resolve(msg.result?.result?.value ?? "(no value)");
  };
});

console.log(reply);
ws.close();
cleanup();
process.exit(0);
