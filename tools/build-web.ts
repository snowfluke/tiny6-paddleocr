// Bundle the browser entry, inlining the relaxed-SIMD kernels and the basic
// ones, so the page fetches only the script and the model files. The shared
// memory variants are derived at load (src/wasm/share-memory.ts).
const read = async (f: string) => new Uint8Array(await Bun.file(f).arrayBuffer());
const plain = await read("src/wasm/kernels.wasm");
const basic = await read("src/wasm/kernels.basic.wasm");

const out = await Bun.build({
  entrypoints: ["src/browser.ts"],
  target: "browser",
  format: "esm",
  minify: true,
  define: {
    __KERNELS_B64__: JSON.stringify(Buffer.from(plain).toString("base64")),
    __KERNELS_BASIC_B64__: JSON.stringify(Buffer.from(basic).toString("base64")),
  },
});
if (!out.success) {
  for (const m of out.logs) console.error(m);
  process.exit(1);
}
const js = await out.outputs[0].text();
await Bun.write("demo/tiny6.js", js);

const gz = Bun.gzipSync(new TextEncoder().encode(js));
console.log(`demo/tiny6.js      ${(js.length / 1024).toFixed(1)} KB  (${(gz.length / 1024).toFixed(1)} KB gzipped)`);
console.log(`  kernels          ${plain.length} bytes relaxed, ${basic.length} bytes basic`);

// The recognition worker ships as its own bundle. The main bundle makes one
// with new URL("./rec-worker.js", import.meta.url), so it has to sit beside it
// and be self-contained: a worker is loaded as a module, not off the main one.
const wout = await Bun.build({
  entrypoints: ["src/workers/rec-worker.ts"],
  target: "browser",
  format: "esm",
  minify: true,
});
if (!wout.success) {
  for (const m of wout.logs) console.error(m);
  process.exit(1);
}
const wjs = wout.outputs.find((o) => o.path.endsWith(".js")) ?? wout.outputs[0];
const wsrc = await wjs.text();
await Bun.write("demo/rec-worker.js", wsrc);
console.log(`demo/rec-worker.js ${(wsrc.length / 1024).toFixed(1)} KB`);
for (const f of ["models/det.onnx", "models/rec.onnx", "models/dict.txt"]) {
  console.log(`${f.padEnd(18)} ${(Bun.file(f).size / 1024 / 1024).toFixed(2)} MB`);
}
