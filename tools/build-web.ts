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
for (const f of ["models/det.onnx", "models/rec.onnx", "models/dict.txt"]) {
  console.log(`${f.padEnd(18)} ${(Bun.file(f).size / 1024 / 1024).toFixed(2)} MB`);
}
