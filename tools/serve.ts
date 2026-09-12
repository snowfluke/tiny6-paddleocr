// Dev server for the demo. The COOP/COEP pair is what makes
// SharedArrayBuffer available, and without it the runtime cannot start
// worker threads and silently falls back to one.
const root = "demo";

// demo/models is gitignored, so stage it here rather than making a fresh
// clone fail with three 404s.
for (const f of ["models/det.onnx", "models/rec.onnx", "models/dict.txt"]) {
  const dst = `${root}/${f}`;
  if (!(await Bun.file(dst).exists())) await Bun.write(dst, Bun.file(f));
}
if (!(await Bun.file(`${root}/receipt.png`).exists())) {
  await Bun.write(`${root}/receipt.png`, Bun.file("test/images/receipt.png"));
}
const server = Bun.serve({
  port: 8099,
  async fetch(req) {
    const p = new URL(req.url).pathname;
    const file = Bun.file(root + (p === "/" ? "/index.html" : p));
    if (!(await file.exists())) return new Response("not found", { status: 404 });
    return new Response(file, {
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    });
  },
});
console.log(`http://localhost:${server.port}`);
