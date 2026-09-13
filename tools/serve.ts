// Dev server for the demo. The COOP/COEP pair is what makes
// SharedArrayBuffer available, and without it the runtime cannot start
// worker threads and silently falls back to one.
import { stageDemo } from "./stage-demo.ts";

const root = "demo";
await stageDemo(root);
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
