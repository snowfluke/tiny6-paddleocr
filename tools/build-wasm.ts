// Builds the kernels three times from one source: a plain module that owns
// its memory, a threaded one that imports a shared memory so workers can
// attach to it, and a basic one without relaxed SIMD for engines that lack
// it (Safari). The browser derives shared variants at load, so the basic
// build has none on disk.
const COMMON = [
  "rustc", "--target", "wasm32-unknown-unknown", "-O",
  "-C", "opt-level=3",
  "-C", "panic=abort",
  "-C", "debuginfo=0",
  "-C", "strip=symbols",
  "--crate-type", "cdylib",
  "-C", "link-arg=--export=__stack_pointer",
];

async function build(out: string, extra: string[]) {
  const p = Bun.spawnSync([...COMMON, ...extra, "src/wasm/kernels.rs", "-o", out]);
  const err = new TextDecoder().decode(p.stderr);
  if (p.exitCode !== 0) {
    console.error(err);
    process.exit(1);
  }
  const real = err.split("\n").filter((l) => l.startsWith("error")).join("\n");
  if (real) console.error(real);
  console.log(`${out.padEnd(32)} ${Bun.file(out).size} bytes`);
}

await build("src/wasm/kernels.wasm", ["-C", "target-feature=+simd128,+relaxed-simd"]);
await build("src/wasm/kernels.basic.wasm", ["-C", "target-feature=+simd128"]);

const { makeMemoryShared } = await import("../src/wasm/share-memory.ts");
const plain = new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer());
const shared = makeMemoryShared(plain, 32768);
await Bun.write("src/wasm/kernels.shared.wasm", shared);
console.log(`${"src/wasm/kernels.shared.wasm".padEnd(32)} ${shared.length} bytes (imports shared memory)`);
