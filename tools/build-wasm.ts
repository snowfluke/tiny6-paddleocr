// Builds the kernels twice: a plain module that owns its memory, and a
// threaded one that imports a shared memory so workers can attach to it.
// Both come from the same source; only the linker flags differ.
const COMMON = [
  "rustc", "--target", "wasm32-unknown-unknown", "-O",
  "-C", "opt-level=3",
  "-C", "panic=abort",
  "-C", "debuginfo=0",
  "-C", "strip=symbols",
  "--crate-type", "cdylib",
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

const { makeMemoryShared } = await import("./share-memory.ts");
const plain = new Uint8Array(await Bun.file("src/wasm/kernels.wasm").arrayBuffer());
const shared = makeMemoryShared(plain, 32768);
await Bun.write("src/wasm/kernels.shared.wasm", shared);
console.log(`${"src/wasm/kernels.shared.wasm".padEnd(32)} ${shared.length} bytes (imports shared memory)`);
