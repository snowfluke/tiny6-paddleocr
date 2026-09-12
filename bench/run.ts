// Each measurement runs in its own process: ORT's thread pool spins, and our
// worker pool holds cores, so whichever engine ran second would look slow.
const cases = [
  ["det", "models/det.onnx", "1x3x960x960", 3],
  ["det", "models/det.onnx", "1x3x256x256", 10],
  ["rec", "models/rec.onnx", "1x3x48x320", 10],
] as const;

const run = (args: string[]) => {
  const p = Bun.spawnSync(["bun", ...args]);
  if (p.exitCode !== 0) throw new Error(new TextDecoder().decode(p.stderr));
  return Number(new TextDecoder().decode(p.stdout).trim());
};

const head = ["model", "input", "1 thread", "4 threads", "ORT", "vs ORT"];
console.log(`${head[0].padEnd(5)} ${head[1].padEnd(13)} ${head[2].padStart(9)} ${head[3].padStart(10)} ${head[4].padStart(8)} ${head[5].padStart(7)}`);
for (const [name, file, dims, runs] of cases) {
  const one = run(["bench/one.ts", "ours", file, dims, String(runs)]);
  const four = run(["tools/thread-one.ts", file, dims, "4", String(runs)]);
  const ort = run(["bench/one.ts", "ort", file, dims, String(runs)]);
  console.log(
    `${name.padEnd(5)} ${dims.padEnd(13)} ${one.toFixed(0).padStart(6)} ms ${four.toFixed(0).padStart(7)} ms ${ort.toFixed(0).padStart(5)} ms ${(four / ort).toFixed(1).padStart(6)}x`,
  );
}
