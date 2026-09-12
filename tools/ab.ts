// Interleaved A/B against a git revision.
//
// Absolute timings on a developer machine are worthless: over this project's
// history the same build measured 116 ms and 193 ms for the same work,
// depending on what else the machine was doing. Ratios survive that, but only
// if both sides are measured in the same conditions, so this alternates which
// side runs first in each round and reports the minimum of each.
//
//   bun tools/ab.ts <rev> <model.onnx> <dims> <threads> [runs] [rounds]
//   bun tools/ab.ts HEAD models/det.onnx 1x3x960x960 4

import { mkdirSync, rmSync } from "node:fs";

const [rev, model, dims, threadsArg, runsArg, roundsArg] = process.argv.slice(2);
if (!rev || !model || !dims || !threadsArg) {
  console.error("usage: bun tools/ab.ts <rev> <model.onnx> <dims> <threads> [runs] [rounds]");
  process.exit(2);
}
const runs = runsArg ?? "3";
const rounds = Number(roundsArg ?? 4);

const base = `/tmp/tiny6-ab-${rev.replace(/[^\w.-]/g, "_")}`;
rmSync(base, { recursive: true, force: true });
mkdirSync(base, { recursive: true });
const tar = Bun.spawnSync(["sh", "-c", `git archive ${rev} | tar -x -C ${base}`]);
if (tar.exitCode !== 0) {
  console.error(`cannot export ${rev}`);
  process.exit(1);
}
if (Bun.spawnSync(["bun", "tools/build-wasm.ts"], { cwd: base }).exitCode !== 0) {
  console.error("baseline build failed");
  process.exit(1);
}

const time = (cwd: string): number => {
  const p = Bun.spawnSync(["bun", "tools/thread-one.ts", model, dims, threadsArg, runs], { cwd });
  const out = new TextDecoder().decode(p.stdout).trim().split("\n").pop() ?? "";
  const v = Number(out);
  if (!Number.isFinite(v)) throw new Error(`bad timing from ${cwd}: ${out}`);
  return v;
};

let bestOld = Infinity;
let bestNew = Infinity;
for (let i = 0; i < rounds; i++) {
  // Alternate which side goes first. A machine whose load is drifting in one
  // direction would otherwise flatter whichever side always ran second.
  const oldFirst = i % 2 === 0;
  const a = oldFirst ? time(base) : time(".");
  const b = oldFirst ? time(".") : time(base);
  const [o, n] = oldFirst ? [a, b] : [b, a];
  bestOld = Math.min(bestOld, o);
  bestNew = Math.min(bestNew, n);
  console.log(`  round ${i + 1}  ${rev} ${o.toFixed(1)}  working tree ${n.toFixed(1)}`);
}
const ratio = bestOld / bestNew;
console.log(
  `${model} ${dims} ${threadsArg}t: ${rev} ${bestOld.toFixed(1)} ms -> ${bestNew.toFixed(1)} ms  ${ratio.toFixed(2)}x`,
);
