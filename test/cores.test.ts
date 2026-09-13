import { expect, test } from "bun:test";
import { measureCores } from "../src/cores.ts";

test("measured core count is sane", async () => {
  const cores = await measureCores();
  const reported = navigator.hardwareConcurrency ?? 8;
  expect(cores).toBeGreaterThanOrEqual(1);
  expect(cores).toBeLessThanOrEqual(8);
  // Half the machine at least; efficiency cores may count as less than one.
  expect(cores * 2).toBeGreaterThanOrEqual(Math.min(8, reported));
  console.log(`measured ${cores} cores, reported ${reported}`);
}, 20_000);
