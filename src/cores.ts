// How many cores a page really has, measured, for browsers that lie.
//
// Brave reports a random `hardwareConcurrency` between 2 and the true count
// per site, which would leave the worker pool on a fraction of the machine.
// The cores themselves are not hidden: workers still land on all of them.
// So spawn workers and time the same loop on 1, 2, 4 and 8 of them at once;
// summing single-worker-time / each-worker-time gives the throughput in
// core equivalents, and the count stops paying once that stops growing.
// About 30 ms, once; the answer is cached per origin.

const KEY = "tiny6.cores";

const LOOP = `onmessage = () => {
  const t0 = performance.now();
  let x = 1;
  for (let i = 0; i < 4e6; i++) x = (x * 1664525 + 1013904223) >>> 0;
  postMessage([performance.now() - t0, x]);
};`;

export async function measureCores(max = 8): Promise<number> {
  try {
    const hit = localStorage.getItem(KEY);
    if (hit) return Number(hit);
  } catch {}
  const url = URL.createObjectURL(new Blob([LOOP], { type: "text/javascript" }));
  const workers = Array.from({ length: max }, () => new Worker(url));
  const time = (n: number) =>
    Promise.all(workers.slice(0, n).map((w) => new Promise<number>((resolve) => {
      w.onmessage = (e) => resolve((e.data as [number, number])[0]);
      w.postMessage(0);
    })));
  try {
    await time(max); // thread start-up and JIT, not counted
    const alone = Math.min(...(await time(1)));
    let cores = 1;
    let best = 1;
    for (let n = 2; n <= max; n *= 2) {
      const sum = (await time(n)).reduce((s, t) => s + alone / t, 0);
      if (sum < best * 1.15) break;
      best = sum;
      cores = n;
    }
    try { localStorage.setItem(KEY, String(cores)); } catch {}
    return cores;
  } finally {
    for (const w of workers) w.terminate();
    URL.revokeObjectURL(url);
  }
}

/** Brave farbles the reported count; nothing else that ships does. */
export function reportsFakeCores(): boolean {
  return typeof navigator !== "undefined" && "brave" in navigator;
}
