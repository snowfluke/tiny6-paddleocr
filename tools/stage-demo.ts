// Copies the models and the sample image into demo/, which is what both the
// dev server and the deploy upload. They are gitignored there: 6 MB of
// binaries that already live in models/.
export async function stageDemo(root = "demo") {
  for (const f of ["models/det.onnx", "models/rec.onnx", "models/dict.txt"]) {
    const dst = `${root}/${f}`;
    if (!(await Bun.file(dst).exists())) await Bun.write(dst, Bun.file(f));
  }
  if (!(await Bun.file(`${root}/receipt.png`).exists())) {
    await Bun.write(`${root}/receipt.png`, Bun.file("test/images/receipt.png"));
  }
}

if (import.meta.main) await stageDemo();
