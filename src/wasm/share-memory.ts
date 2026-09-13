// Turn a module that defines its own memory into one that imports a shared
// memory, by editing the binary.
//
// Why not just build it that way: rust-lld refuses --shared-memory because the
// precompiled libcore for this target was not built with the atomics feature,
// and rebuilding it needs nightly. We do not actually want atomic instructions
// inside the kernels - every worker writes a disjoint slice and all the
// synchronisation is JS-side Atomics on a control block. All that is missing is
// the memory's shared flag, which lives in one byte of the type.
//
// There is exactly one memory either way, so it stays index 0 and no other
// index space moves. Adding a memory import does not shift function indices.

const SECTION_ORDER_IMPORT = 2;
const SECTION_MEMORY = 5;

type Section = { id: number; body: Uint8Array };

function uleb(n: number): number[] {
  const out: number[] = [];
  while (n > 127) {
    out.push((n & 0x7f) | 0x80);
    n >>>= 7;
  }
  out.push(n);
  return out;
}

function readSections(buf: Uint8Array): { header: Uint8Array; sections: Section[] } {
  const sections: Section[] = [];
  let o = 8;
  while (o < buf.length) {
    const id = buf[o++];
    let len = 0;
    let s = 0;
    while (true) {
      const c = buf[o++];
      len |= (c & 0x7f) << s;
      s += 7;
      if (!(c & 0x80)) break;
    }
    sections.push({ id, body: buf.subarray(o, o + len) });
    o += len;
  }
  return { header: buf.subarray(0, 8), sections };
}

/** Reads the single memory entry's limits out of a memory section body. */
function memoryLimits(body: Uint8Array): { min: number; max: number | null } {
  let o = 0;
  const rd = () => {
    let v = 0;
    let s = 0;
    while (true) {
      const c = body[o++];
      v |= (c & 0x7f) << s;
      s += 7;
      if (!(c & 0x80)) break;
    }
    return v;
  };
  const count = rd();
  if (count !== 1) throw new Error(`expected one memory, found ${count}`);
  const flags = rd();
  const min = rd();
  const max = flags & 1 ? rd() : null;
  return { min, max };
}

export function makeMemoryShared(buf: Uint8Array, maxPages: number): Uint8Array {
  const { header, sections } = readSections(buf);
  const mem = sections.find((s) => s.id === SECTION_MEMORY);
  if (!mem) throw new Error("module has no memory section");
  const { min } = memoryLimits(mem.body);
  if (maxPages < min) throw new Error(`maxPages ${maxPages} below the module's ${min}`);

  const name = (s: string) => {
    const b = [...new TextEncoder().encode(s)];
    return [...uleb(b.length), ...b];
  };
  // limits flags 0x03 = has-maximum | shared. A shared memory must state a max.
  const entry = [...name("env"), ...name("memory"), 0x02, 0x03, ...uleb(min), ...uleb(maxPages)];

  const existing = sections.find((s) => s.id === SECTION_ORDER_IMPORT);
  if (existing) throw new Error("module already has imports; merge not implemented");

  const out: number[] = [...header];
  const emit = (id: number, body: Uint8Array | number[]) => {
    out.push(id, ...uleb(body.length), ...body);
  };

  let inserted = false;
  for (const s of sections) {
    if (s.id === SECTION_MEMORY) continue;
    // Imports must come directly after the type section.
    if (!inserted && s.id !== 1 && s.id !== 0) {
      emit(SECTION_ORDER_IMPORT, [...uleb(1), ...entry]);
      inserted = true;
    }
    emit(s.id, s.body);
    if (s.id === 1) {
      emit(SECTION_ORDER_IMPORT, [...uleb(1), ...entry]);
      inserted = true;
    }
  }
  if (!inserted) emit(SECTION_ORDER_IMPORT, [...uleb(1), ...entry]);
  return new Uint8Array(out);
}

if (import.meta.main) {
  const [src, dst, pages] = process.argv.slice(2);
  const shared = makeMemoryShared(new Uint8Array(await Bun.file(src).arrayBuffer()), Number(pages ?? 32768));
  await Bun.write(dst, shared);
  console.log(`${dst.padEnd(32)} ${shared.length} bytes (shared memory import)`);
}
