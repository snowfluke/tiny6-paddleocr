// Minimal protobuf wire-format reader. Only what ONNX needs.

export type Field = { no: number; wire: number; start: number; end: number; varint: bigint };

export class Reader {
  constructor(readonly buf: Uint8Array) {}

  /** Walk the top-level fields of the region [start, end). */
  *fields(start: number, end: number): Generator<Field> {
    let i = start;
    while (i < end) {
      let key = 0n;
      let shift = 0n;
      while (true) {
        const c = this.buf[i++];
        key |= BigInt(c & 0x7f) << shift;
        shift += 7n;
        if (!(c & 0x80)) break;
      }
      const no = Number(key >> 3n);
      const wire = Number(key & 7n);
      if (wire === 2) {
        let len = 0;
        let s = 0;
        while (true) {
          const c = this.buf[i++];
          len |= (c & 0x7f) << s;
          s += 7;
          if (!(c & 0x80)) break;
        }
        yield { no, wire, start: i, end: i + len, varint: 0n };
        i += len;
      } else if (wire === 0) {
        let v = 0n;
        let s = 0n;
        const vs = i;
        while (true) {
          const c = this.buf[i++];
          v |= BigInt(c & 0x7f) << s;
          s += 7n;
          if (!(c & 0x80)) break;
        }
        yield { no, wire, start: vs, end: i, varint: v };
      } else if (wire === 5) {
        yield { no, wire, start: i, end: i + 4, varint: 0n };
        i += 4;
      } else if (wire === 1) {
        yield { no, wire, start: i, end: i + 8, varint: 0n };
        i += 8;
      } else {
        throw new Error(`unsupported wire type ${wire} at ${i}`);
      }
    }
  }

  str(f: Field): string {
    return new TextDecoder().decode(this.buf.subarray(f.start, f.end));
  }

  /** int64 fields are plain two's-complement varints, so -1 arrives as 10 bytes. */
  int(f: Field): number {
    return Number(BigInt.asIntN(64, f.varint));
  }

  f32(f: Field): number {
    return new DataView(this.buf.buffer, this.buf.byteOffset + f.start, 4).getFloat32(0, true);
  }

  /** Packed repeated varints inside a length-delimited field. */
  packedInts(f: Field): number[] {
    const out: number[] = [];
    let i = f.start;
    while (i < f.end) {
      let v = 0n;
      let s = 0n;
      while (true) {
        const c = this.buf[i++];
        v |= BigInt(c & 0x7f) << s;
        s += 7n;
        if (!(c & 0x80)) break;
      }
      out.push(Number(BigInt.asIntN(64, v)));
    }
    return out;
  }

  packedF32(f: Field): Float32Array {
    const n = (f.end - f.start) >> 2;
    const out = new Float32Array(n);
    const dv = new DataView(this.buf.buffer, this.buf.byteOffset + f.start, n * 4);
    for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
    return out;
  }

  bytes(f: Field): Uint8Array {
    return this.buf.subarray(f.start, f.end);
  }
}
