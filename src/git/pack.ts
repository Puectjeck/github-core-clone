// Packfiles: reading (version 2/3, OFS_DELTA and REF_DELTA, thin-pack bases supplied by the
// caller), writing (whole objects, deflated), and the delta format both directions need —
// applying a delta to its base here; producing deltas lives with the fetch packer, which is the
// only place the forge chooses to compress. The trailer is a SHA-1 over everything before it and
// is verified on read.

import { createHash } from 'node:crypto';
import { deflateSync, inflateSync } from 'node:zlib';
import { objectId, type GitObject, type GitObjectType, type ObjectStore } from './objects.ts';

const TYPE_CODES: Readonly<Record<GitObjectType, number>> = { commit: 1, tree: 2, blob: 3, tag: 4 };
const CODE_TYPES: Readonly<Record<number, GitObjectType>> = { 1: 'commit', 2: 'tree', 3: 'blob', 4: 'tag' };
const OFS_DELTA = 6;
const REF_DELTA = 7;

export class PackError extends Error {}

/** Apply a git delta (source-size varint, target-size varint, then copy/insert ops) to `base`. */
export function applyDelta(base: Buffer, delta: Buffer): Buffer {
  let at = 0;
  const readVarint = (): number => {
    let value = 0;
    let shift = 0;
    for (;;) {
      if (at >= delta.length) throw new PackError('delta size is truncated');
      const byte = delta[at] as number;
      at += 1;
      value += (byte & 0x7f) * 2 ** shift;
      shift += 7;
      if ((byte & 0x80) === 0) return value;
    }
  };
  const sourceSize = readVarint();
  if (sourceSize !== base.length) throw new PackError(`delta base is ${base.length} bytes, delta expects ${sourceSize}`);
  const targetSize = readVarint();
  const out: Buffer[] = [];
  let produced = 0;
  while (at < delta.length) {
    const op = delta[at] as number;
    at += 1;
    if (op & 0x80) {
      let offset = 0;
      let size = 0;
      for (let bit = 0; bit < 4; bit += 1) {
        if (op & (1 << bit)) {
          offset += (delta[at] as number) * 2 ** (8 * bit);
          at += 1;
        }
      }
      for (let bit = 0; bit < 3; bit += 1) {
        if (op & (0x10 << bit)) {
          size += (delta[at] as number) * 2 ** (8 * bit);
          at += 1;
        }
      }
      if (size === 0) size = 0x10000;
      out.push(base.subarray(offset, offset + size));
      produced += size;
    } else if (op > 0) {
      out.push(delta.subarray(at, at + op));
      produced += op;
      at += op;
    } else {
      throw new PackError('delta opcode 0 is reserved');
    }
  }
  const result = Buffer.concat(out);
  if (produced !== targetSize || result.length !== targetSize) {
    throw new PackError(`delta produced ${result.length} bytes, expected ${targetSize}`);
  }
  return result;
}

interface PackEntry {
  readonly offset: number;
  readonly kind: GitObjectType | 'ofs-delta' | 'ref-delta';
  readonly size: number;
  readonly data: Buffer;
  readonly baseOffset?: number;
  readonly baseId?: string;
}

/** Inflate one zlib stream at `offset`; returns its bytes and the offset just past the stream. */
function inflateAt(buffer: Buffer, offset: number): { data: Buffer; next: number } {
  const view = buffer.subarray(offset);
  const result = inflateSync(view, { info: true }) as unknown as { buffer: Buffer; engine: { bytesWritten: number } };
  const written = result.engine?.bytesWritten ?? 0;
  if (!written || written <= 0) throw new PackError(`zlib stream at offset ${offset} consumed no bytes`);
  return { data: result.buffer, next: offset + written };
}

/**
 * Read a whole packfile into the store. Objects whose delta base is neither in the pack nor in
 * the store are an error unless a base is supplied through the store first (the caller loads its
 * own objects for a thin pack before calling). Returns the ids written, in pack order.
 */
export function readPack(store: ObjectStore, pack: Buffer): string[] {
  if (pack.length < 12 + 20) throw new PackError('pack is too short');
  if (pack.toString('ascii', 0, 4) !== 'PACK') throw new PackError('pack does not start with PACK');
  const version = pack.readUInt32BE(4);
  if (version !== 2 && version !== 3) throw new PackError(`pack version ${version} is not supported`);
  const count = pack.readUInt32BE(8);
  const trailerStart = pack.length - 20;
  const expected = createHash('sha1').update(pack.subarray(0, trailerStart)).digest();
  if (!expected.equals(pack.subarray(trailerStart))) throw new PackError('pack trailer checksum does not match');

  const entries: PackEntry[] = [];
  let offset = 12;
  for (let i = 0; i < count; i += 1) {
    const entryOffset = offset;
    let byte = pack[offset] as number;
    offset += 1;
    const code = (byte >> 4) & 0x7;
    let size = byte & 0x0f;
    let shift = 4;
    while (byte & 0x80) {
      byte = pack[offset] as number;
      offset += 1;
      size += (byte & 0x7f) * 2 ** shift;
      shift += 7;
    }
    let baseOffset: number | undefined;
    let baseId: string | undefined;
    if (code === OFS_DELTA) {
      let b = pack[offset] as number;
      offset += 1;
      let back = b & 0x7f;
      while (b & 0x80) {
        b = pack[offset] as number;
        offset += 1;
        back = ((back + 1) << 7) + (b & 0x7f);
      }
      baseOffset = entryOffset - back;
    } else if (code === REF_DELTA) {
      baseId = pack.toString('hex', offset, offset + 20);
      offset += 20;
    }
    const inflated = inflateAt(pack, offset);
    offset = inflated.next;
    const kind = code === OFS_DELTA ? 'ofs-delta' : code === REF_DELTA ? 'ref-delta' : CODE_TYPES[code];
    if (kind === undefined) throw new PackError(`pack entry at ${entryOffset} has unknown type code ${code}`);
    entries.push({ offset: entryOffset, kind, size, data: inflated.data, baseOffset, baseId });
  }
  if (offset !== trailerStart) throw new PackError(`pack entries end at ${offset}, trailer starts at ${trailerStart}`);

  const byOffset = new Map<number, PackEntry>();
  for (const entry of entries) byOffset.set(entry.offset, entry);
  const resolved = new Map<number, { id: string; type: GitObjectType; data: Buffer }>();

  const resolve = (entry: PackEntry): { id: string; type: GitObjectType; data: Buffer } => {
    const cached = resolved.get(entry.offset);
    if (cached) return cached;
    let out: { id: string; type: GitObjectType; data: Buffer };
    if (entry.kind === 'ofs-delta' || entry.kind === 'ref-delta') {
      let base: { type: GitObjectType; data: Buffer };
      if (entry.kind === 'ofs-delta') {
        const baseEntry = byOffset.get(entry.baseOffset as number);
        if (!baseEntry) throw new PackError(`ofs-delta at ${entry.offset} names a base outside the pack`);
        const r = resolve(baseEntry);
        base = { type: r.type, data: r.data };
      } else {
        const inPack = entries.find((e) => e.kind !== 'ofs-delta' && e.kind !== 'ref-delta' && objectId(e.kind as GitObjectType, e.data) === entry.baseId);
        if (inPack) {
          const r = resolve(inPack);
          base = { type: r.type, data: r.data };
        } else {
          const stored = store.get(entry.baseId as string);
          if (!stored) throw new PackError(`ref-delta base ${entry.baseId} is neither in the pack nor in the store`);
          base = stored;
        }
      }
      const data = applyDelta(base.data, entry.data);
      out = { id: objectId(base.type, data), type: base.type, data };
    } else {
      if (entry.data.length !== entry.size) {
        throw new PackError(`pack entry at ${entry.offset} declares ${entry.size} bytes but inflated to ${entry.data.length}`);
      }
      out = { id: objectId(entry.kind, entry.data), type: entry.kind, data: entry.data };
    }
    resolved.set(entry.offset, out);
    return out;
  };

  const ids: string[] = [];
  for (const entry of entries) {
    const r = resolve(entry);
    if (!store.objects.has(r.id)) store.objects.set(r.id, { type: r.type, data: r.data });
    ids.push(r.id);
  }
  return ids;
}

/** Encode one entry's type+size header. */
function entryHeader(code: number, size: number): Buffer {
  const bytes: number[] = [];
  let first = (code << 4) | (size & 0x0f);
  size = Math.floor(size / 16);
  if (size > 0) first |= 0x80;
  bytes.push(first);
  while (size > 0) {
    let byte = size & 0x7f;
    size = Math.floor(size / 128);
    if (size > 0) byte |= 0x80;
    bytes.push(byte);
  }
  return Buffer.from(bytes);
}

/** Write a packfile holding the given objects whole (no delta compression). */
export function writePack(objects: readonly GitObject[]): Buffer {
  const parts: Buffer[] = [];
  const head = Buffer.alloc(12);
  head.write('PACK', 0, 'ascii');
  head.writeUInt32BE(2, 4);
  head.writeUInt32BE(objects.length, 8);
  parts.push(head);
  for (const object of objects) {
    parts.push(entryHeader(TYPE_CODES[object.type], object.data.length));
    parts.push(deflateSync(object.data));
  }
  const body = Buffer.concat(parts);
  const trailer = createHash('sha1').update(body).digest();
  return Buffer.concat([body, trailer]);
}
