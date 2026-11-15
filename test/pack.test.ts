// Local unit tests for the git foundation: object ids against git's known vectors, tree ordering,
// pack write/read round-trips, and delta application — including a hand-built pack whose second
// entry is an OFS_DELTA against the first, the shape a real client's push can take.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { deflateSync } from 'node:zlib';
import {
  ObjectStore,
  objectId,
  parseTree,
  serializeTree,
  type TreeEntry,
} from '../src/git/objects.ts';
import { applyDelta, readPack, writePack } from '../src/git/pack.ts';
import { viewerUser, strangerUser } from '../src/scenario.ts';

test('object id matches git for the classic blob', () => {
  // `printf 'hello\n' | git hash-object --stdin` in any real git.
  assert.equal(objectId('blob', Buffer.from('hello\n')), 'ce013625030ba8dba906f756967f9e9ca394464a');
});

test('tree serialization round-trips and sorts directories as name-plus-slash', () => {
  const entries: TreeEntry[] = [
    { mode: '40000', name: 'foo', oid: '11'.repeat(20) },
    { mode: '100644', name: 'foo.txt', oid: '22'.repeat(20) },
    { mode: '100644', name: 'bar', oid: '33'.repeat(20) },
  ];
  const parsed = parseTree(serializeTree(entries));
  assert.deepEqual(parsed.map((entry) => entry.name), ['bar', 'foo.txt', 'foo']);
  assert.deepEqual(parsed[2], entries[0]);
});

test('pack write then read returns the same objects', () => {
  const store = new ObjectStore();
  const blobA = store.put('blob', Buffer.from('alpha\n'));
  const blobB = store.put('blob', Buffer.from('beta\n'));
  const tree = store.put('tree', serializeTree([
    { mode: '100644', name: 'a.txt', oid: blobA },
    { mode: '100644', name: 'b.txt', oid: blobB },
  ]));
  const objects = [store.get(blobA)!, store.get(blobB)!, store.get(tree)!];
  const pack = writePack(objects);
  const fresh = new ObjectStore();
  const ids = readPack(fresh, pack);
  assert.deepEqual(new Set(ids), new Set([blobA, blobB, tree]));
  assert.equal(fresh.get(tree)?.type, 'tree');
  assert.equal(fresh.get(blobB)?.data.toString(), 'beta\n');
});

test('applyDelta copies and inserts', () => {
  const base = Buffer.from('the quick brown fox jumps over the lazy dog');
  // Build a delta by hand: copy base[0..20], insert 'RED ', copy base[20..43].
  const varint = (value: number): Buffer => {
    const bytes: number[] = [];
    let rest = value;
    for (;;) {
      const byte = rest & 0x7f;
      rest = Math.floor(rest / 128);
      if (rest === 0) {
        bytes.push(byte);
        return Buffer.from(bytes);
      }
      bytes.push(byte | 0x80);
    }
  };
  const copy = (offset: number, size: number): Buffer => {
    let op = 0x80;
    const extra: number[] = [];
    for (let bit = 0; bit < 4; bit += 1) {
      const byte = Math.floor(offset / 2 ** (8 * bit)) & 0xff;
      if (byte !== 0) {
        op |= 1 << bit;
        extra.push(byte);
      }
    }
    for (let bit = 0; bit < 3; bit += 1) {
      const byte = Math.floor(size / 2 ** (8 * bit)) & 0xff;
      if (byte !== 0) {
        op |= 0x10 << bit;
        extra.push(byte);
      }
    }
    return Buffer.from([op, ...extra]);
  };
  const insert = Buffer.from('RED ');
  const delta = Buffer.concat([
    varint(base.length),
    varint(20 + insert.length + 23),
    copy(0, 20),
    Buffer.from([insert.length]),
    insert,
    copy(20, 23),
  ]);
  assert.equal(applyDelta(base, delta).toString(), 'the quick brown fox RED jumps over the lazy dog');
});

test('readPack resolves an OFS_DELTA entry against its in-pack base', () => {
  const baseData = Buffer.from('a'.repeat(100));
  const targetData = Buffer.from(`${'a'.repeat(42)}${'Z'.repeat(8)}${'a'.repeat(58)}`);
  // Delta: copy 42 bytes, insert 'ZZZZZZZZ', copy the remaining 58.
  const varint = (value: number): Buffer => {
    const bytes: number[] = [];
    let rest = value;
    for (;;) {
      const byte = rest & 0x7f;
      rest = Math.floor(rest / 128);
      if (rest === 0) {
        bytes.push(byte);
        return Buffer.from(bytes);
      }
      bytes.push(byte | 0x80);
    }
  };
  const copy = (offset: number, size: number): Buffer => {
    let op = 0x80;
    const extra: number[] = [];
    for (let bit = 0; bit < 4; bit += 1) {
      const byte = Math.floor(offset / 2 ** (8 * bit)) & 0xff;
      if (byte !== 0) {
        op |= 1 << bit;
        extra.push(byte);
      }
    }
    for (let bit = 0; bit < 3; bit += 1) {
      const byte = Math.floor(size / 2 ** (8 * bit)) & 0xff;
      if (byte !== 0) {
        op |= 0x10 << bit;
        extra.push(byte);
      }
    }
    return Buffer.from([op, ...extra]);
  };
  const delta = Buffer.concat([
    varint(baseData.length),
    varint(targetData.length),
    copy(0, 42),
    Buffer.from([8]),
    Buffer.from('ZZZZZZZZ'),
    copy(42, 58),
  ]);
  const header = (code: number, size: number): Buffer => {
    const bytes: number[] = [];
    let first = (code << 4) | (size & 0x0f);
    let rest = Math.floor(size / 16);
    if (rest > 0) first |= 0x80;
    bytes.push(first);
    while (rest > 0) {
      let byte = rest & 0x7f;
      rest = Math.floor(rest / 128);
      if (rest > 0) byte |= 0x80;
      bytes.push(byte);
    }
    return Buffer.from(bytes);
  };
  const head = Buffer.alloc(12);
  head.write('PACK', 0, 'ascii');
  head.writeUInt32BE(2, 4);
  head.writeUInt32BE(2, 8);
  const baseEntry = Buffer.concat([header(3, baseData.length), deflateSync(baseData)]);
  const deltaEntryStart = 12 + baseEntry.length;
  // OFS negative offset: distance from the delta entry's start back to the base entry's start.
  const distance = deltaEntryStart - 12;
  const offsetBytes: number[] = [];
  {
    let rest = distance;
    const first = rest & 0x7f;
    rest = Math.floor(rest / 128);
    if (rest === 0) {
      offsetBytes.push(first);
    } else {
      const parts = [first];
      while (rest > 0) {
        rest -= 1;
        parts.unshift(rest & 0x7f);
        rest = Math.floor(rest / 128);
      }
      for (let i = 0; i < parts.length - 1; i += 1) parts[i] = (parts[i] as number) | 0x80;
      offsetBytes.push(...parts);
    }
  }
  const deltaEntry = Buffer.concat([header(6, delta.length), Buffer.from(offsetBytes), deflateSync(delta)]);
  const body = Buffer.concat([head, baseEntry, deltaEntry]);
  const trailer = createHash('sha1').update(body).digest();
  const pack = Buffer.concat([body, trailer]);

  const store = new ObjectStore();
  const ids = readPack(store, pack);
  const baseId = objectId('blob', baseData);
  const targetId = objectId('blob', targetData);
  assert.deepEqual(new Set(ids), new Set([baseId, targetId]));
  assert.equal(store.get(targetId)?.data.toString(), targetData.toString());
});

test('scenario derivation names the suite viewer and stranger', () => {
  assert.equal(viewerUser().login, 'adaiversen');
  assert.equal(viewerUser().id, 1000);
  assert.equal(strangerUser().id, 1002);
});
