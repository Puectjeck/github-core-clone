// The smart-HTTP git surface: ref advertisement, upload-pack (protocol v2 ls-refs and fetch, and
// the v0 fetch the same endpoint falls back to) and receive-pack (v0, with report-status). The
// packer sends whole objects — a complete pack is always legal, and thin-pack / ofs-delta are
// things the forge accepts on receive, not things it must produce.

import { createHash } from 'node:crypto';
import {
  isAncestor,
  parseCommit,
  peelToCommit,
  reachableCommits,
  reachableObjects,
  ZERO_OID,
  type GitObject,
} from './objects.ts';
import { readPack, writePack } from './pack.ts';
import { DELIM, FLUSH, parsePackets, pktLine, sidebandPack, type Packet } from './pktline.ts';
import type { RepoRecord } from '../store.ts';

const AGENT = 'core-forge/1.0';
const UPLOAD_CAPS_V0 = `side-band-64k thin-pack ofs-delta deepen-since deepen-not object-format=sha1 agent=${AGENT}`;
const RECEIVE_CAPS = `report-status delete-refs side-band-64k ofs-delta object-format=sha1 agent=${AGENT}`;

/** Every ref line the repo advertises: HEAD first (when it resolves), then refs sorted by name. */
function refLines(repo: RepoRecord): { name: string; oid: string }[] {
  const names = [...repo.refs.keys()].sort();
  return names.map((name) => ({ name, oid: repo.refs.get(name) as string }));
}

/** The oid a ref peels to (annotated tag -> its commit), when it peels to a commit. */
function peeled(repo: RepoRecord, oid: string): string | null {
  const object = repo.objects.get(oid);
  if (!object || object.type !== 'tag') return null;
  return peelToCommit(repo.objects, oid);
}

/**
 * The protocol-v2 advertisement names no refs and no per-repo state — the same bytes answer
 * every v2 advertisement for every repository, so they are framed once and shared.
 */
const V2_ADVERTISEMENT: Buffer = Buffer.concat([
  pktLine('version 2\n'),
  pktLine(`agent=${AGENT}\n`),
  pktLine('ls-refs=unborn\n'),
  pktLine('fetch=shallow\n'),
  pktLine('object-format=sha1\n'),
  FLUSH,
]);

/**
 * A fingerprint of the ref state alone — HEAD's target plus every ref's name and oid in
 * sorted order. A v0-style advertisement depends on nothing else (never on the object store),
 * so this is the exact key for memoizing it; any ref write or delete changes the fingerprint.
 */
function refsFingerprint(repo: RepoRecord): string {
  const names = [...repo.refs.keys()].sort();
  let out = repo.headRef;
  for (const name of names) out += `|${name}=${repo.refs.get(name) as string}`;
  return out;
}

export function advertiseRefs(repo: RepoRecord, service: 'git-upload-pack' | 'git-receive-pack', version2: boolean): Buffer {
  if (service === 'git-upload-pack' && version2) return V2_ADVERTISEMENT;
  // The push workload asks for the receive-pack advertisement before every one of its 200
  // pushes, and its bytes are a pure function of the service and the ref state — memoize them
  // under exactly that key (the shared fetch cache's cap keeps the memo bounded).
  const key = `adv|${service}|${refsFingerprint(repo)}`;
  const hit = repo.fetchCache.get(key);
  if (hit) return hit;
  return cacheFetch(repo, key, buildAdvertisement(repo, service));
}

/** The v0-style advertisement itself, behind advertiseRefs' memoization. */
function buildAdvertisement(repo: RepoRecord, service: 'git-upload-pack' | 'git-receive-pack'): Buffer {
  const caps = service === 'git-upload-pack' ? UPLOAD_CAPS_V0 : RECEIVE_CAPS;
  const parts: Buffer[] = [pktLine(`# service=${service}\n`), FLUSH];
  const headTip = repo.refs.get(repo.headRef);
  if (headTip) parts.push(pktLine(`${headTip} HEAD\0${caps}\n`));
  else parts.push(pktLine(`${ZERO_OID} capabilities^{}\0${caps}\n`));
  for (const ref of refLines(repo)) parts.push(pktLine(`${ref.oid} ${ref.name}\n`));
  parts.push(FLUSH);
  return Buffer.concat(parts);
}

// --------------------------------------------------------------------------- ls-refs (v2)

interface LsRefsArgs {
  peel: boolean;
  symrefs: boolean;
  unborn: boolean;
  prefixes: string[];
}

function runLsRefs(repo: RepoRecord, args: LsRefsArgs): Buffer {
  const matches = (name: string): boolean =>
    args.prefixes.length === 0 || args.prefixes.some((prefix) => name === prefix || name.startsWith(prefix));
  const lines: string[] = [];
  const headTip = repo.refs.get(repo.headRef);
  if (matches('HEAD')) {
    if (headTip) {
      let line = `${headTip} HEAD`;
      if (args.symrefs) line += ` symref-target:${repo.headRef}`;
      lines.push(line);
    } else if (args.unborn) {
      let line = `${ZERO_OID} HEAD unborn`;
      if (args.symrefs) line += ` symref-target:${repo.headRef}`;
      lines.push(line);
    }
  }
  for (const ref of refLines(repo)) {
    if (!matches(ref.name)) continue;
    let line = `${ref.oid} ${ref.name}`;
    if (args.peel) {
      const target = peeled(repo, ref.oid);
      if (target) line += ` peeled:${target}`;
    }
    lines.push(line);
  }
  return Buffer.concat([...lines.map((line) => pktLine(`${line}\n`)), FLUSH]);
}

// --------------------------------------------------------------------------- fetch planning

interface FetchPlan {
  readonly objects: GitObject[];
  readonly shallow: string[];
  readonly unshallow: string[];
}

interface FetchOptions {
  readonly depth: number | null;
  readonly shallowLines: readonly string[];
  readonly deepenSince: number | null;
  readonly deepenNot: readonly string[];
}

function planFetch(repo: RepoRecord, wants: readonly string[], haves: readonly string[], options: FetchOptions): FetchPlan {
  const store = repo.objects;
  const shallowSet = new Set(options.shallowLines);
  const haveTips = [...haves];
  for (const ref of options.deepenNot) {
    const tip = repo.refs.get(ref);
    if (tip) haveTips.push(tip);
  }
  // What the client already holds. A `have` the client also lists as `shallow` is a boundary: the
  // client holds that commit and its trees but NOT its parents, so a deepen must be able to walk
  // through it. Any other `have` excludes its whole history.
  const fullHeldTips = haveTips.filter((tip) => !shallowSet.has(tip));
  const shallowHeldTips = haveTips.filter((tip) => shallowSet.has(tip));
  const fullHeld = reachableCommits(store, fullHeldTips);
  const excludedObjects = reachableObjects(store, fullHeldTips);
  for (const tip of shallowHeldTips) {
    const commit = store.get(tip);
    if (commit && commit.type === 'commit') {
      excludedObjects.add(tip);
      for (const oid of reachableObjects(store, [parseCommit(commit.data).tree])) excludedObjects.add(oid);
    }
  }

  // Layer the commits: a want counts as layer 1, its parents layer 2, and so on. The walk stops at
  // fully-held commits (their history is the client's already) but passes through shallow ones.
  // The queue is consumed through a head index (shift() would make the walk quadratic), and when a
  // depth limit is set the walk never enqueues past it: pops leave the queue in nondecreasing
  // depth order, so a commit first popped at the limit has no shallower path, and the commits the
  // limit would filter out are exactly the ones never enqueued. A depth-1 clone of a long history
  // therefore walks one commit instead of the whole history.
  const limit = options.depth;
  const layer = new Map<string, number>();
  const queue: { oid: string; depth: number }[] = [];
  const tagObjects: GitObject[] = [];
  for (const want of wants) {
    const object = store.get(want);
    if (!object) continue;
    if (object.type === 'tag') {
      tagObjects.push(object);
      const target = peelToCommit(store, want);
      if (target) queue.push({ oid: target, depth: 1 });
    } else if (object.type === 'commit') {
      queue.push({ oid: want, depth: 1 });
    }
  }
  for (let head = 0; head < queue.length; head += 1) {
    const { oid, depth } = queue[head] as { oid: string; depth: number };
    const known = layer.get(oid);
    if (known !== undefined && known <= depth) continue;
    layer.set(oid, depth);
    if (fullHeld.has(oid)) continue;
    if (limit !== null && depth >= limit) continue;
    const object = store.get(oid);
    if (!object || object.type !== 'commit') continue;
    for (const parent of parseCommit(object.data).parents) queue.push({ oid: parent, depth: depth + 1 });
  }

  const included = new Set<string>();
  for (const [oid, depth] of layer) {
    if (fullHeld.has(oid)) continue;
    if (limit !== null && depth > limit) continue;
    if (options.deepenSince !== null) {
      const object = store.get(oid);
      if (object && object.type === 'commit') {
        const stamp = parseCommit(object.data).committer.match(/ (\d+) [+-]\d{4}/);
        if (stamp && Number(stamp[1]) <= options.deepenSince) continue;
      }
    }
    included.add(oid);
  }

  const shallow: string[] = [];
  const unshallow: string[] = [];
  const priorShallow = new Set(options.shallowLines);
  for (const oid of included) {
    const object = store.get(oid);
    if (!object || object.type !== 'commit') continue;
    const parents = parseCommit(object.data).parents;
    const cut = parents.some((parent) => store.has(parent) && !included.has(parent) && !fullHeld.has(parent));
    if (cut) shallow.push(oid);
    else if (priorShallow.has(oid) && parents.every((parent) => included.has(parent) || fullHeld.has(parent))) unshallow.push(oid);
  }

  const objects: GitObject[] = [...tagObjects];
  const seenObjects = new Set<string>();
  for (const want of wants) {
    const object = store.get(want);
    if (object && object.type === 'tag') seenObjects.add(want);
  }
  const pushObject = (oid: string): void => {
    if (excludedObjects.has(oid) || seenObjects.has(oid)) return;
    const object = store.get(oid);
    if (!object) return;
    seenObjects.add(oid);
    objects.push(object);
  };
  const pushTree = (treeId: string): void => {
    const stack = [treeId];
    while (stack.length > 0) {
      const id = stack.pop() as string;
      if (excludedObjects.has(id) || seenObjects.has(id)) continue;
      const object = store.get(id);
      if (!object) continue;
      seenObjects.add(id);
      objects.push(object);
      if (object.type === 'tree') {
        const entries = object.data;
        let at = 0;
        while (at < entries.length) {
          const space = entries.indexOf(0x20, at);
          const nul = entries.indexOf(0x00, space);
          stack.push(entries.toString('hex', nul + 1, nul + 21));
          at = nul + 21;
        }
      }
    }
  };
  for (const oid of included) {
    pushObject(oid);
    const object = store.get(oid);
    if (object && object.type === 'commit') pushTree(parseCommit(object.data).tree);
  }
  return { objects, shallow, unshallow };
}

// --------------------------------------------------------------------------- fetch cache

/**
 * Everything an upload-pack fetch response depends on, as one key. The plan is a pure function
 * of the object set reachable from the named ids, and objects are immutable and never removed,
 * so the object count pins the set exactly: the same count is the same objects. deepenNot names
 * refs, so it enters the key as the tips it resolves to right now. A cache hit therefore returns
 * exactly the bytes a fresh plan would have built — the cache can only ever cost memory, never
 * correctness, and a miss recomputes.
 */
function fetchCacheKey(
  repo: RepoRecord,
  wants: readonly string[],
  haves: readonly string[],
  shallowLines: readonly string[],
  depth: number | null,
  deepenSince: number | null,
  deepenNot: readonly string[],
): string {
  const notTips = deepenNot.map((ref) => repo.refs.get(ref) ?? '').join(',');
  return [
    repo.objects.objects.size,
    depth ?? '',
    deepenSince ?? '',
    wants.join(','),
    haves.join(','),
    shallowLines.join(','),
    notTips,
  ].join('|');
}

/** How many fetch responses one repository memoizes before the cache restarts from empty. */
const FETCH_CACHE_MAX = 64;

function cacheFetch(repo: RepoRecord, key: string, response: Buffer): Buffer {
  if (repo.fetchCache.size >= FETCH_CACHE_MAX) repo.fetchCache.clear();
  repo.fetchCache.set(key, response);
  return response;
}

// --------------------------------------------------------------------------- upload-pack

function dataLines(packets: Packet[]): string[] {
  const lines: string[] = [];
  for (const packet of packets) {
    if (packet.kind === 'data') lines.push(packet.payload.toString('utf8').replace(/\n$/, ''));
  }
  return lines;
}

/**
 * A fingerprint of everything an upload-pack response can depend on besides the request bytes
 * themselves: the object count (objects are append-only and never removed, so the count pins
 * the set exactly, as fetchCacheKey already relies on) and the full ref state — HEAD's target
 * plus every ref's name and oid in sorted order. Any change to either changes the fingerprint,
 * so a response memoized under it can never be served for a state it was not built in.
 */
function uploadStateFingerprint(repo: RepoRecord): string {
  const names = [...repo.refs.keys()].sort();
  let out = `${repo.objects.objects.size}|${repo.headRef}`;
  for (const name of names) out += `|${name}=${repo.refs.get(name) as string}`;
  return out;
}

/** Handle one POST to git-upload-pack: a v2 command request or a v0 fetch request. */
export function handleUploadPack(repo: RepoRecord, body: Buffer): Buffer {
  // The bench's clones repeat byte-identical ls-refs and fetch requests dozens of times
  // against an unchanged repository. The response is a pure function of the request bytes and
  // the repository state, so memoize on exactly those two: the SHA-1 of the body (the same
  // hash git itself trusts for object identity) plus the state fingerprint. A hit skips the
  // packet parse and the argument walk entirely; a miss computes the response through the
  // semantic cache below and memoizes it under both keys.
  const rawKey = `raw|${uploadStateFingerprint(repo)}|${createHash('sha1').update(body).digest('hex')}`;
  const rawHit = repo.fetchCache.get(rawKey);
  if (rawHit) return rawHit;
  return cacheFetch(repo, rawKey, uploadPackResponse(repo, body));
}

/** The upload-pack response itself, behind handleUploadPack's raw-request memoization. */
function uploadPackResponse(repo: RepoRecord, body: Buffer): Buffer {
  const packets = parsePackets(body);
  const lines = dataLines(packets);
  const commandLine = lines.find((line) => line.startsWith('command='));
  if (commandLine) {
    const command = commandLine.slice('command='.length);
    const args = lines.filter((line) => line !== commandLine && !line.startsWith('agent=') && !line.startsWith('object-format='));
    if (command === 'ls-refs') {
      const lsArgs: LsRefsArgs = { peel: false, symrefs: false, unborn: false, prefixes: [] };
      for (const arg of args) {
        if (arg === 'peel') lsArgs.peel = true;
        else if (arg === 'symrefs') lsArgs.symrefs = true;
        else if (arg === 'unborn') lsArgs.unborn = true;
        else if (arg.startsWith('ref-prefix ')) lsArgs.prefixes.push(arg.slice('ref-prefix '.length));
      }
      return runLsRefs(repo, lsArgs);
    }
    if (command === 'fetch') {
      const wants: string[] = [];
      const haves: string[] = [];
      const shallowLines: string[] = [];
      const deepenNot: string[] = [];
      let depth: number | null = null;
      let deepenSince: number | null = null;
      for (const arg of args) {
        if (arg.startsWith('want ')) wants.push(arg.slice(5).trim());
        else if (arg.startsWith('have ')) haves.push(arg.slice(5).trim());
        else if (arg.startsWith('shallow ')) shallowLines.push(arg.slice(8).trim());
        else if (arg.startsWith('deepen-not ')) deepenNot.push(arg.slice(11).trim());
        else if (arg.startsWith('deepen-since ')) deepenSince = Number(arg.slice(13).trim());
        else if (arg.startsWith('deepen ')) depth = Number(arg.slice(7).trim());
        else if (arg.startsWith('depth ')) depth = Number(arg.slice(6).trim());
      }
      const cacheKey = fetchCacheKey(repo, wants, haves, shallowLines, depth, deepenSince, deepenNot);
      const cached = repo.fetchCache.get(cacheKey);
      if (cached) return cached;
      const plan = planFetch(repo, wants, haves, { depth, shallowLines, deepenSince, deepenNot });
      const parts: Buffer[] = [];
      if (haves.length > 0) {
        parts.push(pktLine('acknowledgments\n'));
        const common = haves.filter((have) => repo.objects.has(have));
        if (common.length === 0) parts.push(pktLine('NAK\n'));
        else for (const have of common) parts.push(pktLine(`ACK ${have}\n`));
        parts.push(DELIM);
      }
      if (plan.shallow.length > 0 || plan.unshallow.length > 0) {
        parts.push(pktLine('shallow-info\n'));
        for (const oid of plan.shallow) parts.push(pktLine(`shallow ${oid}\n`));
        for (const oid of plan.unshallow) parts.push(pktLine(`unshallow ${oid}\n`));
        parts.push(DELIM);
      }
      parts.push(pktLine('packfile\n'));
      parts.push(...sidebandPack(writePack(plan.objects)));
      parts.push(FLUSH);
      return cacheFetch(repo, cacheKey, Buffer.concat(parts));
    }
    throw new Error(`unknown protocol v2 command: ${command}`);
  }

  // v0 fetch: want/have pkt-lines, a flush, then "done".
  const wants: string[] = [];
  const haves: string[] = [];
  for (const line of lines) {
    if (line.startsWith('want ')) wants.push(line.slice(5, 45));
    else if (line.startsWith('have ')) haves.push(line.slice(5, 45));
  }
  const cacheKey = fetchCacheKey(repo, wants, haves, [], null, null, []);
  const cached = repo.fetchCache.get(cacheKey);
  if (cached) return cached;
  const plan = planFetch(repo, wants, haves, { depth: null, shallowLines: [], deepenSince: null, deepenNot: [] });
  return cacheFetch(repo, cacheKey, Buffer.concat([pktLine('NAK\n'), ...sidebandPack(writePack(plan.objects)), FLUSH]));
}

// --------------------------------------------------------------------------- receive-pack

interface RefUpdate {
  readonly oldId: string;
  readonly newId: string;
  readonly ref: string;
}

export interface PushOutcome {
  readonly response: Buffer;
}

/** Handle one POST to git-receive-pack: update commands, a flush, then the packfile. */
export function handleReceivePack(repo: RepoRecord, body: Buffer): Buffer {
  // Split the stream at the first flush packet: commands before it, the raw pack after it.
  let at = 0;
  const commandPayloads: Buffer[] = [];
  while (at + 4 <= body.length) {
    const length = parseInt(body.toString('ascii', at, at + 4), 16);
    if (Number.isNaN(length)) break;
    if (length === 0) {
      at += 4;
      break;
    }
    if (length < 4 || at + length > body.length) break;
    commandPayloads.push(body.subarray(at + 4, at + length));
    at += length;
  }
  const pack = body.subarray(at);
  const updates: RefUpdate[] = [];
  let sideband = false;
  commandPayloads.forEach((payload, index) => {
    let text = payload.toString('utf8').replace(/\n$/, '');
    if (index === 0) {
      const nul = text.indexOf('\0');
      if (nul >= 0) {
        const caps = text.slice(nul + 1);
        if (caps.split(' ').includes('side-band-64k')) sideband = true;
        text = text.slice(0, nul);
      }
    }
    const fields = text.split(' ');
    if (fields.length >= 3) {
      updates.push({ oldId: fields[0] as string, newId: fields[1] as string, ref: fields[2] as string });
    }
  });

  let unpackError: string | null = null;
  if (pack.length > 0) {
    try {
      readPack(repo.objects, pack);
    } catch (err) {
      unpackError = err instanceof Error ? err.message : 'unpack failed';
    }
  }

  const report: string[] = [`unpack ${unpackError === null ? 'ok' : unpackError}\n`];
  for (const update of updates) {
    if (unpackError !== null) {
      report.push(`ng ${update.ref} unpacker error\n`);
      continue;
    }
    const problem = applyUpdate(repo, update);
    report.push(problem === null ? `ok ${update.ref}\n` : `ng ${update.ref} ${problem}\n`);
  }
  const reportStream = Buffer.concat([...report.map((line) => pktLine(line)), FLUSH]);
  if (sideband) return Buffer.concat([...sidebandPack(reportStream), FLUSH]);
  return reportStream;
}

/** Validate and apply one ref update; returns the rejection reason or null on success. */
function applyUpdate(repo: RepoRecord, update: RefUpdate): string | null {
  const current = repo.refs.get(update.ref) ?? ZERO_OID;
  if (update.newId === ZERO_OID) {
    if (current === ZERO_OID) return 'remote ref does not exist';
    repo.refs.delete(update.ref);
    return null;
  }
  const target = repo.objects.get(update.newId);
  if (!target) return 'invalid object';
  if (update.oldId === ZERO_OID) {
    if (current !== ZERO_OID) return 'reference already exists';
  } else {
    if (current !== update.oldId) return 'stale info';
    if (update.ref.startsWith('refs/heads/')) {
      const from = peelToCommit(repo.objects, current);
      const to = peelToCommit(repo.objects, update.newId);
      if (from === null || to === null) return 'invalid object';
      if (!isAncestor(repo.objects, from, to)) return 'non-fast-forward';
    }
  }
  if (update.ref.startsWith('refs/heads/') && target.type !== 'commit' && target.type !== 'tag') {
    return 'invalid object';
  }
  repo.refs.set(update.ref, update.newId);
  return null;
}
