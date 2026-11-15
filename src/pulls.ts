// Pull-request machinery shared by REST, GraphQL and MCP: flattening trees to path maps, finding
// a merge base, the three-way tree merge behind a squash merge, and the file diff a pull request
// reports. A merge conflicts exactly when both sides changed the same path away from the merge
// base in different ways — the rule the suite's conflict check exercises.

import { parseCommit, parseTree, serializeTree, type ObjectStore, type TreeEntry } from './git/objects.ts';

export interface FlatEntry {
  readonly mode: string;
  readonly oid: string;
}

/** Flatten a tree into path -> entry, recursing into directories. */
export function flattenTree(store: ObjectStore, treeId: string | null): Map<string, FlatEntry> {
  const out = new Map<string, FlatEntry>();
  const walk = (id: string, prefix: string): void => {
    const object = store.get(id);
    if (!object || object.type !== 'tree') return;
    for (const entry of parseTree(object.data)) {
      const path = prefix === '' ? entry.name : `${prefix}/${entry.name}`;
      if (entry.mode === '40000') walk(entry.oid, path);
      else out.set(path, { mode: entry.mode, oid: entry.oid });
    }
  };
  if (treeId !== null) walk(treeId, '');
  return out;
}

/** Rebuild a tree (and its subtrees) from a flattened path map. */
export function buildTree(store: ObjectStore, flat: ReadonlyMap<string, FlatEntry>): string {
  const build = (entries: readonly { name: string; rest: string[]; entry: FlatEntry }[]): string => {
    const files: TreeEntry[] = [];
    const dirs = new Map<string, { name: string; rest: string[]; entry: FlatEntry }[]>();
    for (const item of entries) {
      if (item.rest.length === 1) {
        files.push({ mode: item.entry.mode, name: item.name, oid: item.entry.oid });
      } else {
        const list = dirs.get(item.name) ?? [];
        list.push({ name: item.rest[1] as string, rest: item.rest.slice(1), entry: item.entry });
        dirs.set(item.name, list);
      }
    }
    const all: TreeEntry[] = [...files];
    for (const [name, children] of dirs) all.push({ mode: '40000', name, oid: build(children) });
    return store.put('tree', serializeTree(all));
  };
  const items = [...flat.entries()].map(([path, entry]) => {
    const segments = path.split('/');
    return { name: segments[0] as string, rest: segments, entry };
  });
  return build(items);
}

/** The tree a commit holds, or null. */
export function commitTree(store: ObjectStore, commitId: string | null): string | null {
  if (commitId === null) return null;
  const object = store.get(commitId);
  if (!object || object.type !== 'commit') return null;
  return parseCommit(object.data).tree;
}

/** The best common ancestor of two commits (nearest to both), or null when unrelated. */
export function mergeBase(store: ObjectStore, a: string, b: string): string | null {
  if (a === b) return a;
  const ancestorsOf = (tip: string): Map<string, number> => {
    const seen = new Map<string, number>();
    const queue: { oid: string; depth: number }[] = [{ oid: tip, depth: 0 }];
    while (queue.length > 0) {
      const { oid, depth } = queue.shift() as { oid: string; depth: number };
      if (seen.has(oid)) continue;
      seen.set(oid, depth);
      const object = store.get(oid);
      if (object && object.type === 'commit') {
        for (const parent of parseCommit(object.data).parents) queue.push({ oid: parent, depth: depth + 1 });
      }
    }
    return seen;
  };
  const fromA = ancestorsOf(a);
  const fromB = ancestorsOf(b);
  let best: string | null = null;
  let bestScore = Number.POSITIVE_INFINITY;
  for (const [oid, depthA] of fromA) {
    const depthB = fromB.get(oid);
    if (depthB !== undefined && depthA + depthB < bestScore) {
      best = oid;
      bestScore = depthA + depthB;
    }
  }
  return best;
}

export type MergeOutcome =
  | { readonly ok: true; readonly tree: string }
  | { readonly ok: false; readonly conflictPath: string };

/** Three-way merge of flattened trees; `base` is the merge base, `ours` the target, `theirs` the head. */
export function threeWayMerge(store: ObjectStore, baseTree: string | null, ourTree: string | null, theirTree: string | null): MergeOutcome {
  const base = flattenTree(store, baseTree);
  const ours = flattenTree(store, ourTree);
  const theirs = flattenTree(store, theirTree);
  const merged = new Map<string, FlatEntry>();
  const paths = new Set([...base.keys(), ...ours.keys(), ...theirs.keys()]);
  for (const path of paths) {
    const b = base.get(path);
    const o = ours.get(path);
    const t = theirs.get(path);
    const same = (x: FlatEntry | undefined, y: FlatEntry | undefined): boolean => x?.oid === y?.oid && x?.mode === y?.mode;
    if (same(o, t)) {
      if (o) merged.set(path, o);
    } else if (same(b, o)) {
      if (t) merged.set(path, t);
    } else if (same(b, t)) {
      if (o) merged.set(path, o);
    } else {
      return { ok: false, conflictPath: path };
    }
  }
  return { ok: true, tree: buildTree(store, merged) };
}

export interface FileDiff {
  readonly filename: string;
  readonly status: 'added' | 'modified' | 'removed';
  readonly additions: number;
  readonly deletions: number;
  readonly changes: number;
  readonly sha: string | null;
}

/** Line counts of a text diff via a longest-common-subsequence over lines (files here are small). */
function lineDiff(before: Buffer, after: Buffer): { additions: number; deletions: number } {
  const a = before.toString('utf8').split('\n');
  const b = after.toString('utf8').split('\n');
  if (a.length > 4000 || b.length > 4000) {
    return { additions: b.length, deletions: a.length };
  }
  const dp: number[] = new Array(b.length + 1).fill(0);
  for (let i = a.length - 1; i >= 0; i -= 1) {
    let diagonal = 0;
    for (let j = b.length - 1; j >= 0; j -= 1) {
      const saved = dp[j] as number;
      if (a[i] === b[j]) dp[j] = diagonal + 1;
      else dp[j] = Math.max(dp[j] as number, dp[j + 1] as number);
      diagonal = saved;
    }
  }
  const common = dp[0] as number;
  return { additions: b.length - common, deletions: a.length - common };
}

/** The files a pull request changes: merge-base tree against head tree. */
export function diffFiles(store: ObjectStore, baseTree: string | null, headTree: string | null): FileDiff[] {
  const base = flattenTree(store, baseTree);
  const head = flattenTree(store, headTree);
  const out: FileDiff[] = [];
  const paths = new Set([...base.keys(), ...head.keys()]);
  for (const path of [...paths].sort()) {
    const b = base.get(path);
    const h = head.get(path);
    if (b && h && b.oid === h.oid) continue;
    const before = b ? (store.get(b.oid)?.data ?? Buffer.alloc(0)) : Buffer.alloc(0);
    const after = h ? (store.get(h.oid)?.data ?? Buffer.alloc(0)) : Buffer.alloc(0);
    const { additions, deletions } = lineDiff(before, after);
    out.push({
      filename: path,
      status: !b ? 'added' : !h ? 'removed' : 'modified',
      additions,
      deletions,
      changes: additions + deletions,
      sha: h?.oid ?? b?.oid ?? null,
    });
  }
  return out;
}
