// Tree surgery for the contents API: reading a path out of a commit's tree and rebuilding the
// tree chain when a file is written or removed. Trees are immutable — every write produces new
// tree objects along the changed path and leaves the rest shared, exactly as git does.

import {
  parseTree,
  serializeTree,
  type ObjectStore,
  type TreeEntry,
} from './objects.ts';

export const EMPTY_TREE = '4b825dc642cb6eb9a060e54bf8d69288fbee4904';

/** The entries of a tree object; an absent tree reads as empty. */
export function treeEntries(store: ObjectStore, treeId: string | null): TreeEntry[] {
  if (treeId === null) return [];
  const object = store.get(treeId);
  if (!object || object.type !== 'tree') return [];
  return parseTree(object.data);
}

/** Resolve a path (segments) inside a tree: the entry it names, or null. */
export function lookupPath(store: ObjectStore, rootTree: string | null, segments: readonly string[]): TreeEntry | null {
  let entries = treeEntries(store, rootTree);
  for (let depth = 0; depth < segments.length; depth += 1) {
    const entry = entries.find((candidate) => candidate.name === segments[depth]);
    if (!entry) return null;
    if (depth === segments.length - 1) return entry;
    if (entry.mode !== '40000') return null;
    entries = treeEntries(store, entry.oid);
  }
  return null;
}

/**
 * Return the id of a tree like `rootTree` but with `leaf` placed at `segments` (or removed when
 * `leaf` is null). Intermediate directories are created as needed; a directory emptied by a
 * removal disappears, as in git.
 */
export function updatePath(
  store: ObjectStore,
  rootTree: string | null,
  segments: readonly string[],
  leaf: { mode: string; oid: string } | null,
): string {
  const build = (treeId: string | null, depth: number): string | null => {
    const entries = treeEntries(store, treeId).filter((entry) => entry.name !== segments[depth]);
    if (depth === segments.length - 1) {
      if (leaf !== null) entries.push({ mode: leaf.mode, name: segments[depth] as string, oid: leaf.oid });
    } else {
      const existing = treeEntries(store, treeId).find((entry) => entry.name === segments[depth]);
      const childBefore = existing && existing.mode === '40000' ? existing.oid : null;
      const childAfter = build(childBefore, depth + 1);
      if (childAfter !== null) entries.push({ mode: '40000', name: segments[depth] as string, oid: childAfter });
    }
    if (entries.length === 0) return null;
    return store.put('tree', serializeTree(entries));
  };
  return build(rootTree, 0) ?? store.put('tree', serializeTree([]));
}
