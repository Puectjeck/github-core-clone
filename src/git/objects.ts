// The git object model: blob, tree, commit and tag objects, their canonical bytes and SHA-1 ids.
// Everything the forge serves over git, REST contents and GraphQL is built on these primitives.
// Objects are stored by id in a plain Map; the bytes stored are the payload (without the
// "<type> <length>\0" header), and the id is the SHA-1 of header + payload, exactly as git computes.

import { createHash } from 'node:crypto';

export type GitObjectType = 'blob' | 'tree' | 'commit' | 'tag';

export interface GitObject {
  readonly type: GitObjectType;
  readonly data: Buffer;
}

export const ZERO_OID = '0000000000000000000000000000000000000000';

export function isOid(text: string): boolean {
  return /^[0-9a-f]{40}$/.test(text);
}

/** The id of an object with this type and payload. */
export function objectId(type: GitObjectType, data: Buffer): string {
  // Hash.update takes the header as a string directly: one less Buffer allocated per object id,
  // and object ids are computed for every object a pack carries in or out.
  return createHash('sha1').update(`${type} ${data.length}\0`, 'utf8').update(data).digest('hex');
}

/** An in-memory object database. */
export class ObjectStore {
  readonly objects = new Map<string, GitObject>();

  /** Store a payload; returns its id. Storing the same object twice is a no-op. */
  put(type: GitObjectType, data: Buffer): string {
    const id = objectId(type, data);
    if (!this.objects.has(id)) this.objects.set(id, { type, data });
    return id;
  }

  get(id: string): GitObject | undefined {
    return this.objects.get(id);
  }

  has(id: string): boolean {
    return this.objects.has(id);
  }
}

// --------------------------------------------------------------------------- trees

export interface TreeEntry {
  /** Octal mode as written in the tree: '40000' for a directory, '100644', '100755', '120000'. */
  readonly mode: string;
  readonly name: string;
  readonly oid: string;
}

/** Git's tree order: entries sort by name, directories as if their name ended in '/'. */
export function treeEntryCompare(a: TreeEntry, b: TreeEntry): number {
  const an = a.mode === '40000' ? `${a.name}/` : a.name;
  const bn = b.mode === '40000' ? `${b.name}/` : b.name;
  return an < bn ? -1 : an > bn ? 1 : 0;
}

export function serializeTree(entries: readonly TreeEntry[]): Buffer {
  const sorted = [...entries].sort(treeEntryCompare);
  const parts: Buffer[] = [];
  for (const entry of sorted) {
    parts.push(Buffer.from(`${entry.mode} ${entry.name}\0`, 'utf8'));
    parts.push(Buffer.from(entry.oid, 'hex'));
  }
  return Buffer.concat(parts);
}

export function parseTree(data: Buffer): TreeEntry[] {
  const entries: TreeEntry[] = [];
  let at = 0;
  while (at < data.length) {
    const space = data.indexOf(0x20, at);
    const nul = data.indexOf(0x00, space);
    const mode = data.toString('utf8', at, space);
    const name = data.toString('utf8', space + 1, nul);
    const oid = data.toString('hex', nul + 1, nul + 21);
    entries.push({ mode, name, oid });
    at = nul + 21;
  }
  return entries;
}

// --------------------------------------------------------------------------- commits

export interface CommitInfo {
  readonly tree: string;
  readonly parents: readonly string[];
  /** The raw author line after "author ". */
  readonly author: string;
  readonly committer: string;
  readonly message: string;
}

export function parseCommit(data: Buffer): CommitInfo {
  const text = data.toString('utf8');
  const split = text.indexOf('\n\n');
  const head = split >= 0 ? text.slice(0, split) : text;
  const message = split >= 0 ? text.slice(split + 2) : '';
  let tree = '';
  let author = '';
  let committer = '';
  const parents: string[] = [];
  for (const line of head.split('\n')) {
    if (line.startsWith('tree ')) tree = line.slice(5).trim();
    else if (line.startsWith('parent ')) parents.push(line.slice(7).trim());
    else if (line.startsWith('author ')) author = line.slice(7);
    else if (line.startsWith('committer ')) committer = line.slice(10);
  }
  return { tree, parents, author, committer, message };
}

export function serializeCommit(info: {
  tree: string;
  parents: readonly string[];
  author: string;
  committer: string;
  message: string;
}): Buffer {
  let head = `tree ${info.tree}\n`;
  for (const parent of info.parents) head += `parent ${parent}\n`;
  head += `author ${info.author}\ncommitter ${info.committer}\n`;
  return Buffer.from(`${head}\n${info.message}`, 'utf8');
}

/** The unix timestamp inside an ident line ("Name <email> 1700000000 +0000"), or null. */
export function identTimestamp(ident: string): number | null {
  const match = / (\d+) [+-]\d{4}$/.exec(ident);
  return match ? Number(match[1]) : null;
}

// --------------------------------------------------------------------------- tags

export interface TagInfo {
  readonly object: string;
  readonly type: string;
  readonly tag: string;
  readonly tagger: string;
  readonly message: string;
}

export function parseTag(data: Buffer): TagInfo {
  const text = data.toString('utf8');
  const split = text.indexOf('\n\n');
  const head = split >= 0 ? text.slice(0, split) : text;
  const message = split >= 0 ? text.slice(split + 2) : '';
  let object = '';
  let type = '';
  let tag = '';
  let tagger = '';
  for (const line of head.split('\n')) {
    if (line.startsWith('object ')) object = line.slice(7).trim();
    else if (line.startsWith('type ')) type = line.slice(5).trim();
    else if (line.startsWith('tag ')) tag = line.slice(4).trim();
    else if (line.startsWith('tagger ')) tagger = line.slice(7);
  }
  return { object, type, tag, tagger, message };
}

/** Follow tag objects to the commit they name; returns the commit id or null. */
export function peelToCommit(store: ObjectStore, oid: string): string | null {
  let current = oid;
  for (let depth = 0; depth < 16; depth += 1) {
    const object = store.get(current);
    if (!object) return null;
    if (object.type === 'commit') return current;
    if (object.type !== 'tag') return null;
    current = parseTag(object.data).object;
  }
  return null;
}

/** Every object reachable from `tips` (commits walk to trees, blobs and parents). */
export function reachableObjects(store: ObjectStore, tips: readonly string[]): Set<string> {
  const seen = new Set<string>();
  const stack = [...tips];
  while (stack.length > 0) {
    const id = stack.pop() as string;
    if (seen.has(id)) continue;
    const object = store.get(id);
    if (!object) continue;
    seen.add(id);
    if (object.type === 'commit') {
      const commit = parseCommit(object.data);
      stack.push(commit.tree, ...commit.parents);
    } else if (object.type === 'tree') {
      for (const entry of parseTree(object.data)) stack.push(entry.oid);
    } else if (object.type === 'tag') {
      stack.push(parseTag(object.data).object);
    }
  }
  return seen;
}

/** The set of commits reachable from `tips` (tips included), following parents only. */
export function reachableCommits(store: ObjectStore, tips: readonly string[]): Set<string> {
  const seen = new Set<string>();
  const stack = [...tips];
  while (stack.length > 0) {
    const id = stack.pop() as string;
    if (seen.has(id)) continue;
    const object = store.get(id);
    if (!object || object.type !== 'commit') continue;
    seen.add(id);
    stack.push(...parseCommit(object.data).parents);
  }
  return seen;
}

/**
 * True when `ancestor` is `descendant` itself or one of its ancestors. The walk is breadth-first
 * with an early exit the moment the ancestor is seen as a parent, so the fast-forward check on a
 * push — where the ancestor is almost always the new commit's direct parent — costs one commit
 * parse instead of a walk over the whole history (which made a chain of n pushes quadratic).
 * The visited set keeps the worst case (a genuine non-fast-forward) at one parse per commit,
 * exactly the set reachableCommits would have built.
 */
export function isAncestor(store: ObjectStore, ancestor: string, descendant: string): boolean {
  if (ancestor === descendant) return true;
  const seen = new Set<string>([descendant]);
  const queue = [descendant];
  for (let head = 0; head < queue.length; head += 1) {
    const object = store.get(queue[head] as string);
    if (!object || object.type !== 'commit') continue;
    for (const parent of parseCommit(object.data).parents) {
      if (parent === ancestor) return true;
      if (!seen.has(parent)) {
        seen.add(parent);
        queue.push(parent);
      }
    }
  }
  return false;
}
