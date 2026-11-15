// The forge's state: one shared in-memory store behind git, REST, GraphQL and MCP, so an object
// created through one surface reads back identically through the others. All timestamps come from
// a deterministic clock seeded by MAKEFUN_TRIAL_NOW (the trial's opening time) and advanced by a
// fixed step per event, so a replayed trial produces byte-identical responses.

import { ObjectStore } from './git/objects.ts';
import { viewerUser, type ScenarioUser } from './scenario.ts';

/** A clock that never reads the wall: seeded once, then one second per tick. */
export class ForgeClock {
  #now: number;

  constructor() {
    const raw = process.env.MAKEFUN_TRIAL_NOW;
    let base = raw ? Date.parse(raw) : Number.NaN;
    if (Number.isNaN(base)) base = Date.parse('2026-11-14T06:30:00.000Z');
    this.#now = base;
  }

  /** The current instant as an ISO string; every call advances the clock by one second. */
  tick(): string {
    const value = new Date(this.#now).toISOString();
    this.#now += 1000;
    return value;
  }

  /** The current instant without advancing (for fields that must agree with a sibling field). */
  peek(): string {
    return new Date(this.#now).toISOString();
  }
}

export interface CommentRecord {
  readonly id: number;
  readonly body: string;
  readonly author: string;
  readonly createdAt: string;
  readonly updatedAt: string;
}

export interface IssueRecord {
  readonly kind: 'issue';
  readonly number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  labels: string[];
  comments: CommentRecord[];
  readonly author: string;
  readonly createdAt: string;
  updatedAt: string;
}

export interface PullRecord {
  readonly kind: 'pull';
  readonly number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  labels: string[];
  comments: CommentRecord[];
  readonly author: string;
  readonly createdAt: string;
  updatedAt: string;
  headRef: string;
  baseRef: string;
  headSha: string;
  baseSha: string;
  merged: boolean;
  mergedAt: string | null;
  mergeCommitSha: string | null;
}

export type ItemRecord = IssueRecord | PullRecord;

export interface LabelRecord {
  readonly name: string;
  readonly color: string;
  readonly description: string;
}

export interface RepoRecord {
  readonly id: number;
  name: string;
  readonly owner: string;
  description: string;
  isPrivate: boolean;
  readonly createdAt: string;
  updatedAt: string;
  pushedAt: string;
  readonly objects: ObjectStore;
  /** Full ref name (refs/heads/main, refs/tags/v1) to object id. */
  readonly refs: Map<string, string>;
  /** The ref HEAD points at. */
  headRef: string;
  readonly labels: Map<string, LabelRecord>;
  /**
   * Memoized upload-pack fetch responses, keyed by everything the response depends on (see
   * fetchCacheKey in git/server.ts). Objects are immutable and never leave the store, so a key
   * that includes the object count can never go stale: the same count is the same object set.
   * Purely an in-memory speed cache — a miss recomputes the identical bytes.
   */
  readonly fetchCache: Map<string, Buffer>;
  /** Issues and pull requests share one number sequence, as on GitHub. */
  readonly items: Map<number, ItemRecord>;
  nextItemNumber: number;
  nextCommentId: number;
}

export class ForgeStore {
  readonly clock = new ForgeClock();
  readonly viewer: ScenarioUser = viewerUser();
  readonly repos = new Map<string, RepoRecord>();
  readonly reposById = new Map<number, RepoRecord>();
  #nextRepoId = 1;

  static key(owner: string, name: string): string {
    return `${owner}/${name}`;
  }

  createRepo(input: { owner: string; name: string; description?: string; isPrivate?: boolean }): RepoRecord {
    const now = this.clock.tick();
    const repo: RepoRecord = {
      id: this.#nextRepoId,
      name: input.name,
      owner: input.owner,
      description: input.description ?? '',
      isPrivate: input.isPrivate ?? false,
      createdAt: now,
      updatedAt: now,
      pushedAt: now,
      objects: new ObjectStore(),
      refs: new Map(),
      headRef: 'refs/heads/main',
      fetchCache: new Map(),
      labels: new Map(),
      items: new Map(),
      nextItemNumber: 1,
      nextCommentId: 1,
    };
    this.#nextRepoId += 1;
    this.repos.set(ForgeStore.key(repo.owner, repo.name), repo);
    this.reposById.set(repo.id, repo);
    return repo;
  }

  getRepo(owner: string, name: string): RepoRecord | undefined {
    return this.repos.get(ForgeStore.key(owner, name));
  }

  deleteRepo(repo: RepoRecord): void {
    this.repos.delete(ForgeStore.key(repo.owner, repo.name));
    this.reposById.delete(repo.id);
  }

  reposOf(owner: string): RepoRecord[] {
    return [...this.repos.values()].filter((repo) => repo.owner === owner).sort((a, b) => a.id - b.id);
  }

  /** The tip of the repo's default branch, or null while the repository is empty. */
  defaultTip(repo: RepoRecord): string | null {
    return repo.refs.get(repo.headRef) ?? null;
  }
}
