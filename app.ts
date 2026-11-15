// The forge's HTTP surface: the git smart-HTTP endpoints and the REST v3 API on one shared
// in-memory store. Every REST reply is JSON with a request id; every failure is a JSON body with a
// string `message`; the server never crashes on a bad request — a malformed body is a 400, an
// unknown route a 404, and a repository under another owner reads as absent, never forbidden.

import { createServer as createHttpServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { parseCommit, parseTree, peelToCommit, serializeCommit } from './src/git/objects.ts';
import { advertiseRefs, handleReceivePack, handleUploadPack } from './src/git/server.ts';
import { lookupPath, treeEntries, updatePath } from './src/git/trees.ts';
import { commitTree, diffFiles, mergeBase, threeWayMerge } from './src/pulls.ts';
import { ForgeStore, type IssueRecord, type LabelRecord, type PullRecord, type RepoRecord } from './src/store.ts';
import { handleGraphQL } from './src/graphql.ts';
import { handleMcp } from './src/mcp.ts';

const MAX_BODY = 64 * 1024 * 1024;
const REQUEST_ID = 'forge-1';

interface Ctx {
  readonly store: ForgeStore;
  readonly base: string;
  readonly token: string | null;
}

function authorized(ctx: Ctx, req: IncomingMessage): boolean {
  if (ctx.token === null) return true;
  return req.headers.authorization === `Bearer ${ctx.token}`;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = body === undefined ? '' : JSON.stringify(body);
  // Only the headers the wire contract names: JSON content type and a non-empty request id.
  // The X-RateLimit-* trio GitHub sends is explicitly not required, and three fewer header
  // writes per response is free speed on the read/write workloads' thousands of replies.
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'x-github-request-id': REQUEST_ID,
  });
  res.end(text);
}

function sendError(res: ServerResponse, status: number, message: string): void {
  sendJson(res, status, { message, documentation_url: 'https://docs.github.com/rest' });
}

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_BODY) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function readJsonBody(req: IncomingMessage): Promise<{ ok: true; value: Record<string, unknown> } | { ok: false }> {
  const raw = await readBody(req);
  if (raw.length === 0) return { ok: true, value: {} };
  try {
    const value = JSON.parse(raw.toString('utf8')) as unknown;
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return { ok: false };
    return { ok: true, value: value as Record<string, unknown> };
  } catch {
    return { ok: false };
  }
}

/**
 * The viewer's JSON is a pure function of the store's viewer and the request's base URL, and it
 * is embedded in nearly every REST response (repo owner, issue author, both pull sides). Build it
 * once per (store, base) and share the object: callers only ever read it into JSON.stringify.
 */
const userJsonCache = new WeakMap<ForgeStore, Map<string, Record<string, unknown>>>();

function userJson(ctx: Ctx): Record<string, unknown> {
  let byBase = userJsonCache.get(ctx.store);
  if (!byBase) {
    byBase = new Map();
    userJsonCache.set(ctx.store, byBase);
  }
  const cached = byBase.get(ctx.base);
  if (cached) return cached;
  const viewer = ctx.store.viewer;
  const built: Record<string, unknown> = {
    login: viewer.login,
    id: viewer.id,
    node_id: Buffer.from(`04:User${viewer.id}`).toString('base64'),
    type: 'User',
    site_admin: false,
    name: viewer.name,
    email: viewer.email,
    avatar_url: `${ctx.base}/avatars/${viewer.id}`,
    url: `${ctx.base}/api/v3/users/${viewer.login}`,
    html_url: `${ctx.base}/${viewer.login}`,
  };
  byBase.set(ctx.base, built);
  return built;
}

function repoJson(ctx: Ctx, repo: RepoRecord): Record<string, unknown> {
  const fullName = `${repo.owner}/${repo.name}`;
  const openIssues = [...repo.items.values()].filter((item) => item.state === 'open').length;
  return {
    id: repo.id,
    node_id: Buffer.from(`010:Repository${repo.id}`).toString('base64'),
    name: repo.name,
    full_name: fullName,
    private: repo.isPrivate,
    owner: userJson(ctx),
    description: repo.description,
    fork: false,
    created_at: repo.createdAt,
    updated_at: repo.updatedAt,
    pushed_at: repo.pushedAt,
    size: 0,
    stargazers_count: 0,
    watchers_count: 0,
    forks_count: 0,
    open_issues_count: openIssues,
    language: null,
    topics: [],
    visibility: repo.isPrivate ? 'private' : 'public',
    default_branch: repo.headRef.replace('refs/heads/', ''),
    archived: false,
    disabled: false,
    url: `${ctx.base}/api/v3/repos/${fullName}`,
    html_url: `${ctx.base}/${fullName}`,
    clone_url: `${ctx.base}/${fullName}.git`,
    git_url: `git://${ctx.base.replace(/^https?:\/\//, '')}/${fullName}.git`,
    ssh_url: `git@${ctx.base.replace(/^https?:\/\//, '')}:${fullName}.git`,
    homepage: null,
    license: null,
    allow_forking: true,
    is_template: false,
    has_issues: true,
    has_projects: true,
    has_wiki: true,
    has_pages: false,
    has_downloads: true,
  };
}

/** Parse ?page / ?per_page with GitHub's defaults (page 1, 30 per page, at most 100). */
function paging(url: URL): { page: number; perPage: number } {
  const page = Math.max(1, Number(url.searchParams.get('page') ?? '1') || 1);
  const perPage = Math.min(100, Math.max(1, Number(url.searchParams.get('per_page') ?? '30') || 30));
  return { page, perPage };
}

function sendPage<T>(ctx: Ctx, res: ServerResponse, url: URL, all: readonly T[], render: (item: T) => unknown): void {
  const { page, perPage } = paging(url);
  const totalPages = Math.max(1, Math.ceil(all.length / perPage));
  const slice = all.slice((page - 1) * perPage, page * perPage);
  const links: string[] = [];
  const linkFor = (target: number, rel: string): string => {
    const next = new URL(url.toString());
    next.searchParams.set('page', String(target));
    next.searchParams.set('per_page', String(perPage));
    return `<${next.toString()}>; rel="${rel}"`;
  };
  if (page < totalPages) links.push(linkFor(page + 1, 'next'), linkFor(totalPages, 'last'));
  if (page > 1) links.push(linkFor(page - 1, 'prev'), linkFor(1, 'first'));
  if (links.length > 0) res.setHeader('link', links.join(', '));
  sendJson(res, 200, slice.map(render));
}

const REPO_NAME_OK = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** Create one commit on a branch ref holding `treeId`, authored by the viewer at the forge clock. */
function makeCommit(ctx: Ctx, repo: RepoRecord, branchRef: string, treeId: string, message: string): { sha: string; at: string } {
  const at = ctx.store.clock.tick();
  const epoch = Math.floor(Date.parse(at) / 1000);
  const viewer = ctx.store.viewer;
  const ident = `${viewer.name} <${viewer.email}> ${epoch} +0000`;
  const parent = repo.refs.get(branchRef);
  const data = serializeCommit({ tree: treeId, parents: parent ? [parent] : [], author: ident, committer: ident, message });
  const sha = repo.objects.put('commit', data);
  repo.refs.set(branchRef, sha);
  repo.pushedAt = at;
  repo.updatedAt = at;
  return { sha, at };
}

/** The commit JSON the contents API returns for a commit the forge just made. */
function contentsCommitJson(ctx: Ctx, repo: RepoRecord, sha: string): Record<string, unknown> {
  const object = repo.objects.get(sha);
  const info = object && object.type === 'commit' ? parseCommit(object.data) : { tree: '', parents: [], author: '', committer: '', message: '' };
  const viewer = ctx.store.viewer;
  const ident = (line: string): { name: string; email: string; date: string } => {
    const match = /^(.*?) <(.*?)> (\d+) [+-]\d{4}$/.exec(line);
    if (!match) return { name: viewer.name, email: viewer.email, date: repo.updatedAt };
    return { name: match[1] as string, email: match[2] as string, date: new Date(Number(match[3]) * 1000).toISOString() };
  };
  return {
    sha,
    node_id: Buffer.from(`010:Commit${sha}`).toString('base64'),
    url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/commits/${sha}`,
    html_url: `${ctx.base}/${repo.owner}/${repo.name}/commit/${sha}`,
    author: ident(info.author),
    committer: ident(info.committer),
    tree: { sha: info.tree, url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/trees/${info.tree}` },
    message: info.message,
    parents: info.parents.map((parent) => ({ sha: parent, url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/commits/${parent}` })),
  };
}

async function handleRest(ctx: Ctx, req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  const path = url.pathname.replace(/\/+$/, '') || '/';
  const method = req.method ?? 'GET';
  const store = ctx.store;
  const viewer = store.viewer;

  if (path === '/api/v3') {
    res.setHeader('x-oauth-scopes', '');
    sendJson(res, 200, {
      current_user_url: `${ctx.base}/api/v3/user`,
      repository_url: `${ctx.base}/api/v3/repos/{owner}/{repo}`,
      repository_search_url: `${ctx.base}/api/v3/search/repositories?q={query}`,
      issue_search_url: `${ctx.base}/api/v3/search/issues?q={query}`,
      issues_url: `${ctx.base}/api/v3/issues`,
      organization_repositories_url: `${ctx.base}/api/v3/orgs/{org}/repos`,
      repository_collaborators_url: `${ctx.base}/api/v3/repos/{owner}/{repo}/collaborators{/collaborator}`,
    });
    return;
  }
  if (!authorized(ctx, req)) {
    sendError(res, 401, 'Bad credentials');
    return;
  }
  if (path === '/api/v3/meta') {
    sendJson(res, 200, {
      installed_version: '1.0.0-core-forge',
      verifiable_password_authentication: false,
      github_services_sha: 'core-forge',
      hooks: [],
      git: [],
      api: [],
      pages: [],
      importer: [],
      packages: [],
      actions: [],
    });
    return;
  }
  if (path === '/api/v3/user' && method === 'GET') {
    sendJson(res, 200, userJson(ctx));
    return;
  }
  if (path === '/api/v3/user/repos') {
    if (method === 'GET') {
      sendPage(ctx, res, url, store.reposOf(viewer.login), (repo) => repoJson(ctx, repo));
      return;
    }
    if (method === 'POST') {
      const body = await readJsonBody(req);
      if (!body.ok) {
        sendError(res, 400, 'Problems parsing JSON');
        return;
      }
      const name = typeof body.value.name === 'string' ? body.value.name : '';
      if (!REPO_NAME_OK.test(name) || name === '.' || name === '..') {
        sendJson(res, 422, { message: 'Repository creation failed.', errors: [{ resource: 'Repository', code: 'custom', field: 'name', message: 'name is invalid' }] });
        return;
      }
      if (store.getRepo(viewer.login, name)) {
        sendJson(res, 422, { message: 'Repository creation failed.', errors: [{ resource: 'Repository', code: 'custom', field: 'name', message: 'name already exists on this account' }] });
        return;
      }
      const repo = store.createRepo({
        owner: viewer.login,
        name,
        description: typeof body.value.description === 'string' ? body.value.description : '',
        isPrivate: body.value.private === true,
      });
      sendJson(res, 201, repoJson(ctx, repo));
      return;
    }
  }
  const userRepos = /^\/api\/v3\/users\/([^/]+)\/repos$/.exec(path);
  if (userRepos && method === 'GET') {
    const owner = decodeURIComponent(userRepos[1] as string);
    if (owner !== viewer.login) {
      sendJson(res, 200, []);
      return;
    }
    sendPage(ctx, res, url, store.reposOf(owner), (repo) => repoJson(ctx, repo));
    return;
  }

  const repoMatch = /^\/api\/v3\/repos\/([^/]+)\/([^/]+)(\/.*)?$/.exec(path);
  if (repoMatch) {
    const owner = decodeURIComponent(repoMatch[1] as string);
    const name = decodeURIComponent(repoMatch[2] as string);
    const rest = repoMatch[3] ?? '';
    const repo = owner === viewer.login ? store.getRepo(owner, name) : undefined;
    if (!repo) {
      sendError(res, 404, 'Not Found');
      return;
    }
    if (rest === '') {
      if (method === 'GET') {
        sendJson(res, 200, repoJson(ctx, repo));
        return;
      }
      if (method === 'PATCH') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        if (typeof body.value.description === 'string') repo.description = body.value.description;
        if (typeof body.value.private === 'boolean') repo.isPrivate = body.value.private;
        if (typeof body.value.name === 'string' && REPO_NAME_OK.test(body.value.name) && body.value.name !== repo.name) {
          store.deleteRepo(repo);
          repo.name = body.value.name;
          store.repos.set(ForgeStore.key(repo.owner, repo.name), repo);
          store.reposById.set(repo.id, repo);
        }
        repo.updatedAt = store.clock.tick();
        sendJson(res, 200, repoJson(ctx, repo));
        return;
      }
      if (method === 'DELETE') {
        store.deleteRepo(repo);
        res.writeHead(204, { 'x-github-request-id': REQUEST_ID });
        res.end();
        return;
      }
    }
    if (rest === '/branches' && method === 'GET') {
      const branches = [...repo.refs.keys()]
        .filter((ref) => ref.startsWith('refs/heads/'))
        .sort()
        .map((ref) => ({
          name: ref.slice('refs/heads/'.length),
          commit: { sha: repo.refs.get(ref) as string, url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/commits/${repo.refs.get(ref)}` },
          protected: false,
        }));
      sendJson(res, 200, branches);
      return;
    }
    if (rest === '/commits' && method === 'GET') {
      const shaParam = url.searchParams.get('sha');
      let tip: string | null = null;
      if (shaParam) {
        tip = repo.refs.get(`refs/heads/${shaParam}`) ?? (repo.objects.has(shaParam) ? peelToCommit(repo.objects, shaParam) : null);
      } else {
        tip = store.defaultTip(repo);
      }
      const commits: Record<string, unknown>[] = [];
      const seen = new Set<string>();
      const queue = tip ? [tip] : [];
      while (queue.length > 0 && commits.length < 250) {
        const id = queue.shift() as string;
        if (seen.has(id)) continue;
        seen.add(id);
        const object = repo.objects.get(id);
        if (!object || object.type !== 'commit') continue;
        const info = parseCommit(object.data);
        const ident = (line: string): { name: string; email: string; date: string } => {
          const match = /^(.*?) <(.*?)> (\d+) [+-]\d{4}$/.exec(line);
          if (!match) return { name: line, email: '', date: repo.createdAt };
          return { name: match[1] as string, email: match[2] as string, date: new Date(Number(match[3]) * 1000).toISOString() };
        };
        commits.push({
          sha: id,
          node_id: Buffer.from(`010:Commit${id}`).toString('base64'),
          commit: {
            author: ident(info.author),
            committer: ident(info.committer),
            message: info.message,
            tree: { sha: info.tree, url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/trees/${info.tree}` },
            comment_count: 0,
          },
          author: { login: viewer.login, id: viewer.id, type: 'User' },
          committer: { login: viewer.login, id: viewer.id, type: 'User' },
          parents: info.parents.map((parent) => ({ sha: parent, url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/commits/${parent}` })),
        });
        queue.push(...info.parents);
      }
      sendPage(ctx, res, url, commits, (commit) => commit);
      return;
    }
    if (rest === '/git/refs' && method === 'POST') {
      const body = await readJsonBody(req);
      if (!body.ok) {
        sendError(res, 400, 'Problems parsing JSON');
        return;
      }
      const ref = typeof body.value.ref === 'string' ? body.value.ref : '';
      const sha = typeof body.value.sha === 'string' ? body.value.sha : '';
      const target = repo.objects.get(sha);
      if (!ref.startsWith('refs/') || !target) {
        sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'Reference', code: 'custom', message: 'ref or sha is invalid' }] });
        return;
      }
      if (repo.refs.has(ref)) {
        sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'Reference', code: 'custom', message: 'Reference already exists' }] });
        return;
      }
      repo.refs.set(ref, sha);
      sendJson(res, 201, {
        ref,
        node_id: Buffer.from(`010:Reference${ref}`).toString('base64'),
        url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/refs/${ref.slice('refs/'.length)}`,
        object: { sha, type: target.type, url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/${target.type}s/${sha}` },
      });
      return;
    }

    if (rest === '/contents' || rest.startsWith('/contents/')) {
      const segments = rest === '/contents' ? [] : rest.slice('/contents/'.length).split('/').map((piece) => decodeURIComponent(piece));
      const filePath = segments.join('/');
      const fileJson = (entry: { mode: string; name: string; oid: string }, path: string): Record<string, unknown> => {
        const blob = repo.objects.get(entry.oid);
        const size = blob ? blob.data.length : 0;
        return {
          name: entry.name,
          path,
          sha: entry.oid,
          size,
          url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/contents/${path}`,
          html_url: `${ctx.base}/${repo.owner}/${repo.name}/blob/${repo.headRef.replace('refs/heads/', '')}/${path}`,
          git_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/blobs/${entry.oid}`,
          download_url: `${ctx.base}/${repo.owner}/${repo.name}/raw/${repo.headRef.replace('refs/heads/', '')}/${path}`,
          type: 'file',
          encoding: 'base64',
          content: blob ? blob.data.toString('base64') : '',
          _links: {
            self: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/contents/${path}`,
            git: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/blobs/${entry.oid}`,
            html: `${ctx.base}/${repo.owner}/${repo.name}/blob/${repo.headRef.replace('refs/heads/', '')}/${path}`,
          },
        };
      };
      const resolveCommit = (refParam: string | null): string | null => {
        if (refParam) {
          const byBranch = repo.refs.get(`refs/heads/${refParam}`) ?? repo.refs.get(refParam);
          if (byBranch) return peelToCommit(repo.objects, byBranch);
          if (repo.objects.has(refParam)) return peelToCommit(repo.objects, refParam);
          return null;
        }
        return store.defaultTip(repo);
      };

      if (method === 'GET') {
        const tip = resolveCommit(url.searchParams.get('ref'));
        if (!tip) {
          sendError(res, 404, 'Not Found');
          return;
        }
        const commitObject = repo.objects.get(tip);
        const rootTree = commitObject && commitObject.type === 'commit' ? parseCommit(commitObject.data).tree : null;
        if (segments.length === 0) {
          const entries = treeEntries(repo.objects, rootTree).map((entry) => ({
            name: entry.name,
            path: entry.name,
            sha: entry.oid,
            size: entry.mode === '40000' ? 0 : (repo.objects.get(entry.oid)?.data.length ?? 0),
            type: entry.mode === '40000' ? 'dir' : 'file',
            url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/contents/${entry.name}`,
          }));
          sendJson(res, 200, entries);
          return;
        }
        const entry = lookupPath(repo.objects, rootTree, segments);
        if (!entry) {
          sendError(res, 404, 'Not Found');
          return;
        }
        if (entry.mode === '40000') {
          const entries = treeEntries(repo.objects, entry.oid).map((child) => ({
            name: child.name,
            path: `${filePath}/${child.name}`,
            sha: child.oid,
            size: child.mode === '40000' ? 0 : (repo.objects.get(child.oid)?.data.length ?? 0),
            type: child.mode === '40000' ? 'dir' : 'file',
            url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/contents/${filePath}/${child.name}`,
          }));
          sendJson(res, 200, entries);
          return;
        }
        sendJson(res, 200, fileJson(entry, filePath));
        return;
      }

      if (method === 'PUT' || method === 'DELETE') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        const branchName = typeof body.value.branch === 'string' ? body.value.branch : repo.headRef.replace('refs/heads/', '');
        const branchRef = `refs/heads/${branchName}`;
        if (branchRef !== repo.headRef && !repo.refs.has(branchRef)) {
          sendError(res, 404, 'Not Found');
          return;
        }
        const tip = repo.refs.get(branchRef) ?? null;
        const tipObject = tip ? repo.objects.get(tip) : undefined;
        const rootTree = tipObject && tipObject.type === 'commit' ? parseCommit(tipObject.data).tree : null;
        const existing = segments.length > 0 ? lookupPath(repo.objects, rootTree, segments) : null;
        const message = typeof body.value.message === 'string' ? body.value.message : `${method === 'PUT' ? 'Update' : 'Delete'} ${filePath}`;
        if (method === 'PUT') {
          if (existing && existing.mode === '40000') {
            sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'Content', code: 'custom', message: 'path is a directory' }] });
            return;
          }
          const givenSha = typeof body.value.sha === 'string' ? body.value.sha : null;
          if (existing && givenSha !== existing.oid) {
            sendJson(res, 409, { message: `${filePath} does not match ${givenSha ?? '(no sha given)'}`, errors: [{ resource: 'Content', code: 'custom', message: 'sha does not match' }] });
            return;
          }
          const raw = typeof body.value.content === 'string' ? Buffer.from(body.value.content, 'base64') : Buffer.alloc(0);
          const blobId = repo.objects.put('blob', raw);
          const newTree = updatePath(repo.objects, rootTree, segments, { mode: '100644', oid: blobId });
          const commit = makeCommit(ctx, repo, branchRef, newTree, message);
          const entry = { mode: '100644', name: segments[segments.length - 1] as string, oid: blobId };
          sendJson(res, existing ? 200 : 201, { content: fileJson(entry, filePath), commit: contentsCommitJson(ctx, repo, commit.sha) });
          return;
        }
        // DELETE
        if (!existing || existing.mode === '40000') {
          sendError(res, 404, 'Not Found');
          return;
        }
        const givenSha = typeof body.value.sha === 'string' ? body.value.sha : null;
        if (givenSha !== existing.oid) {
          sendJson(res, 409, { message: `${filePath} does not match ${givenSha ?? '(no sha given)'}`, errors: [{ resource: 'Content', code: 'custom', message: 'sha does not match' }] });
          return;
        }
        const newTree = updatePath(repo.objects, rootTree, segments, null);
        const commit = makeCommit(ctx, repo, branchRef, newTree, message);
        sendJson(res, 200, { content: null, commit: contentsCommitJson(ctx, repo, commit.sha) });
        return;
      }
    }

    const labelJson = (label: LabelRecord): Record<string, unknown> => ({
      id: [...repo.labels.keys()].indexOf(label.name) + 1,
      node_id: Buffer.from(`010:Label${repo.id}:${label.name}`).toString('base64'),
      url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/labels/${encodeURIComponent(label.name)}`,
      name: label.name,
      color: label.color,
      description: label.description,
      default: false,
    });
    const issueJson = (issue: IssueRecord): Record<string, unknown> => ({
      id: repo.id * 1_000_000 + issue.number,
      node_id: Buffer.from(`010:Issue${repo.id}:${issue.number}`).toString('base64'),
      number: issue.number,
      title: issue.title,
      body: issue.body,
      state: issue.state,
      labels: issue.labels.map((name) => {
        const label = repo.labels.get(name);
        return label ? labelJson(label) : { name };
      }),
      user: userJson(ctx),
      author_association: 'OWNER',
      comments: issue.comments.length,
      created_at: issue.createdAt,
      updated_at: issue.updatedAt,
      closed_at: issue.state === 'closed' ? issue.updatedAt : null,
      url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}`,
      repository_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}`,
      labels_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}/labels{/name}`,
      comments_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}/comments`,
      events_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}/events`,
      html_url: `${ctx.base}/${repo.owner}/${repo.name}/issues/${issue.number}`,
    });

    if (rest === '/labels') {
      if (method === 'GET') {
        sendJson(res, 200, [...repo.labels.values()].map(labelJson));
        return;
      }
      if (method === 'POST') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        const name = typeof body.value.name === 'string' ? body.value.name : '';
        if (name === '' || repo.labels.has(name)) {
          sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'Label', code: 'custom', field: 'name', message: name === '' ? 'name is required' : 'name already exists' }] });
          return;
        }
        const label: LabelRecord = {
          name,
          color: typeof body.value.color === 'string' ? body.value.color.replace(/^#/, '') : 'ededed',
          description: typeof body.value.description === 'string' ? body.value.description : '',
        };
        repo.labels.set(name, label);
        sendJson(res, 201, labelJson(label));
        return;
      }
    }

    if (rest === '/issues') {
      if (method === 'GET') {
        const state = url.searchParams.get('state') ?? 'open';
        const direction = url.searchParams.get('direction') ?? 'desc';
        let issues = [...repo.items.values()].filter((item): item is IssueRecord => item.kind === 'issue');
        if (state === 'open' || state === 'closed') issues = issues.filter((issue) => issue.state === state);
        issues.sort((a, b) => (direction === 'asc' ? a.number - b.number : b.number - a.number));
        sendPage(ctx, res, url, issues, (issue) => issueJson(issue));
        return;
      }
      if (method === 'POST') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        const title = typeof body.value.title === 'string' ? body.value.title : '';
        if (title === '') {
          sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'Issue', code: 'missing_field', field: 'title' }] });
          return;
        }
        const labelNames = Array.isArray(body.value.labels) ? body.value.labels.filter((entry): entry is string => typeof entry === 'string') : [];
        for (const name of labelNames) {
          if (!repo.labels.has(name)) {
            sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'Label', code: 'custom', field: 'labels', message: `label ${name} does not exist` }] });
            return;
          }
        }
        const now = store.clock.tick();
        const issue: IssueRecord = {
          kind: 'issue',
          number: repo.nextItemNumber,
          title,
          body: typeof body.value.body === 'string' ? body.value.body : '',
          state: 'open',
          labels: [...labelNames],
          comments: [],
          author: viewer.login,
          createdAt: now,
          updatedAt: now,
        };
        repo.nextItemNumber += 1;
        repo.items.set(issue.number, issue);
        sendJson(res, 201, issueJson(issue));
        return;
      }
    }

    const issueMatch = /^\/issues\/(\d+)(\/comments)?$/.exec(rest);
    if (issueMatch) {
      const number = Number(issueMatch[1]);
      const item = repo.items.get(number);
      const issue = item && item.kind === 'issue' ? item : undefined;
      if (!issue) {
        sendError(res, 404, 'Not Found');
        return;
      }
      if (issueMatch[2] === '/comments') {
        if (method === 'GET') {
          sendPage(ctx, res, url, issue.comments, (comment) => ({
            id: comment.id,
            node_id: Buffer.from(`010:IssueComment${comment.id}`).toString('base64'),
            body: comment.body,
            user: userJson(ctx),
            author_association: 'OWNER',
            created_at: comment.createdAt,
            updated_at: comment.updatedAt,
            url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/comments/${comment.id}`,
            html_url: `${ctx.base}/${repo.owner}/${repo.name}/issues/${issue.number}#issuecomment-${comment.id}`,
            issue_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}`,
          }));
          return;
        }
        if (method === 'POST') {
          const body = await readJsonBody(req);
          if (!body.ok) {
            sendError(res, 400, 'Problems parsing JSON');
            return;
          }
          const now = store.clock.tick();
          const comment = {
            id: repo.id * 1_000_000 + repo.nextCommentId,
            body: typeof body.value.body === 'string' ? body.value.body : '',
            author: viewer.login,
            createdAt: now,
            updatedAt: now,
          };
          repo.nextCommentId += 1;
          issue.comments.push(comment);
          issue.updatedAt = now;
          sendJson(res, 201, {
            id: comment.id,
            node_id: Buffer.from(`010:IssueComment${comment.id}`).toString('base64'),
            body: comment.body,
            user: userJson(ctx),
            author_association: 'OWNER',
            created_at: comment.createdAt,
            updated_at: comment.updatedAt,
            url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/comments/${comment.id}`,
            html_url: `${ctx.base}/${repo.owner}/${repo.name}/issues/${issue.number}#issuecomment-${comment.id}`,
            issue_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}`,
          });
          return;
        }
      }
      if (method === 'GET') {
        sendJson(res, 200, issueJson(issue));
        return;
      }
      if (method === 'PATCH') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        if (typeof body.value.title === 'string') issue.title = body.value.title;
        if (typeof body.value.body === 'string') issue.body = body.value.body;
        if (body.value.state === 'open' || body.value.state === 'closed') issue.state = body.value.state;
        if (Array.isArray(body.value.labels)) {
          const names = body.value.labels.filter((entry): entry is string => typeof entry === 'string');
          for (const name of names) {
            if (!repo.labels.has(name)) {
              sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'Label', code: 'custom', field: 'labels', message: `label ${name} does not exist` }] });
              return;
            }
          }
          issue.labels = [...names];
        }
        issue.updatedAt = store.clock.tick();
        sendJson(res, 200, issueJson(issue));
        return;
      }
    }

    const pullJson = (pull: PullRecord): Record<string, unknown> => {
      const headSha = repo.refs.get(`refs/heads/${pull.headRef}`) ?? pull.headSha;
      const baseSha = repo.refs.get(`refs/heads/${pull.baseRef}`) ?? pull.baseSha;
      const base = mergeBase(repo.objects, baseSha, headSha);
      const files = pull.merged ? [] : diffFiles(repo.objects, commitTree(repo.objects, base), commitTree(repo.objects, headSha));
      const conflict = !pull.merged && pull.state === 'open'
        ? !threeWayMerge(repo.objects, commitTree(repo.objects, base), commitTree(repo.objects, baseSha), commitTree(repo.objects, headSha)).ok
        : false;
      const side = (ref: string, sha: string): Record<string, unknown> => ({
        label: `${repo.owner}:${ref}`,
        ref,
        sha,
        user: userJson(ctx),
        repo: repoJson(ctx, repo),
      });
      return {
        id: repo.id * 1_000_000 + pull.number,
        node_id: Buffer.from(`010:PullRequest${repo.id}:${pull.number}`).toString('base64'),
        number: pull.number,
        state: pull.state,
        title: pull.title,
        body: pull.body,
        locked: false,
        draft: false,
        user: userJson(ctx),
        labels: pull.labels.map((name) => {
          const label = repo.labels.get(name);
          return label ? labelJson(label) : { name };
        }),
        head: side(pull.headRef, headSha),
        base: side(pull.baseRef, baseSha),
        merged: pull.merged,
        merged_at: pull.mergedAt,
        merge_commit_sha: pull.mergeCommitSha,
        closed_at: pull.state === 'closed' ? (pull.mergedAt ?? pull.updatedAt) : null,
        mergeable: pull.state === 'open' ? !conflict : null,
        mergeable_state: pull.state !== 'open' ? 'unknown' : conflict ? 'dirty' : 'clean',
        rebaseable: pull.state === 'open' && !conflict,
        maintainer_can_modify: false,
        commits: 0,
        additions: files.reduce((sum, file) => sum + file.additions, 0),
        deletions: files.reduce((sum, file) => sum + file.deletions, 0),
        changed_files: files.length,
        comments: pull.comments.length,
        review_comments: 0,
        created_at: pull.createdAt,
        updated_at: pull.updatedAt,
        author_association: 'OWNER',
        active_lock_reason: null,
        url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/pulls/${pull.number}`,
        html_url: `${ctx.base}/${repo.owner}/${repo.name}/pull/${pull.number}`,
        diff_url: `${ctx.base}/${repo.owner}/${repo.name}/pull/${pull.number}.diff`,
        patch_url: `${ctx.base}/${repo.owner}/${repo.name}/pull/${pull.number}.patch`,
        issue_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${pull.number}`,
        commits_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/pulls/${pull.number}/commits`,
        review_comments_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/pulls/${pull.number}/comments`,
        comments_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/issues/${pull.number}/comments`,
        statuses_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/statuses/${headSha}`,
      };
    };

    if (rest === '/pulls') {
      if (method === 'GET') {
        const state = url.searchParams.get('state') ?? 'open';
        let pulls = [...repo.items.values()].filter((item): item is PullRecord => item.kind === 'pull');
        if (state === 'open' || state === 'closed') pulls = pulls.filter((pull) => pull.state === state);
        pulls.sort((a, b) => b.number - a.number);
        sendPage(ctx, res, url, pulls, (pull) => pullJson(pull));
        return;
      }
      if (method === 'POST') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        const title = typeof body.value.title === 'string' ? body.value.title : '';
        const head = typeof body.value.head === 'string' ? body.value.head.replace(/^[^:]+:/, '') : '';
        const base = typeof body.value.base === 'string' ? body.value.base.replace(/^[^:]+:/, '') : '';
        const headSha = repo.refs.get(`refs/heads/${head}`);
        const baseSha = repo.refs.get(`refs/heads/${base}`);
        if (title === '' || !headSha || !baseSha || head === base) {
          sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'PullRequest', code: 'custom', message: 'title, head and base must name two different existing branches' }] });
          return;
        }
        const duplicate = [...repo.items.values()].some(
          (item) => item.kind === 'pull' && item.state === 'open' && item.headRef === head && item.baseRef === base,
        );
        if (duplicate) {
          sendJson(res, 422, { message: 'Validation Failed', errors: [{ resource: 'PullRequest', code: 'custom', message: 'A pull request already exists for this head and base' }] });
          return;
        }
        const now = store.clock.tick();
        const pull: PullRecord = {
          kind: 'pull',
          number: repo.nextItemNumber,
          title,
          body: typeof body.value.body === 'string' ? body.value.body : '',
          state: 'open',
          labels: [],
          comments: [],
          author: viewer.login,
          createdAt: now,
          updatedAt: now,
          headRef: head,
          baseRef: base,
          headSha,
          baseSha,
          merged: false,
          mergedAt: null,
          mergeCommitSha: null,
        };
        repo.nextItemNumber += 1;
        repo.items.set(pull.number, pull);
        sendJson(res, 201, pullJson(pull));
        return;
      }
    }

    const pullMatch = /^\/pulls\/(\d+)(\/files|\/merge)?$/.exec(rest);
    if (pullMatch) {
      const number = Number(pullMatch[1]);
      const item = repo.items.get(number);
      const pull = item && item.kind === 'pull' ? item : undefined;
      if (!pull) {
        sendError(res, 404, 'Not Found');
        return;
      }
      if (pullMatch[2] === '/files' && method === 'GET') {
        const headSha = repo.refs.get(`refs/heads/${pull.headRef}`) ?? pull.headSha;
        const baseSha = repo.refs.get(`refs/heads/${pull.baseRef}`) ?? pull.baseSha;
        const base = mergeBase(repo.objects, baseSha, headSha);
        const files = diffFiles(repo.objects, commitTree(repo.objects, base), commitTree(repo.objects, headSha));
        sendPage(ctx, res, url, files, (file) => ({
          sha: file.sha,
          filename: file.filename,
          status: file.status,
          additions: file.additions,
          deletions: file.deletions,
          changes: file.changes,
          blob_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/git/blobs/${file.sha ?? ''}`,
          raw_url: `${ctx.base}/${repo.owner}/${repo.name}/raw/${headSha}/${file.filename}`,
          contents_url: `${ctx.base}/api/v3/repos/${repo.owner}/${repo.name}/contents/${file.filename}?ref=${headSha}`,
        }));
        return;
      }
      if (pullMatch[2] === '/merge' && method === 'PUT') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        if (pull.state !== 'open' || pull.merged) {
          sendJson(res, 405, { message: 'Pull Request is not mergeable', documentation_url: 'https://docs.github.com/rest/pulls/pulls#merge-a-pull-request' });
          return;
        }
        const headSha = repo.refs.get(`refs/heads/${pull.headRef}`) ?? pull.headSha;
        const baseRef = `refs/heads/${pull.baseRef}`;
        const baseSha = repo.refs.get(baseRef) ?? pull.baseSha;
        const base = mergeBase(repo.objects, baseSha, headSha);
        if (base === headSha) {
          sendJson(res, 405, { message: 'Pull Request is not mergeable', documentation_url: 'https://docs.github.com/rest/pulls/pulls#merge-a-pull-request' });
          return;
        }
        const outcome = threeWayMerge(
          repo.objects,
          commitTree(repo.objects, base),
          commitTree(repo.objects, baseSha),
          commitTree(repo.objects, headSha),
        );
        if (!outcome.ok) {
          sendJson(res, 405, { message: 'Pull Request is not mergeable due to conflicts', documentation_url: 'https://docs.github.com/rest/pulls/pulls#merge-a-pull-request' });
          return;
        }
        const message = `${pull.title} (#${pull.number})${pull.body ? `\n\n${pull.body}` : ''}`;
        const commit = makeCommit(ctx, repo, baseRef, outcome.tree, message);
        pull.merged = true;
        pull.mergedAt = commit.at;
        pull.mergeCommitSha = commit.sha;
        pull.state = 'closed';
        pull.updatedAt = commit.at;
        sendJson(res, 200, { sha: commit.sha, merged: true, message: 'Pull Request successfully merged' });
        return;
      }
      if (method === 'GET') {
        sendJson(res, 200, pullJson(pull));
        return;
      }
      if (method === 'PATCH') {
        const body = await readJsonBody(req);
        if (!body.ok) {
          sendError(res, 400, 'Problems parsing JSON');
          return;
        }
        if (typeof body.value.title === 'string') pull.title = body.value.title;
        if (typeof body.value.body === 'string') pull.body = body.value.body;
        if ((body.value.state === 'open' || body.value.state === 'closed') && !pull.merged) pull.state = body.value.state;
        pull.updatedAt = store.clock.tick();
        sendJson(res, 200, pullJson(pull));
        return;
      }
    }

    // Deeper REST routes (git trees) land in the next milestone.
    sendError(res, 404, 'Not Found');
    return;
  }

  sendError(res, 404, 'Not Found');
}

async function handleGit(ctx: Ctx, req: IncomingMessage, res: ServerResponse, url: URL): Promise<boolean> {
  const match = /^\/([^/]+)\/([^/]+?)(\.git)?\/(info\/refs|git-upload-pack|git-receive-pack)$/.exec(url.pathname);
  if (!match) return false;
  const owner = decodeURIComponent(match[1] as string);
  const name = decodeURIComponent(match[2] as string);
  const action = match[4] as string;
  if (!authorized(ctx, req)) {
    sendError(res, 401, 'Bad credentials');
    return true;
  }
  const repo = owner === ctx.store.viewer.login ? ctx.store.getRepo(owner, name) : undefined;
  if (!repo) {
    sendError(res, 404, 'Not Found');
    return true;
  }
  try {
    if (action === 'info/refs') {
      const service = url.searchParams.get('service');
      if (service !== 'git-upload-pack' && service !== 'git-receive-pack') {
        sendError(res, 403, 'Invalid service');
        return true;
      }
      const version2 = (req.headers['git-protocol'] ?? '').toString().includes('version=2');
      const body = advertiseRefs(repo, service, version2);
      res.writeHead(200, {
        'content-type': `application/x-${service}-advertisement`,
        'cache-control': 'no-cache',
        'x-github-request-id': REQUEST_ID,
      });
      res.end(body);
      return true;
    }
    const body = await readBody(req);
    if (action === 'git-upload-pack') {
      const out = handleUploadPack(repo, body);
      res.writeHead(200, { 'content-type': 'application/x-git-upload-pack-result', 'cache-control': 'no-cache' });
      res.end(out);
      return true;
    }
    const out = handleReceivePack(repo, body);
    res.writeHead(200, { 'content-type': 'application/x-git-receive-pack-result', 'cache-control': 'no-cache' });
    res.end(out);
    return true;
  } catch {
    // A malformed git request must never take the server down (repo-survives-errors).
    if (!res.headersSent) sendError(res, 400, 'Invalid git request');
    else res.end();
    return true;
  }
}

/** The entry main.ts runs: one HTTP server carrying every surface on the shared store. */
export function createApp(): Server {
  const store = new ForgeStore();
  const token = process.env.MAKEFUN_SHOP_TOKEN ?? null;
  return createHttpServer((req, res) => {
    void (async () => {
      try {
        const host = req.headers.host ?? '127.0.0.1';
        const url = new URL(req.url ?? '/', `http://${host}`);
        const ctx: Ctx = { store, base: `http://${host}`, token };
        if (url.pathname === '/healthz') {
          res.writeHead(200, { 'content-type': 'text/plain' });
          res.end('ok');
          return;
        }
        if (await handleGit(ctx, req, res, url)) return;
        if (url.pathname === '/api/graphql') {
          await handleGraphQL(ctx.store, req, res, authorized(ctx, req));
          return;
        }
        if (url.pathname === '/mcp') {
          await handleMcp(ctx.store, req, res, authorized(ctx, req));
          return;
        }
        if (url.pathname.startsWith('/api/v3')) {
          await handleRest(ctx, req, res, url);
          return;
        }
        sendError(res, 404, 'Not Found');
      } catch {
        if (!res.headersSent) sendError(res, 500, 'Internal Server Error');
        else res.end();
      }
    })();
  });
}
