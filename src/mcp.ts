// mcp.ts: the Model Context Protocol surface — POST /mcp speaking JSON-RPC 2.0 over the Streamable HTTP
// transport, in both protocol eras the conformance client negotiates:
//
//   modern (2026-07-28): no handshake. `server/discover` answers the server's identity, capabilities and
//     supportedVersions in one request; every later POST is answered by one JSON body.
//   legacy (2025-11-25 and earlier): `initialize` echoes the revision the client offered, then
//     `notifications/initialized` completes the handshake; no session id is handed out (stateless server).
//
// Six tools mirror the REST surface over the same ForgeStore, so an object created over MCP reads back
// identically over REST and vice versa: create_repo, get_repo, create_issue, list_issues,
// create_pull_request, merge_pull_request. The tools array is one frozen module constant, serialized
// identically on every tools/list call. Errors follow JSON-RPC: an unparseable body is HTTP 400 with
// -32700, an unknown tool or missing required arguments answer -32602, an unknown method -32601, and a
// request without the bearer token is refused with HTTP 401 before any parsing happens.
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { ForgeStore, RepoRecord, IssueRecord, PullRecord } from './store.ts';
import { mergeBase, threeWayMerge, commitTree } from './pulls.ts';
import { commitOnBranch, repoGlobalId, issueGlobalId, pullGlobalId } from './graphql.ts';

/** The server's identity, reported identically by server/discover and initialize. */
const SERVER_INFO = { name: 'github-core-clone', version: '1.0.0' } as const;

/** Every revision this server speaks, newest first; the first is the current one. */
const SUPPORTED_VERSIONS = ['2026-07-28', '2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'] as const;

/** The capabilities both eras report: tools only, no list-change notifications (the list never changes). */
const CAPABILITIES = { tools: { listChanged: false } } as const;

/** One tool definition as tools/list serves it. */
interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, { type: string; description: string }>;
    required: string[];
  };
}

const STRING = (description: string): { type: string; description: string } => ({ type: 'string', description });

/**
 * The six tools, in a stable order, as one frozen constant: tools/list returns this exact array every
 * time, so two calls are byte-identical by construction. Required lists stay minimal — exactly the
 * arguments a call cannot proceed without.
 */
export const TOOLS: readonly ToolDefinition[] = Object.freeze([
  {
    name: 'create_repo',
    description: 'Create a new repository owned by the authenticated user and return it.',
    inputSchema: {
      type: 'object',
      properties: {
        name: STRING('The repository name.'),
        description: STRING('A short description of the repository.'),
        private: { type: 'boolean', description: 'Whether the repository is private.' },
      },
      required: ['name'],
    },
  },
  {
    name: 'get_repo',
    description: 'Get a repository by its owner login and name.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: STRING('The repository owner login.'),
        repo: STRING('The repository name.'),
      },
      required: ['owner', 'repo'],
    },
  },
  {
    name: 'create_issue',
    description: 'Create an issue in a repository and return it.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: STRING('The repository owner login.'),
        repo: STRING('The repository name.'),
        title: STRING('The issue title.'),
        body: STRING('The issue body.'),
      },
      required: ['owner', 'repo', 'title'],
    },
  },
  {
    name: 'list_issues',
    description: 'List the issues of a repository, optionally filtered by state (open or closed).',
    inputSchema: {
      type: 'object',
      properties: {
        owner: STRING('The repository owner login.'),
        repo: STRING('The repository name.'),
        state: STRING('Filter by issue state: open or closed.'),
      },
      required: ['owner', 'repo'],
    },
  },
  {
    name: 'create_pull_request',
    description: 'Create a pull request proposing the head branch into the base branch and return it.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: STRING('The repository owner login.'),
        repo: STRING('The repository name.'),
        title: STRING('The pull request title.'),
        body: STRING('The pull request body.'),
        head: STRING('The branch the changes come from.'),
        base: STRING('The branch the changes go into.'),
      },
      required: ['owner', 'repo', 'title', 'head', 'base'],
    },
  },
  {
    name: 'merge_pull_request',
    description: 'Merge an open pull request with a merge commit and return it.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: STRING('The repository owner login.'),
        repo: STRING('The repository name.'),
        pull_number: { type: 'number', description: 'The pull request number.' },
      },
      required: ['owner', 'repo', 'pull_number'],
    },
  },
]);

/** The REST-shaped repository object a tool result carries. */
function repoObject(store: ForgeStore, repo: RepoRecord): Record<string, unknown> {
  return {
    id: repo.id,
    node_id: repoGlobalId(repo),
    name: repo.name,
    full_name: `${store.viewer.login}/${repo.name}`,
    private: repo.isPrivate,
    description: repo.description,
    default_branch: repo.headRef.replace('refs/heads/', ''),
    owner: { login: store.viewer.login, id: store.viewer.id },
  };
}

/** The REST-shaped issue object (issues and pull requests share the numbering sequence). */
function issueObject(store: ForgeStore, repo: RepoRecord, issue: IssueRecord): Record<string, unknown> {
  return {
    id: repo.id * 1_000_000 + issue.number,
    node_id: issueGlobalId(repo, issue),
    number: issue.number,
    title: issue.title,
    body: issue.body,
    state: issue.state,
    user: { login: store.viewer.login, id: store.viewer.id },
    created_at: issue.createdAt,
    updated_at: issue.updatedAt,
  };
}

/** The REST-shaped pull request object a tool result carries. */
function pullObject(store: ForgeStore, repo: RepoRecord, pull: PullRecord): Record<string, unknown> {
  return {
    id: repo.id * 1_000_000 + pull.number,
    node_id: pullGlobalId(repo, pull),
    number: pull.number,
    title: pull.title,
    body: pull.body,
    state: pull.state,
    merged: pull.merged,
    merged_at: pull.mergedAt,
    head: { ref: pull.headRef },
    base: { ref: pull.baseRef },
    user: { login: store.viewer.login, id: store.viewer.id },
    created_at: pull.createdAt,
    updated_at: pull.updatedAt,
  };
}

/** A tool-level failure: the call was well-formed, the tool could not do the work (isError result). */
class ToolFailure extends Error {}

/** What running one tool produced, before it is framed as a JSON-RPC answer. */
type ToolOutcome =
  | { kind: 'result'; value: unknown }
  | { kind: 'tool-error'; message: string }
  | { kind: 'invalid-params'; message: string };

/** A required string argument, already presence-checked by the dispatcher. */
function argString(args: Record<string, unknown>, key: string): string {
  const value = args[key];
  return typeof value === 'string' ? value : String(value ?? '');
}

/** The repository a tool call names; only the viewer's own repositories are visible over MCP. */
function findRepo(store: ForgeStore, args: Record<string, unknown>): RepoRecord {
  const owner = argString(args, 'owner');
  const name = argString(args, 'repo');
  if (owner !== store.viewer.login) throw new ToolFailure(`repository ${owner}/${name} not found`);
  const repo = store.getRepo(store.viewer.login, name);
  if (repo === undefined) throw new ToolFailure(`repository ${owner}/${name} not found`);
  return repo;
}

/** create_repo: an idempotent create — naming an existing repository returns it unchanged. */
function toolCreateRepo(store: ForgeStore, args: Record<string, unknown>): Record<string, unknown> {
  const name = argString(args, 'name');
  const existing = store.getRepo(store.viewer.login, name);
  if (existing !== undefined) return repoObject(store, existing);
  const repo = store.createRepo({
    owner: store.viewer.login,
    name,
    description: typeof args.description === 'string' ? args.description : undefined,
    isPrivate: typeof args.private === 'boolean' ? args.private : undefined,
  });
  return repoObject(store, repo);
}

/** create_issue: one clock tick, the next shared number, state open (the REST/GraphQL record shape). */
function toolCreateIssue(store: ForgeStore, args: Record<string, unknown>): Record<string, unknown> {
  const repo = findRepo(store, args);
  const now = store.clock.tick();
  const issue: IssueRecord = {
    kind: 'issue',
    number: repo.nextItemNumber,
    title: argString(args, 'title'),
    body: typeof args.body === 'string' ? args.body : '',
    state: 'open',
    labels: [],
    comments: [],
    author: store.viewer.login,
    createdAt: now,
    updatedAt: now,
  };
  repo.nextItemNumber += 1;
  repo.items.set(issue.number, issue);
  return issueObject(store, repo, issue);
}

/** list_issues: issues only (never pull requests), newest first, optionally filtered by state. */
function toolListIssues(store: ForgeStore, args: Record<string, unknown>): unknown {
  const repo = findRepo(store, args);
  const state = typeof args.state === 'string' ? args.state : undefined;
  const issues = [...repo.items.values()]
    .filter((item) => item.kind === 'issue')
    .filter((item) => state === undefined || item.state === state)
    .sort((a, b) => b.number - a.number)
    .map((item) => issueObject(store, repo, item));
  return issues;
}

/** create_pull_request: the REST/GraphQL validation — both branches exist and differ, no open duplicate. */
function toolCreatePullRequest(store: ForgeStore, args: Record<string, unknown>): Record<string, unknown> {
  const repo = findRepo(store, args);
  const head = argString(args, 'head');
  const base = argString(args, 'base');
  const headSha = repo.refs.get(`refs/heads/${head}`);
  const baseSha = repo.refs.get(`refs/heads/${base}`);
  if (headSha === undefined) throw new ToolFailure(`head branch ${head} does not exist`);
  if (baseSha === undefined) throw new ToolFailure(`base branch ${base} does not exist`);
  if (head === base) throw new ToolFailure('head and base must differ');
  for (const item of repo.items.values()) {
    if (item.kind === 'pull' && item.state === 'open' && item.headRef === head && item.baseRef === base) {
      throw new ToolFailure('a pull request already exists for this head and base');
    }
  }
  const now = store.clock.tick();
  const pull: PullRecord = {
    kind: 'pull',
    number: repo.nextItemNumber,
    title: argString(args, 'title'),
    body: typeof args.body === 'string' ? args.body : '',
    state: 'open',
    labels: [],
    comments: [],
    author: store.viewer.login,
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
  return pullObject(store, repo, pull);
}

/** merge_pull_request: the GraphQL merge — three-way merge over trees, one commit on the base branch. */
function toolMergePullRequest(store: ForgeStore, args: Record<string, unknown>): Record<string, unknown> {
  const repo = findRepo(store, args);
  const number = typeof args.pull_number === 'number' ? args.pull_number : Number(args.pull_number);
  const item = Number.isInteger(number) ? repo.items.get(number) : undefined;
  if (item === undefined || item.kind !== 'pull') throw new ToolFailure(`pull request ${String(args.pull_number)} not found`);
  const pull = item as PullRecord;
  if (pull.state !== 'open' || pull.merged) throw new ToolFailure('pull request is not open');
  const headSha = repo.refs.get(`refs/heads/${pull.headRef}`) ?? pull.headSha;
  const baseSha = repo.refs.get(`refs/heads/${pull.baseRef}`) ?? pull.baseSha;
  const base = mergeBase(repo.objects, baseSha, headSha);
  if (base === null || base === headSha) throw new ToolFailure('pull request is not mergeable');
  const outcome = threeWayMerge(repo.objects, commitTree(repo.objects, base), commitTree(repo.objects, baseSha), commitTree(repo.objects, headSha));
  if (!outcome.ok) throw new ToolFailure('pull request is not mergeable due to conflicts');
  const headline = `${pull.title} (#${pull.number})`;
  const message = pull.body ? `${headline}\n\n${pull.body}` : headline;
  const commit = commitOnBranch(store, repo, `refs/heads/${pull.baseRef}`, outcome.tree, message);
  pull.merged = true;
  pull.mergedAt = commit.at;
  pull.mergeCommitSha = commit.sha;
  pull.state = 'closed';
  pull.updatedAt = commit.at;
  return pullObject(store, repo, pull);
}

/**
 * Run one tools/call: an unknown tool name and missing required arguments are protocol errors
 * (-32602, decided by the caller), everything the tool itself refuses is a tool-error result.
 */
function runTool(store: ForgeStore, name: string, rawArgs: unknown): ToolOutcome {
  const definition = TOOLS.find((tool) => tool.name === name);
  if (definition === undefined) return { kind: 'invalid-params', message: `unknown tool: ${name}` };
  const args: Record<string, unknown> = typeof rawArgs === 'object' && rawArgs !== null && !Array.isArray(rawArgs) ? (rawArgs as Record<string, unknown>) : {};
  const missing = definition.inputSchema.required.filter((key) => args[key] === undefined || args[key] === null || args[key] === '');
  if (missing.length > 0) return { kind: 'invalid-params', message: `missing required arguments: ${missing.join(', ')}` };
  try {
    switch (name) {
      case 'create_repo':
        return { kind: 'result', value: toolCreateRepo(store, args) };
      case 'get_repo':
        return { kind: 'result', value: repoObject(store, findRepo(store, args)) };
      case 'create_issue':
        return { kind: 'result', value: toolCreateIssue(store, args) };
      case 'list_issues':
        return { kind: 'result', value: toolListIssues(store, args) };
      case 'create_pull_request':
        return { kind: 'result', value: toolCreatePullRequest(store, args) };
      case 'merge_pull_request':
        return { kind: 'result', value: toolMergePullRequest(store, args) };
      default:
        return { kind: 'invalid-params', message: `unknown tool: ${name}` };
    }
  } catch (error) {
    if (error instanceof ToolFailure) return { kind: 'tool-error', message: error.message };
    throw error;
  }
}

// ------------------------------------------------------------------------------------------------
// JSON-RPC framing and the HTTP handler.
// ------------------------------------------------------------------------------------------------

/** A success envelope. */
function rpcResult(id: unknown, result: unknown): Record<string, unknown> {
  return { jsonrpc: '2.0', id: id ?? null, result };
}

/** An error envelope. */
function rpcError(id: unknown, code: number, message: string): Record<string, unknown> {
  return { jsonrpc: '2.0', id: id ?? null, error: { code, message } };
}

/** The server/discover answer: identity, capabilities and every supported revision in one result. */
function discoverResult(): Record<string, unknown> {
  return {
    resultType: 'complete',
    supportedVersions: [...SUPPORTED_VERSIONS],
    capabilities: CAPABILITIES,
    serverInfo: { ...SERVER_INFO },
    _meta: { 'io.modelcontextprotocol/serverInfo': { ...SERVER_INFO } },
  };
}

/** The legacy initialize answer: echo the offered revision when spoken here, else the newest legacy one. */
function initializeResult(params: Record<string, unknown>): Record<string, unknown> {
  const offered = typeof params.protocolVersion === 'string' ? params.protocolVersion : '';
  const protocolVersion = (SUPPORTED_VERSIONS as readonly string[]).includes(offered) ? offered : '2025-11-25';
  return { protocolVersion, capabilities: CAPABILITIES, serverInfo: { ...SERVER_INFO } };
}

/** Frame a tool outcome as a tools/call result: text block plus structured content for object values. */
function toolCallResult(outcome: Extract<ToolOutcome, { kind: 'result' | 'tool-error' }>): Record<string, unknown> {
  if (outcome.kind === 'tool-error') {
    return { content: [{ type: 'text', text: outcome.message }], isError: true };
  }
  const result: Record<string, unknown> = {
    content: [{ type: 'text', text: JSON.stringify(outcome.value) }],
    isError: false,
  };
  if (typeof outcome.value === 'object' && outcome.value !== null && !Array.isArray(outcome.value)) {
    result.structuredContent = outcome.value;
  }
  return result;
}

/**
 * Dispatch one JSON-RPC request object. Returns the response object, or null for a notification
 * (a request object without an id), which is answered by the HTTP status alone.
 */
function dispatch(store: ForgeStore, request: Record<string, unknown>): Record<string, unknown> | null {
  const isNotification = request.id === undefined;
  const id = request.id ?? null;
  const method = typeof request.method === 'string' ? request.method : '';
  const params: Record<string, unknown> =
    typeof request.params === 'object' && request.params !== null && !Array.isArray(request.params) ? (request.params as Record<string, unknown>) : {};
  switch (method) {
    case 'server/discover':
      return rpcResult(id, discoverResult());
    case 'initialize':
      return rpcResult(id, initializeResult(params));
    case 'ping':
      return rpcResult(id, {});
    case 'tools/list':
      return rpcResult(id, { tools: TOOLS });
    case 'tools/call': {
      const name = typeof params.name === 'string' ? params.name : '';
      const outcome = runTool(store, name, params.arguments);
      if (outcome.kind === 'invalid-params') return rpcError(id, -32602, outcome.message);
      return rpcResult(id, toolCallResult(outcome));
    }
    default:
      return isNotification ? null : rpcError(id, -32601, `Method not found: ${method}`);
  }
}

/** Write one JSON body with the given status. */
function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

/**
 * POST /mcp: the bearer token is required before any parsing (a tokenless call is HTTP 401 even for a
 * malformed body); a body that is not JSON is HTTP 400 with the JSON-RPC parse error -32700; every
 * other request is dispatched and answered with one JSON body (notifications get HTTP 202, no body).
 */
export async function handleMcp(store: ForgeStore, req: IncomingMessage, res: ServerResponse, isAuthorized: boolean): Promise<void> {
  if (!isAuthorized) {
    sendJson(res, 401, { message: 'Bad credentials', documentation_url: 'https://docs.github.com/rest' });
    return;
  }
  if (req.method !== 'POST') {
    sendJson(res, 405, rpcError(null, -32601, 'Method not allowed: the MCP endpoint only accepts POST'));
    return;
  }
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString('utf8');
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    sendJson(res, 400, rpcError(null, -32700, 'Parse error: the request body is not valid JSON'));
    return;
  }
  if (Array.isArray(parsed)) {
    const responses = parsed
      .map((entry) => (typeof entry === 'object' && entry !== null ? dispatch(store, entry as Record<string, unknown>) : rpcError(null, -32600, 'Invalid Request')))
      .filter((entry) => entry !== null);
    sendJson(res, 200, responses);
    return;
  }
  if (typeof parsed !== 'object' || parsed === null) {
    sendJson(res, 400, rpcError(null, -32600, 'Invalid Request: expected a JSON-RPC request object'));
    return;
  }
  const response = dispatch(store, parsed as Record<string, unknown>);
  if (response === null) {
    res.writeHead(202);
    res.end();
    return;
  }
  sendJson(res, 200, response);
}

