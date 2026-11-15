// The GraphQL v4 surface: a small self-contained engine (lexer, parser, schema, executor)
// serving POST /api/graphql from the same in-memory store the git and REST surfaces use, so
// an object created through one surface reads back identically through the others. The schema
// describes exactly the documents gh and github-mcp-server send: viewer and repository reads,
// the __type feature-detection probes, issue and pull-request connections with cursor
// pagination, and the create/close/merge/comment mutations. Global ids follow the REST
// node_id convention (base64 of a typed, numbered payload), so an id minted by a REST read
// decodes here and vice versa. An unknown field is a 200 with an errors array naming it,
// never an HTTP error; an unauthenticated request is a 401 before any parsing happens.

import type { IncomingMessage, ServerResponse } from 'node:http';
import { ForgeStore, type CommentRecord, type IssueRecord, type PullRecord, type RepoRecord } from './store.ts';
import { commitTree, diffFiles, mergeBase, threeWayMerge } from './pulls.ts';
import { serializeCommit } from './git/objects.ts';

// ------------------------------------------------------------------------------------------------
// Lexer.
// ------------------------------------------------------------------------------------------------

type Token =
  | { readonly kind: 'name'; readonly value: string }
  | { readonly kind: 'string'; readonly value: string }
  | { readonly kind: 'number'; readonly value: string }
  | { readonly kind: 'punct'; readonly value: string }
  | { readonly kind: 'eof' };

class GQLSyntaxError extends Error {}

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const isNameStart = (c: string): boolean => /[A-Za-z_]/.test(c);
  const isNameChar = (c: string): boolean => /[A-Za-z0-9_]/.test(c);
  while (i < source.length) {
    const c = source[i] as string;
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r' || c === ',' || c === '﻿') {
      i += 1;
      continue;
    }
    if (c === '#') {
      while (i < source.length && source[i] !== '\n') i += 1;
      continue;
    }
    if (source.startsWith('...', i)) {
      tokens.push({ kind: 'punct', value: '...' });
      i += 3;
      continue;
    }
    if ('!$():=@[]{}|&'.includes(c)) {
      tokens.push({ kind: 'punct', value: c });
      i += 1;
      continue;
    }
    if (c === '"') {
      if (source.startsWith('"""', i)) {
        const end = source.indexOf('"""', i + 3);
        if (end < 0) throw new GQLSyntaxError('Unterminated block string');
        const raw = source.slice(i + 3, end);
        const lines = raw.split('\n');
        while (lines.length > 0 && (lines[0] as string).trim() === '') lines.shift();
        while (lines.length > 0 && (lines[lines.length - 1] as string).trim() === '') lines.pop();
        let indent = Infinity;
        for (const line of lines.slice(1)) {
          if (line.trim() === '') continue;
          const lead = line.length - line.replace(/^ +/, '').length;
          if (lead < indent) indent = lead;
        }
        if (!Number.isFinite(indent)) indent = 0;
        const value = lines.map((line, index) => (index === 0 ? line : line.slice(indent))).join('\n');
        tokens.push({ kind: 'string', value });
        i = end + 3;
        continue;
      }
      let j = i + 1;
      let value = '';
      for (;;) {
        if (j >= source.length) throw new GQLSyntaxError('Unterminated string');
        const ch = source[j] as string;
        if (ch === '"') break;
        if (ch === '\\') {
          const esc = source[j + 1] as string;
          if (esc === 'u') {
            value += String.fromCharCode(parseInt(source.slice(j + 2, j + 6), 16));
            j += 6;
            continue;
          }
          const map: Record<string, string> = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
          const replacement = map[esc];
          if (replacement === undefined) throw new GQLSyntaxError(`Invalid escape sequence \\${esc}`);
          value += replacement;
          j += 2;
          continue;
        }
        value += ch;
        j += 1;
      }
      tokens.push({ kind: 'string', value });
      i = j + 1;
      continue;
    }
    if (c === '-' || /[0-9]/.test(c)) {
      const match = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/.exec(source.slice(i));
      if (!match) throw new GQLSyntaxError(`Invalid number at offset ${i}`);
      tokens.push({ kind: 'number', value: match[0] });
      i += match[0].length;
      continue;
    }
    if (isNameStart(c)) {
      let j = i + 1;
      while (j < source.length && isNameChar(source[j] as string)) j += 1;
      tokens.push({ kind: 'name', value: source.slice(i, j) });
      i = j;
      continue;
    }
    throw new GQLSyntaxError(`Unexpected character ${JSON.stringify(c)} at offset ${i}`);
  }
  tokens.push({ kind: 'eof' });
  return tokens;
}

// ------------------------------------------------------------------------------------------------
// Parser: documents, operations, fragments, fields, arguments and values.
// ------------------------------------------------------------------------------------------------

type ValueNode =
  | { readonly kind: 'variable'; readonly name: string }
  | { readonly kind: 'string'; readonly value: string }
  | { readonly kind: 'number'; readonly value: number }
  | { readonly kind: 'boolean'; readonly value: boolean }
  | { readonly kind: 'enum'; readonly value: string }
  | { readonly kind: 'null' }
  | { readonly kind: 'list'; readonly values: readonly ValueNode[] }
  | { readonly kind: 'object'; readonly fields: readonly { readonly name: string; readonly value: ValueNode }[] };

interface ArgumentNode {
  readonly name: string;
  readonly value: ValueNode;
}

interface FieldNode {
  readonly kind: 'field';
  readonly alias: string | null;
  readonly name: string;
  readonly args: readonly ArgumentNode[];
  readonly selectionSet: readonly SelectionNode[] | null;
}

interface InlineFragmentNode {
  readonly kind: 'inline';
  readonly typeCondition: string | null;
  readonly selectionSet: readonly SelectionNode[];
}

interface FragmentSpreadNode {
  readonly kind: 'spread';
  readonly name: string;
}

type SelectionNode = FieldNode | InlineFragmentNode | FragmentSpreadNode;

interface OperationNode {
  readonly operation: 'query' | 'mutation';
  readonly name: string | null;
  readonly selectionSet: readonly SelectionNode[];
}

interface FragmentNode {
  readonly name: string;
  readonly typeCondition: string;
  readonly selectionSet: readonly SelectionNode[];
}

interface DocumentNode {
  readonly operations: readonly OperationNode[];
  readonly fragments: ReadonlyMap<string, FragmentNode>;
}

class Parser {
  readonly #tokens: Token[];
  #pos = 0;

  constructor(source: string) {
    this.#tokens = tokenize(source);
  }

  #peek(): Token {
    return this.#tokens[this.#pos] as Token;
  }

  #next(): Token {
    const token = this.#tokens[this.#pos] as Token;
    this.#pos += 1;
    return token;
  }

  #atPunct(value: string): boolean {
    const token = this.#peek();
    return token.kind === 'punct' && token.value === value;
  }

  #eatPunct(value: string): boolean {
    if (!this.#atPunct(value)) return false;
    this.#pos += 1;
    return true;
  }

  #expectPunct(value: string): void {
    if (!this.#eatPunct(value)) throw new GQLSyntaxError(`Expected "${value}", found ${this.#describe()}`);
  }

  #expectName(): string {
    const token = this.#next();
    if (token.kind !== 'name') throw new GQLSyntaxError(`Expected a name, found ${this.#describe()}`);
    return token.value;
  }

  #describe(): string {
    const token = this.#peek();
    return token.kind === 'eof' ? 'end of document' : JSON.stringify(token.value);
  }

  parseDocument(): DocumentNode {
    const operations: OperationNode[] = [];
    const fragments = new Map<string, FragmentNode>();
    while (this.#peek().kind !== 'eof') {
      if (this.#atPunct('{')) {
        operations.push({ operation: 'query', name: null, selectionSet: this.#parseSelectionSet() });
        continue;
      }
      const keyword = this.#expectName();
      if (keyword === 'query' || keyword === 'mutation' || keyword === 'subscription') {
        let name: string | null = null;
        if (this.#peek().kind === 'name') name = this.#expectName();
        if (this.#atPunct('(')) this.#parseVariableDefinitions();
        this.#parseDirectives();
        operations.push({
          operation: keyword === 'mutation' ? 'mutation' : 'query',
          name,
          selectionSet: this.#parseSelectionSet(),
        });
        continue;
      }
      if (keyword === 'fragment') {
        const name = this.#expectName();
        const on = this.#expectName();
        if (on !== 'on') throw new GQLSyntaxError(`Expected "on" in fragment ${name}`);
        const typeCondition = this.#expectName();
        this.#parseDirectives();
        fragments.set(name, { name, typeCondition, selectionSet: this.#parseSelectionSet() });
        continue;
      }
      throw new GQLSyntaxError(`Unexpected definition "${keyword}"`);
    }
    if (operations.length === 0) throw new GQLSyntaxError('The document holds no operation');
    return { operations, fragments };
  }

  #parseVariableDefinitions(): void {
    this.#expectPunct('(');
    while (!this.#eatPunct(')')) {
      this.#expectPunct('$');
      this.#expectName();
      this.#expectPunct(':');
      this.#parseType();
      if (this.#eatPunct('=')) this.#parseValue();
      this.#parseDirectives();
    }
  }

  #parseType(): void {
    if (this.#eatPunct('[')) {
      this.#parseType();
      this.#expectPunct(']');
    } else {
      this.#expectName();
    }
    this.#eatPunct('!');
  }

  #parseDirectives(): void {
    while (this.#eatPunct('@')) {
      this.#expectName();
      if (this.#atPunct('(')) this.#parseArguments();
    }
  }

  #parseArguments(): ArgumentNode[] {
    const args: ArgumentNode[] = [];
    this.#expectPunct('(');
    while (!this.#eatPunct(')')) {
      const name = this.#expectName();
      this.#expectPunct(':');
      args.push({ name, value: this.#parseValue() });
    }
    return args;
  }

  #parseValue(): ValueNode {
    const token = this.#peek();
    if (token.kind === 'punct' && token.value === '$') {
      this.#next();
      return { kind: 'variable', name: this.#expectName() };
    }
    if (token.kind === 'string') {
      this.#next();
      return { kind: 'string', value: token.value };
    }
    if (token.kind === 'number') {
      this.#next();
      return { kind: 'number', value: Number(token.value) };
    }
    if (token.kind === 'punct' && token.value === '[') {
      this.#next();
      const values: ValueNode[] = [];
      while (!this.#eatPunct(']')) values.push(this.#parseValue());
      return { kind: 'list', values };
    }
    if (token.kind === 'punct' && token.value === '{') {
      this.#next();
      const fields: { name: string; value: ValueNode }[] = [];
      while (!this.#eatPunct('}')) {
        const name = this.#expectName();
        this.#expectPunct(':');
        fields.push({ name, value: this.#parseValue() });
      }
      return { kind: 'object', fields };
    }
    if (token.kind === 'name') {
      this.#next();
      if (token.value === 'true') return { kind: 'boolean', value: true };
      if (token.value === 'false') return { kind: 'boolean', value: false };
      if (token.value === 'null') return { kind: 'null' };
      return { kind: 'enum', value: token.value };
    }
    throw new GQLSyntaxError(`Expected a value, found ${this.#describe()}`);
  }

  #parseSelectionSet(): SelectionNode[] {
    const selections: SelectionNode[] = [];
    this.#expectPunct('{');
    while (!this.#eatPunct('}')) selections.push(this.#parseSelection());
    return selections;
  }

  #parseSelection(): SelectionNode {
    if (this.#eatPunct('...')) {
      if (this.#peek().kind === 'name' && (this.#tokens[this.#pos] as Token).kind === 'name') {
        const next = this.#tokens[this.#pos] as Token;
        if (next.kind === 'name' && next.value === 'on') {
          this.#next();
          const typeCondition = this.#expectName();
          this.#parseDirectives();
          return { kind: 'inline', typeCondition, selectionSet: this.#parseSelectionSet() };
        }
        const name = this.#expectName();
        this.#parseDirectives();
        return { kind: 'spread', name };
      }
      this.#parseDirectives();
      return { kind: 'inline', typeCondition: null, selectionSet: this.#parseSelectionSet() };
    }
    let name = this.#expectName();
    let alias: string | null = null;
    if (this.#eatPunct(':')) {
      alias = name;
      name = this.#expectName();
    }
    const args = this.#atPunct('(') ? this.#parseArguments() : [];
    this.#parseDirectives();
    const selectionSet = this.#atPunct('{') ? this.#parseSelectionSet() : null;
    return { kind: 'field', alias, name, args, selectionSet };
  }
}

// ------------------------------------------------------------------------------------------------
// Schema: the types and fields introspection describes and the executor validates against.
// ------------------------------------------------------------------------------------------------

interface InputValueDef {
  readonly name: string;
  readonly type: string;
  readonly defaultValue?: string;
}

interface FieldDef {
  readonly name: string;
  readonly type: string;
  readonly args?: readonly InputValueDef[];
}

interface TypeDef {
  readonly name: string;
  readonly kind: 'OBJECT' | 'INTERFACE' | 'ENUM' | 'INPUT_OBJECT' | 'SCALAR';
  readonly fields?: readonly FieldDef[];
  readonly enumValues?: readonly string[];
  readonly inputFields?: readonly InputValueDef[];
  readonly possibleTypes?: readonly string[];
}

const SCALARS: readonly TypeDef[] = ['String', 'Int', 'Float', 'Boolean', 'ID', 'DateTime', 'URI'].map((name) => ({
  name,
  kind: 'SCALAR' as const,
}));

const ENUMS: readonly TypeDef[] = [
  { name: 'IssueState', kind: 'ENUM', enumValues: ['OPEN', 'CLOSED'] },
  { name: 'PullRequestState', kind: 'ENUM', enumValues: ['OPEN', 'CLOSED', 'MERGED'] },
  { name: 'MergeableState', kind: 'ENUM', enumValues: ['MERGEABLE', 'CONFLICTING', 'UNKNOWN'] },
  { name: 'PullRequestMergeMethod', kind: 'ENUM', enumValues: ['MERGE', 'SQUASH', 'REBASE'] },
  { name: 'RepositoryPermission', kind: 'ENUM', enumValues: ['ADMIN', 'MAINTAIN', 'WRITE', 'TRIAGE', 'READ'] },
  { name: 'IssueOrderField', kind: 'ENUM', enumValues: ['CREATED_AT', 'UPDATED_AT', 'COMMENTS'] },
  { name: 'OrderDirection', kind: 'ENUM', enumValues: ['ASC', 'DESC'] },
  { name: 'SearchType', kind: 'ENUM', enumValues: ['ISSUE', 'REPOSITORY', 'USER', 'DISCUSSION', 'CODE', 'COMMIT'] },
  { name: 'DiffEntryChangeType', kind: 'ENUM', enumValues: ['ADDED', 'DELETED', 'MODIFIED', 'RENAMED', 'COPIED', 'CHANGED'] },
];

const FIRST_ARG: InputValueDef = { name: 'first', type: 'Int' };
const AFTER_ARG: InputValueDef = { name: 'after', type: 'String' };

const OBJECTS: readonly TypeDef[] = [
  {
    name: 'Query',
    kind: 'OBJECT',
    fields: [
      { name: 'viewer', type: 'User!' },
      { name: 'repository', type: 'Repository', args: [{ name: 'owner', type: 'String!' }, { name: 'name', type: 'String!' }] },
      { name: 'node', type: 'Node', args: [{ name: 'id', type: 'ID!' }] },
      { name: 'nodes', type: '[Node]!', args: [{ name: 'ids', type: '[ID!]!' }] },
      { name: 'user', type: 'User', args: [{ name: 'login', type: 'String!' }] },
    ],
  },
  {
    name: 'Mutation',
    kind: 'OBJECT',
    fields: [
      { name: 'createIssue', type: 'CreateIssuePayload', args: [{ name: 'input', type: 'CreateIssueInput!' }] },
      { name: 'closeIssue', type: 'CloseIssuePayload', args: [{ name: 'input', type: 'CloseIssueInput!' }] },
      { name: 'reopenIssue', type: 'ReopenIssuePayload', args: [{ name: 'input', type: 'ReopenIssueInput!' }] },
      { name: 'createPullRequest', type: 'CreatePullRequestPayload', args: [{ name: 'input', type: 'CreatePullRequestInput!' }] },
      { name: 'mergePullRequest', type: 'MergePullRequestPayload', args: [{ name: 'input', type: 'MergePullRequestInput!' }] },
      { name: 'addComment', type: 'AddCommentPayload', args: [{ name: 'input', type: 'AddCommentInput!' }] },
    ],
  },
  {
    name: 'User',
    kind: 'OBJECT',
    fields: [
      { name: 'id', type: 'ID!' },
      { name: 'login', type: 'String!' },
      { name: 'name', type: 'String' },
      { name: 'email', type: 'String' },
      { name: 'databaseId', type: 'Int' },
      { name: 'avatarUrl', type: 'URI' },
      { name: 'url', type: 'URI' },
    ],
  },
  {
    name: 'Repository',
    kind: 'OBJECT',
    fields: [
      { name: 'id', type: 'ID!' },
      { name: 'databaseId', type: 'Int!' },
      { name: 'name', type: 'String!' },
      { name: 'nameWithOwner', type: 'String!' },
      { name: 'owner', type: 'User!' },
      { name: 'description', type: 'String' },
      { name: 'isPrivate', type: 'Boolean!' },
      { name: 'createdAt', type: 'DateTime!' },
      { name: 'updatedAt', type: 'DateTime!' },
      { name: 'url', type: 'URI!' },
      { name: 'defaultBranchRef', type: 'Ref' },
      { name: 'viewerPermission', type: 'RepositoryPermission' },
      { name: 'hasIssuesEnabled', type: 'Boolean!' },
      { name: 'issue', type: 'Issue', args: [{ name: 'number', type: 'Int!' }] },
      {
        name: 'issues',
        type: 'IssueConnection!',
        args: [
          FIRST_ARG,
          AFTER_ARG,
          { name: 'states', type: '[IssueState!]' },
          { name: 'orderBy', type: 'IssueOrder' },
          { name: 'labels', type: '[String!]' },
        ],
      },
      { name: 'pullRequest', type: 'PullRequest', args: [{ name: 'number', type: 'Int!' }] },
      {
        name: 'pullRequests',
        type: 'PullRequestConnection!',
        args: [FIRST_ARG, AFTER_ARG, { name: 'states', type: '[PullRequestState!]' }],
      },
    ],
  },
  {
    name: 'Ref',
    kind: 'OBJECT',
    fields: [
      { name: 'name', type: 'String!' },
      { name: 'prefix', type: 'String!' },
      { name: 'repository', type: 'Repository!' },
    ],
  },
  {
    name: 'Issue',
    kind: 'OBJECT',
    fields: [
      { name: 'id', type: 'ID!' },
      { name: 'number', type: 'Int!' },
      { name: 'title', type: 'String!' },
      { name: 'state', type: 'IssueState!' },
      { name: 'body', type: 'String' },
      { name: 'author', type: 'User' },
      { name: 'repository', type: 'Repository!' },
      { name: 'createdAt', type: 'DateTime!' },
      { name: 'updatedAt', type: 'DateTime!' },
      { name: 'closedAt', type: 'DateTime' },
      { name: 'url', type: 'URI!' },
      { name: 'labels', type: 'LabelConnection!', args: [FIRST_ARG, AFTER_ARG] },
      { name: 'comments', type: 'IssueCommentConnection!', args: [FIRST_ARG, AFTER_ARG] },
    ],
  },
  {
    name: 'PullRequest',
    kind: 'OBJECT',
    fields: [
      { name: 'id', type: 'ID!' },
      { name: 'number', type: 'Int!' },
      { name: 'title', type: 'String!' },
      { name: 'body', type: 'String' },
      { name: 'state', type: 'PullRequestState!' },
      { name: 'merged', type: 'Boolean!' },
      { name: 'mergeable', type: 'MergeableState!' },
      { name: 'headRefName', type: 'String!' },
      { name: 'baseRefName', type: 'String!' },
      { name: 'headRef', type: 'Ref' },
      { name: 'baseRef', type: 'Ref' },
      { name: 'author', type: 'User' },
      { name: 'repository', type: 'Repository!' },
      { name: 'createdAt', type: 'DateTime!' },
      { name: 'updatedAt', type: 'DateTime!' },
      { name: 'closedAt', type: 'DateTime' },
      { name: 'mergedAt', type: 'DateTime' },
      { name: 'additions', type: 'Int!' },
      { name: 'deletions', type: 'Int!' },
      { name: 'changedFiles', type: 'Int!' },
      { name: 'url', type: 'URI!' },
      { name: 'files', type: 'PullRequestChangedFileConnection', args: [FIRST_ARG, AFTER_ARG] },
      { name: 'comments', type: 'IssueCommentConnection!', args: [FIRST_ARG, AFTER_ARG] },
    ],
  },
  {
    name: 'PullRequestChangedFile',
    kind: 'OBJECT',
    fields: [
      { name: 'path', type: 'String!' },
      { name: 'additions', type: 'Int!' },
      { name: 'deletions', type: 'Int!' },
      { name: 'changeType', type: 'DiffEntryChangeType!' },
    ],
  },
  {
    name: 'IssueComment',
    kind: 'OBJECT',
    fields: [
      { name: 'id', type: 'ID!' },
      { name: 'body', type: 'String!' },
      { name: 'author', type: 'User' },
      { name: 'createdAt', type: 'DateTime!' },
      { name: 'updatedAt', type: 'DateTime!' },
    ],
  },
  {
    name: 'Label',
    kind: 'OBJECT',
    fields: [
      { name: 'id', type: 'ID!' },
      { name: 'name', type: 'String!' },
      { name: 'color', type: 'String!' },
      { name: 'description', type: 'String' },
    ],
  },
  {
    name: 'PageInfo',
    kind: 'OBJECT',
    fields: [
      { name: 'hasNextPage', type: 'Boolean!' },
      { name: 'hasPreviousPage', type: 'Boolean!' },
      { name: 'startCursor', type: 'String' },
      { name: 'endCursor', type: 'String' },
    ],
  },
  {
    name: 'IssueConnection',
    kind: 'OBJECT',
    fields: [
      { name: 'nodes', type: '[Issue]' },
      { name: 'edges', type: '[IssueEdge]' },
      { name: 'pageInfo', type: 'PageInfo!' },
      { name: 'totalCount', type: 'Int!' },
    ],
  },
  {
    name: 'IssueEdge',
    kind: 'OBJECT',
    fields: [
      { name: 'node', type: 'Issue' },
      { name: 'cursor', type: 'String!' },
    ],
  },
  {
    name: 'PullRequestConnection',
    kind: 'OBJECT',
    fields: [
      { name: 'nodes', type: '[PullRequest]' },
      { name: 'edges', type: '[PullRequestEdge]' },
      { name: 'pageInfo', type: 'PageInfo!' },
      { name: 'totalCount', type: 'Int!' },
    ],
  },
  {
    name: 'PullRequestEdge',
    kind: 'OBJECT',
    fields: [
      { name: 'node', type: 'PullRequest' },
      { name: 'cursor', type: 'String!' },
    ],
  },
  {
    name: 'PullRequestChangedFileConnection',
    kind: 'OBJECT',
    fields: [
      { name: 'nodes', type: '[PullRequestChangedFile]' },
      { name: 'pageInfo', type: 'PageInfo!' },
      { name: 'totalCount', type: 'Int!' },
    ],
  },
  {
    name: 'IssueCommentConnection',
    kind: 'OBJECT',
    fields: [
      { name: 'nodes', type: '[IssueComment]' },
      { name: 'edges', type: '[IssueCommentEdge]' },
      { name: 'pageInfo', type: 'PageInfo!' },
      { name: 'totalCount', type: 'Int!' },
    ],
  },
  {
    name: 'IssueCommentEdge',
    kind: 'OBJECT',
    fields: [
      { name: 'node', type: 'IssueComment' },
      { name: 'cursor', type: 'String!' },
    ],
  },
  {
    name: 'LabelConnection',
    kind: 'OBJECT',
    fields: [
      { name: 'nodes', type: '[Label]' },
      { name: 'pageInfo', type: 'PageInfo!' },
      { name: 'totalCount', type: 'Int!' },
    ],
  },
  {
    name: 'CreateIssuePayload',
    kind: 'OBJECT',
    fields: [
      { name: 'clientMutationId', type: 'String' },
      { name: 'issue', type: 'Issue' },
    ],
  },
  {
    name: 'CloseIssuePayload',
    kind: 'OBJECT',
    fields: [
      { name: 'clientMutationId', type: 'String' },
      { name: 'issue', type: 'Issue' },
    ],
  },
  {
    name: 'ReopenIssuePayload',
    kind: 'OBJECT',
    fields: [
      { name: 'clientMutationId', type: 'String' },
      { name: 'issue', type: 'Issue' },
    ],
  },
  {
    name: 'CreatePullRequestPayload',
    kind: 'OBJECT',
    fields: [
      { name: 'clientMutationId', type: 'String' },
      { name: 'pullRequest', type: 'PullRequest' },
    ],
  },
  {
    name: 'MergePullRequestPayload',
    kind: 'OBJECT',
    fields: [
      { name: 'clientMutationId', type: 'String' },
      { name: 'pullRequest', type: 'PullRequest' },
    ],
  },
  {
    name: 'AddCommentPayload',
    kind: 'OBJECT',
    fields: [
      { name: 'clientMutationId', type: 'String' },
      { name: 'commentEdge', type: 'IssueCommentEdge' },
      { name: 'subject', type: 'Node' },
    ],
  },
  {
    name: 'Node',
    kind: 'INTERFACE',
    fields: [{ name: 'id', type: 'ID!' }],
    possibleTypes: ['Repository', 'Issue', 'PullRequest', 'IssueComment', 'User'],
  },
  {
    name: '__Type',
    kind: 'OBJECT',
    fields: [
      { name: 'kind', type: 'String!' },
      { name: 'name', type: 'String' },
      { name: 'description', type: 'String' },
      { name: 'fields', type: '[__Field]', args: [{ name: 'includeDeprecated', type: 'Boolean', defaultValue: 'false' }] },
      { name: 'enumValues', type: '[__EnumValue]', args: [{ name: 'includeDeprecated', type: 'Boolean', defaultValue: 'false' }] },
      { name: 'inputFields', type: '[__InputValue]' },
      { name: 'possibleTypes', type: '[__Type]' },
      { name: 'ofType', type: '__Type' },
      { name: 'specifiedByURL', type: 'String' },
    ],
  },
  {
    name: '__Field',
    kind: 'OBJECT',
    fields: [
      { name: 'name', type: 'String!' },
      { name: 'description', type: 'String' },
      { name: 'args', type: '[__InputValue]!' },
      { name: 'type', type: '__Type!' },
      { name: 'isDeprecated', type: 'Boolean!' },
      { name: 'deprecationReason', type: 'String' },
    ],
  },
  {
    name: '__EnumValue',
    kind: 'OBJECT',
    fields: [
      { name: 'name', type: 'String!' },
      { name: 'description', type: 'String' },
      { name: 'isDeprecated', type: 'Boolean!' },
      { name: 'deprecationReason', type: 'String' },
    ],
  },
  {
    name: '__InputValue',
    kind: 'OBJECT',
    fields: [
      { name: 'name', type: 'String!' },
      { name: 'description', type: 'String' },
      { name: 'type', type: '__Type!' },
      { name: 'defaultValue', type: 'String' },
    ],
  },
  {
    name: '__Schema',
    kind: 'OBJECT',
    fields: [
      { name: 'queryType', type: '__Type!' },
      { name: 'mutationType', type: '__Type' },
      { name: 'types', type: '[__Type!]!' },
    ],
  },
];

const INPUTS: readonly TypeDef[] = [
  {
    name: 'CreateIssueInput',
    kind: 'INPUT_OBJECT',
    inputFields: [
      { name: 'repositoryId', type: 'ID!' },
      { name: 'title', type: 'String!' },
      { name: 'body', type: 'String' },
      { name: 'clientMutationId', type: 'String' },
    ],
  },
  {
    name: 'CloseIssueInput',
    kind: 'INPUT_OBJECT',
    inputFields: [
      { name: 'issueId', type: 'ID!' },
      { name: 'clientMutationId', type: 'String' },
    ],
  },
  {
    name: 'ReopenIssueInput',
    kind: 'INPUT_OBJECT',
    inputFields: [
      { name: 'issueId', type: 'ID!' },
      { name: 'clientMutationId', type: 'String' },
    ],
  },
  {
    name: 'CreatePullRequestInput',
    kind: 'INPUT_OBJECT',
    inputFields: [
      { name: 'repositoryId', type: 'ID!' },
      { name: 'baseRefName', type: 'String!' },
      { name: 'headRefName', type: 'String!' },
      { name: 'title', type: 'String!' },
      { name: 'body', type: 'String' },
      { name: 'draft', type: 'Boolean' },
      { name: 'clientMutationId', type: 'String' },
    ],
  },
  {
    name: 'MergePullRequestInput',
    kind: 'INPUT_OBJECT',
    inputFields: [
      { name: 'pullRequestId', type: 'ID!' },
      { name: 'mergeMethod', type: 'PullRequestMergeMethod' },
      { name: 'commitHeadline', type: 'String' },
      { name: 'clientMutationId', type: 'String' },
    ],
  },
  {
    name: 'AddCommentInput',
    kind: 'INPUT_OBJECT',
    inputFields: [
      { name: 'subjectId', type: 'ID!' },
      { name: 'body', type: 'String!' },
      { name: 'clientMutationId', type: 'String' },
    ],
  },
  {
    name: 'IssueOrder',
    kind: 'INPUT_OBJECT',
    inputFields: [
      { name: 'field', type: 'IssueOrderField!' },
      { name: 'direction', type: 'OrderDirection!' },
    ],
  },
];

export const SCHEMA: ReadonlyMap<string, TypeDef> = new Map(
  [...SCALARS, ...ENUMS, ...OBJECTS, ...INPUTS].map((def) => [def.name, def]),
);

/** The named type inside a type reference such as `[IssueState!]` or `Repository!`. */
function namedTypeOf(typeRef: string): string {
  return typeRef.replace(/[[\]!]/g, '');
}

// ------------------------------------------------------------------------------------------------
// Global ids: the REST node_id payloads, so ids agree across surfaces.
// ------------------------------------------------------------------------------------------------

export function repoGlobalId(repo: RepoRecord): string {
  return Buffer.from(`010:Repository${repo.id}`).toString('base64');
}

export function issueGlobalId(repo: RepoRecord, issue: IssueRecord): string {
  return Buffer.from(`010:Issue${repo.id}:${issue.number}`).toString('base64');
}

export function pullGlobalId(repo: RepoRecord, pull: PullRecord): string {
  return Buffer.from(`010:PullRequest${repo.id}:${pull.number}`).toString('base64');
}

export function commentGlobalId(comment: CommentRecord): string {
  return Buffer.from(`010:IssueComment${comment.id}`).toString('base64');
}

export function userGlobalId(userId: number): string {
  return Buffer.from(`04:User${userId}`).toString('base64');
}

interface DecodedId {
  readonly type: 'Repository' | 'Issue' | 'PullRequest' | 'IssueComment' | 'User';
  readonly repoId?: number;
  readonly number?: number;
  readonly commentId?: number;
  readonly userId?: number;
}

function decodeGlobalId(id: string): DecodedId | null {
  let text: string;
  try {
    text = Buffer.from(id, 'base64').toString('utf8');
  } catch {
    return null;
  }
  let match = /^\d+:Repository(\d+)$/.exec(text);
  if (match) return { type: 'Repository', repoId: Number(match[1]) };
  match = /^\d+:Issue(\d+):(\d+)$/.exec(text);
  if (match) return { type: 'Issue', repoId: Number(match[1]), number: Number(match[2]) };
  match = /^\d+:PullRequest(\d+):(\d+)$/.exec(text);
  if (match) return { type: 'PullRequest', repoId: Number(match[1]), number: Number(match[2]) };
  match = /^\d+:IssueComment(\d+)$/.exec(text);
  if (match) return { type: 'IssueComment', commentId: Number(match[1]) };
  match = /^\d+:User(\d+)$/.exec(text);
  if (match) return { type: 'User', userId: Number(match[1]) };
  return null;
}

// ------------------------------------------------------------------------------------------------
// Executor.
// ------------------------------------------------------------------------------------------------

/** A runtime value: a GraphQL type name over a surface-specific payload. */
interface GValue {
  readonly type: string;
  readonly data: Record<string, unknown>;
}

function gvalue(type: string, data: Record<string, unknown>): GValue {
  return { type, data };
}

interface ExecError {
  readonly message: string;
  readonly path?: readonly (string | number)[];
}

interface ExecCtx {
  readonly store: ForgeStore;
  readonly variables: Readonly<Record<string, unknown>>;
  readonly fragments: ReadonlyMap<string, FragmentNode>;
  readonly errors: ExecError[];
}

type Resolver = (ctx: ExecCtx, source: GValue, args: Record<string, unknown>) => unknown;

function valueFromAST(node: ValueNode, variables: Readonly<Record<string, unknown>>): unknown {
  switch (node.kind) {
    case 'variable':
      return variables[node.name];
    case 'string':
    case 'enum':
      return node.value;
    case 'number':
    case 'boolean':
      return node.value;
    case 'null':
      return null;
    case 'list':
      return node.values.map((entry) => valueFromAST(entry, variables));
    case 'object': {
      const out: Record<string, unknown> = {};
      for (const field of node.fields) out[field.name] = valueFromAST(field.value, variables);
      return out;
    }
  }
}

function argsOf(field: FieldNode, ctx: ExecCtx): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const arg of field.args) args[arg.name] = valueFromAST(arg.value, ctx.variables);
  return args;
}

function typeCompatible(runtimeType: string, condition: string): boolean {
  if (runtimeType === condition) return true;
  const def = SCHEMA.get(condition);
  if (def && def.kind === 'INTERFACE' && def.possibleTypes) return def.possibleTypes.includes(runtimeType);
  return false;
}

function executeSelectionSet(
  ctx: ExecCtx,
  typeName: string,
  source: GValue,
  selections: readonly SelectionNode[],
  path: readonly (string | number)[],
  out: Record<string, unknown>,
): void {
  for (const selection of selections) {
    if (selection.kind === 'inline') {
      if (selection.typeCondition === null || typeCompatible(source.type, selection.typeCondition)) {
        executeSelectionSet(ctx, source.type, source, selection.selectionSet, path, out);
      }
      continue;
    }
    if (selection.kind === 'spread') {
      const fragment = ctx.fragments.get(selection.name);
      if (fragment && typeCompatible(source.type, fragment.typeCondition)) {
        executeSelectionSet(ctx, source.type, source, fragment.selectionSet, path, out);
      }
      continue;
    }
    const key = selection.alias ?? selection.name;
    const fieldPath = [...path, key];
    if (selection.name === '__typename') {
      out[key] = source.type;
      continue;
    }
    if (typeName === 'Query' && selection.name === '__type') {
      const args = argsOf(selection, ctx);
      const name = typeof args.name === 'string' ? args.name : '';
      const def = SCHEMA.get(name);
      const value = def ? gvalue('__Type', { def }) : null;
      out[key] = completeValue(ctx, '__Type', value, selection, fieldPath);
      continue;
    }
    if (typeName === 'Query' && selection.name === '__schema') {
      const value = gvalue('__Schema', {});
      out[key] = completeValue(ctx, '__Schema', value, selection, fieldPath);
      continue;
    }
    const typeDef = SCHEMA.get(typeName);
    const fieldDef = typeDef?.fields?.find((field) => field.name === selection.name);
    if (!fieldDef) {
      ctx.errors.push({ message: `Field '${selection.name}' doesn't exist on type '${typeName}'`, path: fieldPath });
      out[key] = null;
      continue;
    }
    const resolver = RESOLVERS[typeName]?.[selection.name];
    let raw: unknown;
    try {
      raw = resolver ? resolver(ctx, source, argsOf(selection, ctx)) : source.data[selection.name];
    } catch (err) {
      ctx.errors.push({ message: err instanceof Error ? err.message : String(err), path: fieldPath });
      out[key] = null;
      continue;
    }
    out[key] = completeValue(ctx, fieldDef.type, raw, selection, fieldPath);
  }
}

function completeValue(
  ctx: ExecCtx,
  typeRef: string,
  raw: unknown,
  field: FieldNode,
  path: readonly (string | number)[],
): unknown {
  if (raw === null || raw === undefined) return null;
  const named = namedTypeOf(typeRef);
  const def = SCHEMA.get(named);
  if (Array.isArray(raw)) {
    return raw.map((entry, index) => completeValue(ctx, named, entry, field, [...path, index]));
  }
  if (def && (def.kind === 'OBJECT' || def.kind === 'INTERFACE')) {
    const value = raw as GValue;
    if (!field.selectionSet) return null;
    const out: Record<string, unknown> = {};
    executeSelectionSet(ctx, value.type, value, field.selectionSet, path, out);
    return out;
  }
  return raw;
}

// ------------------------------------------------------------------------------------------------
// Shared builders: users, repositories, issues, pull requests, comments, connections.
// ------------------------------------------------------------------------------------------------

function userValue(ctx: ExecCtx): GValue {
  const viewer = ctx.store.viewer;
  return gvalue('User', {
    id: userGlobalId(viewer.id),
    login: viewer.login,
    name: viewer.name,
    email: viewer.email,
    databaseId: viewer.id,
    avatarUrl: `/avatars/${viewer.id}`,
    url: `/${viewer.login}`,
  });
}

function repoValue(repo: RepoRecord): GValue {
  return gvalue('Repository', { repo });
}

function issueValue(repo: RepoRecord, issue: IssueRecord): GValue {
  return gvalue('Issue', { repo, issue });
}

function pullValue(repo: RepoRecord, pull: PullRecord): GValue {
  return gvalue('PullRequest', { repo, pull });
}

function commentValue(repo: RepoRecord, item: IssueRecord | PullRecord, comment: CommentRecord): GValue {
  return gvalue('IssueComment', { repo, item, comment });
}

function refValue(repo: RepoRecord, branch: string): GValue {
  return gvalue('Ref', { repo, name: branch, prefix: 'refs/heads/' });
}

interface ConnectionEntry {
  readonly node: GValue;
  readonly cursor: string;
}

function cursorFor(index: number): string {
  return Buffer.from(`arrayconnection:${index}`).toString('base64');
}

function indexFromCursor(cursor: string): number | null {
  try {
    const match = /^arrayconnection:(\d+)$/.exec(Buffer.from(cursor, 'base64').toString('utf8'));
    return match ? Number(match[1]) : null;
  } catch {
    return null;
  }
}

/** Slice an ordered list into a connection page: `first` items after the `after` cursor's index. */
function pageOf<T>(
  all: readonly T[],
  args: Record<string, unknown>,
  wrap: (item: T) => GValue,
  connectionType: string,
  edgeType: string,
): GValue {
  let start = 0;
  if (typeof args.after === 'string' && args.after !== '') {
    const index = indexFromCursor(args.after);
    if (index !== null) start = index + 1;
  }
  const first = typeof args.first === 'number' && Number.isFinite(args.first) ? Math.max(0, Math.floor(args.first)) : all.length;
  const slice = all.slice(start, start + first);
  const entries: ConnectionEntry[] = slice.map((item, offset) => ({ node: wrap(item), cursor: cursorFor(start + offset) }));
  const hasNextPage = start + slice.length < all.length;
  const pageInfo = gvalue('PageInfo', {
    hasNextPage,
    hasPreviousPage: start > 0,
    startCursor: entries.length > 0 ? (entries[0] as ConnectionEntry).cursor : null,
    endCursor: entries.length > 0 ? (entries[entries.length - 1] as ConnectionEntry).cursor : null,
  });
  return gvalue(connectionType, { entries, pageInfo, totalCount: all.length, edgeType });
}

/** The current head/base shas of a pull request, following the refs as REST reads them. */
function pullShas(repo: RepoRecord, pull: PullRecord): { headSha: string; baseSha: string } {
  return {
    headSha: repo.refs.get(`refs/heads/${pull.headRef}`) ?? pull.headSha,
    baseSha: repo.refs.get(`refs/heads/${pull.baseRef}`) ?? pull.baseSha,
  };
}

/** Whether an open pull request's head merges cleanly into its base (the REST conflict rule). */
function pullConflict(repo: RepoRecord, pull: PullRecord): boolean {
  const { headSha, baseSha } = pullShas(repo, pull);
  const base = mergeBase(repo.objects, baseSha, headSha);
  return !threeWayMerge(
    repo.objects,
    commitTree(repo.objects, base),
    commitTree(repo.objects, baseSha),
    commitTree(repo.objects, headSha),
  ).ok;
}

interface PullFile {
  readonly path: string;
  readonly additions: number;
  readonly deletions: number;
  readonly changeType: string;
}

function pullFiles(repo: RepoRecord, pull: PullRecord): PullFile[] {
  const { headSha, baseSha } = pullShas(repo, pull);
  const base = mergeBase(repo.objects, baseSha, headSha);
  const changeTypes: Record<string, string> = { added: 'ADDED', removed: 'DELETED', modified: 'MODIFIED', renamed: 'RENAMED', copied: 'COPIED' };
  return diffFiles(repo.objects, commitTree(repo.objects, base), commitTree(repo.objects, headSha)).map((file) => ({
    path: file.filename,
    additions: file.additions,
    deletions: file.deletions,
    changeType: changeTypes[file.status] ?? 'CHANGED',
  }));
}

/** Order issues the way the connection promises: orderBy field + direction, ties by number. */
function orderIssues(issues: IssueRecord[], args: Record<string, unknown>): IssueRecord[] {
  const orderBy = (args.orderBy ?? null) as { field?: string; direction?: string } | null;
  const field = typeof orderBy?.field === 'string' ? orderBy.field : 'CREATED_AT';
  const sign = (typeof orderBy?.direction === 'string' ? orderBy.direction : 'DESC') === 'ASC' ? 1 : -1;
  const keyOf = (issue: IssueRecord): string | number => {
    if (field === 'UPDATED_AT') return issue.updatedAt;
    if (field === 'COMMENTS') return issue.comments.length;
    return issue.createdAt;
  };
  return [...issues].sort((a, b) => {
    const ka = keyOf(a);
    const kb = keyOf(b);
    if (ka !== kb) return (ka < kb ? -1 : 1) * sign;
    return (a.number - b.number) * sign;
  });
}

/** Create one commit on a branch ref holding `treeId` — the REST makeCommit, shared shape. */
export function commitOnBranch(store: ForgeStore, repo: RepoRecord, branchRef: string, treeId: string, message: string): { sha: string; at: string } {
  const at = store.clock.tick();
  const epoch = Math.floor(Date.parse(at) / 1000);
  const viewer = store.viewer;
  const ident = `${viewer.name} <${viewer.email}> ${epoch} +0000`;
  const parent = repo.refs.get(branchRef);
  const data = serializeCommit({ tree: treeId, parents: parent ? [parent] : [], author: ident, committer: ident, message });
  const sha = repo.objects.put('commit', data);
  repo.refs.set(branchRef, sha);
  repo.pushedAt = at;
  repo.updatedAt = at;
  return { sha, at };
}

// ------------------------------------------------------------------------------------------------
// Resolvers.
// ------------------------------------------------------------------------------------------------

const RESOLVERS: Record<string, Record<string, Resolver>> = {
  Query: {
    viewer: (ctx) => userValue(ctx),
    repository: (ctx, _source, args) => {
      const owner = typeof args.owner === 'string' ? args.owner : '';
      const name = typeof args.name === 'string' ? args.name : '';
      if (owner !== ctx.store.viewer.login) return null;
      const repo = ctx.store.getRepo(owner, name);
      return repo ? repoValue(repo) : null;
    },
    user: (ctx, _source, args) => (args.login === ctx.store.viewer.login ? userValue(ctx) : null),
    node: (ctx, _source, args) => resolveNode(ctx, typeof args.id === 'string' ? args.id : ''),
    nodes: (ctx, _source, args) => {
      const ids = Array.isArray(args.ids) ? args.ids : [];
      return ids.map((id) => (typeof id === 'string' ? resolveNode(ctx, id) : null));
    },
  },
  Repository: {
    id: (_ctx, source) => repoGlobalId(source.data.repo as RepoRecord),
    databaseId: (_ctx, source) => (source.data.repo as RepoRecord).id,
    name: (_ctx, source) => (source.data.repo as RepoRecord).name,
    nameWithOwner: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      return `${repo.owner}/${repo.name}`;
    },
    owner: (ctx) => userValue(ctx),
    description: (_ctx, source) => (source.data.repo as RepoRecord).description,
    isPrivate: (_ctx, source) => (source.data.repo as RepoRecord).isPrivate,
    createdAt: (_ctx, source) => (source.data.repo as RepoRecord).createdAt,
    updatedAt: (_ctx, source) => (source.data.repo as RepoRecord).updatedAt,
    url: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      return `/${repo.owner}/${repo.name}`;
    },
    defaultBranchRef: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      if (!repo.refs.has(repo.headRef)) return null;
      return refValue(repo, repo.headRef.replace('refs/heads/', ''));
    },
    viewerPermission: () => 'ADMIN',
    hasIssuesEnabled: () => true,
    issue: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      const item = typeof args.number === 'number' ? repo.items.get(args.number) : undefined;
      return item && item.kind === 'issue' ? issueValue(repo, item) : null;
    },
    issues: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      let issues = [...repo.items.values()].filter((item): item is IssueRecord => item.kind === 'issue');
      if (Array.isArray(args.states)) {
        const wanted = new Set(args.states.map((state) => String(state).toLowerCase()));
        issues = issues.filter((issue) => wanted.has(issue.state));
      }
      if (Array.isArray(args.labels)) {
        const wanted = new Set(args.labels.map((label) => String(label)));
        issues = issues.filter((issue) => issue.labels.some((label) => wanted.has(label)));
      }
      return pageOf(orderIssues(issues, args), args, (issue) => issueValue(repo, issue), 'IssueConnection', 'IssueEdge');
    },
    pullRequest: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      const item = typeof args.number === 'number' ? repo.items.get(args.number) : undefined;
      return item && item.kind === 'pull' ? pullValue(repo, item) : null;
    },
    pullRequests: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      let pulls = [...repo.items.values()].filter((item): item is PullRecord => item.kind === 'pull');
      if (Array.isArray(args.states)) {
        const wanted = new Set(args.states.map((state) => String(state)));
        pulls = pulls.filter((pull) => wanted.has(pull.merged ? 'MERGED' : pull.state.toUpperCase()));
      }
      pulls.sort((a, b) => b.number - a.number);
      return pageOf(pulls, args, (pull) => pullValue(repo, pull), 'PullRequestConnection', 'PullRequestEdge');
    },
  },
  Issue: {
    id: (_ctx, source) => issueGlobalId(source.data.repo as RepoRecord, source.data.issue as IssueRecord),
    number: (_ctx, source) => (source.data.issue as IssueRecord).number,
    title: (_ctx, source) => (source.data.issue as IssueRecord).title,
    state: (_ctx, source) => (source.data.issue as IssueRecord).state.toUpperCase(),
    body: (_ctx, source) => (source.data.issue as IssueRecord).body,
    author: (ctx) => userValue(ctx),
    repository: (_ctx, source) => repoValue(source.data.repo as RepoRecord),
    createdAt: (_ctx, source) => (source.data.issue as IssueRecord).createdAt,
    updatedAt: (_ctx, source) => (source.data.issue as IssueRecord).updatedAt,
    closedAt: (_ctx, source) => {
      const issue = source.data.issue as IssueRecord;
      return issue.state === 'closed' ? issue.updatedAt : null;
    },
    url: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      const issue = source.data.issue as IssueRecord;
      return `/${repo.owner}/${repo.name}/issues/${issue.number}`;
    },
    labels: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      const issue = source.data.issue as IssueRecord;
      const labels = issue.labels
        .map((name) => repo.labels.get(name))
        .filter((label) => label !== undefined)
        .map((label) => gvalue('Label', { id: Buffer.from(`010:Label${repo.id}:${label.name}`).toString('base64'), name: label.name, color: label.color, description: label.description }));
      return pageOf(labels, args, (label) => label, 'LabelConnection', 'LabelEdge');
    },
    comments: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      const issue = source.data.issue as IssueRecord;
      return pageOf(issue.comments, args, (comment) => commentValue(repo, issue, comment), 'IssueCommentConnection', 'IssueCommentEdge');
    },
  },
  PullRequest: {
    id: (_ctx, source) => pullGlobalId(source.data.repo as RepoRecord, source.data.pull as PullRecord),
    number: (_ctx, source) => (source.data.pull as PullRecord).number,
    title: (_ctx, source) => (source.data.pull as PullRecord).title,
    body: (_ctx, source) => (source.data.pull as PullRecord).body,
    state: (_ctx, source) => {
      const pull = source.data.pull as PullRecord;
      return pull.merged ? 'MERGED' : pull.state.toUpperCase();
    },
    merged: (_ctx, source) => (source.data.pull as PullRecord).merged,
    mergeable: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      const pull = source.data.pull as PullRecord;
      if (pull.merged || pull.state !== 'open') return 'UNKNOWN';
      return pullConflict(repo, pull) ? 'CONFLICTING' : 'MERGEABLE';
    },
    headRefName: (_ctx, source) => (source.data.pull as PullRecord).headRef,
    baseRefName: (_ctx, source) => (source.data.pull as PullRecord).baseRef,
    headRef: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      const pull = source.data.pull as PullRecord;
      return repo.refs.has(`refs/heads/${pull.headRef}`) ? refValue(repo, pull.headRef) : null;
    },
    baseRef: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      const pull = source.data.pull as PullRecord;
      return repo.refs.has(`refs/heads/${pull.baseRef}`) ? refValue(repo, pull.baseRef) : null;
    },
    author: (ctx) => userValue(ctx),
    repository: (_ctx, source) => repoValue(source.data.repo as RepoRecord),
    createdAt: (_ctx, source) => (source.data.pull as PullRecord).createdAt,
    updatedAt: (_ctx, source) => (source.data.pull as PullRecord).updatedAt,
    closedAt: (_ctx, source) => {
      const pull = source.data.pull as PullRecord;
      return pull.state === 'closed' ? (pull.mergedAt ?? pull.updatedAt) : null;
    },
    mergedAt: (_ctx, source) => (source.data.pull as PullRecord).mergedAt,
    additions: (_ctx, source) => pullFiles(source.data.repo as RepoRecord, source.data.pull as PullRecord).reduce((sum, file) => sum + file.additions, 0),
    deletions: (_ctx, source) => pullFiles(source.data.repo as RepoRecord, source.data.pull as PullRecord).reduce((sum, file) => sum + file.deletions, 0),
    changedFiles: (_ctx, source) => pullFiles(source.data.repo as RepoRecord, source.data.pull as PullRecord).length,
    url: (_ctx, source) => {
      const repo = source.data.repo as RepoRecord;
      const pull = source.data.pull as PullRecord;
      return `/${repo.owner}/${repo.name}/pull/${pull.number}`;
    },
    files: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      const pull = source.data.pull as PullRecord;
      return pageOf(
        pullFiles(repo, pull),
        args,
        (file) => gvalue('PullRequestChangedFile', { path: file.path, additions: file.additions, deletions: file.deletions, changeType: file.changeType }),
        'PullRequestChangedFileConnection',
        'PullRequestChangedFileEdge',
      );
    },
    comments: (_ctx, source, args) => {
      const repo = source.data.repo as RepoRecord;
      const pull = source.data.pull as PullRecord;
      return pageOf(pull.comments, args, (comment) => commentValue(repo, pull, comment), 'IssueCommentConnection', 'IssueCommentEdge');
    },
  },
  Ref: {
    name: (_ctx, source) => source.data.name,
    prefix: (_ctx, source) => source.data.prefix,
    repository: (_ctx, source) => repoValue(source.data.repo as RepoRecord),
  },
  IssueComment: {
    id: (_ctx, source) => commentGlobalId(source.data.comment as CommentRecord),
    body: (_ctx, source) => (source.data.comment as CommentRecord).body,
    author: (ctx) => userValue(ctx),
    createdAt: (_ctx, source) => (source.data.comment as CommentRecord).createdAt,
    updatedAt: (_ctx, source) => (source.data.comment as CommentRecord).updatedAt,
  },
  '__Type': {
    kind: (_ctx, source) => (source.data.def as TypeDef).kind,
    name: (_ctx, source) => (source.data.def as TypeDef).name,
    description: () => null,
    fields: (_ctx, source) => {
      const def = source.data.def as TypeDef;
      if (def.kind !== 'OBJECT' && def.kind !== 'INTERFACE') return null;
      return (def.fields ?? []).map((field) => gvalue('__Field', { def: field }));
    },
    enumValues: (_ctx, source) => {
      const def = source.data.def as TypeDef;
      if (def.kind !== 'ENUM') return null;
      return (def.enumValues ?? []).map((name) => gvalue('__EnumValue', { name }));
    },
    inputFields: (_ctx, source) => {
      const def = source.data.def as TypeDef;
      if (def.kind !== 'INPUT_OBJECT') return null;
      return (def.inputFields ?? []).map((field) => gvalue('__InputValue', { def: field }));
    },
    possibleTypes: (_ctx, source) => {
      const def = source.data.def as TypeDef;
      return (def.possibleTypes ?? []).map((name) => gvalue('__Type', { def: SCHEMA.get(name) }));
    },
    ofType: () => null,
    specifiedByURL: () => null,
  },
  '__Field': {
    name: (_ctx, source) => (source.data.def as FieldDef).name,
    description: () => null,
    args: (_ctx, source) => ((source.data.def as FieldDef).args ?? []).map((arg) => gvalue('__InputValue', { def: arg })),
    type: (_ctx, source) => {
      const def = SCHEMA.get(namedTypeOf((source.data.def as FieldDef).type));
      return def ? gvalue('__Type', { def }) : null;
    },
    isDeprecated: () => false,
    deprecationReason: () => null,
  },
  '__EnumValue': {
    name: (_ctx, source) => source.data.name,
    description: () => null,
    isDeprecated: () => false,
    deprecationReason: () => null,
  },
  '__InputValue': {
    name: (_ctx, source) => (source.data.def as InputValueDef).name,
    description: () => null,
    type: (_ctx, source) => {
      const def = SCHEMA.get(namedTypeOf((source.data.def as InputValueDef).type));
      return def ? gvalue('__Type', { def }) : null;
    },
    defaultValue: (_ctx, source) => (source.data.def as InputValueDef).defaultValue ?? null,
  },
  '__Schema': {
    queryType: () => gvalue('__Type', { def: SCHEMA.get('Query') }),
    mutationType: () => gvalue('__Type', { def: SCHEMA.get('Mutation') }),
    types: () => [...SCHEMA.values()].map((def) => gvalue('__Type', { def })),
  },
};

/** Connection field resolvers, registered per connection type (nodes/edges/pageInfo/totalCount). */
for (const connectionType of ['IssueConnection', 'PullRequestConnection', 'PullRequestChangedFileConnection', 'IssueCommentConnection', 'LabelConnection']) {
  RESOLVERS[connectionType] = {
    nodes: (_ctx, source) => (source.data.entries as ConnectionEntry[]).map((entry) => entry.node),
    edges: (_ctx, source) =>
      (source.data.entries as ConnectionEntry[]).map((entry) =>
        gvalue(String(source.data.edgeType), { node: entry.node, cursor: entry.cursor }),
      ),
    pageInfo: (_ctx, source) => source.data.pageInfo,
    totalCount: (_ctx, source) => source.data.totalCount,
  };
}

for (const edgeType of ['IssueEdge', 'PullRequestEdge', 'IssueCommentEdge']) {
  RESOLVERS[edgeType] = {
    node: (_ctx, source) => source.data.node,
    cursor: (_ctx, source) => source.data.cursor,
  };
}

/** Resolve a Relay global id to the object it names, or null when it names nothing here. */
function resolveNode(ctx: ExecCtx, id: string): GValue | null {
  const decoded = decodeGlobalId(id);
  if (!decoded) return null;
  if (decoded.type === 'User') return decoded.userId === ctx.store.viewer.id ? userValue(ctx) : null;
  if (decoded.type === 'Repository') {
    const repo = ctx.store.reposById.get(decoded.repoId as number);
    return repo && repo.owner === ctx.store.viewer.login ? repoValue(repo) : null;
  }
  if (decoded.type === 'IssueComment') {
    const commentId = decoded.commentId as number;
    const repo = ctx.store.reposById.get(Math.floor(commentId / 1_000_000));
    if (!repo) return null;
    for (const item of repo.items.values()) {
      const comment = item.comments.find((entry) => entry.id === commentId);
      if (comment) return commentValue(repo, item, comment);
    }
    return null;
  }
  const repo = ctx.store.reposById.get(decoded.repoId as number);
  if (!repo || repo.owner !== ctx.store.viewer.login) return null;
  const item = repo.items.get(decoded.number as number);
  if (!item) return null;
  if (decoded.type === 'Issue') return item.kind === 'issue' ? issueValue(repo, item) : null;
  return item.kind === 'pull' ? pullValue(repo, item) : null;
}

// ------------------------------------------------------------------------------------------------
// Mutations: the same store writes the REST surface performs, with the same clock discipline.
// ------------------------------------------------------------------------------------------------

function inputOf(args: Record<string, unknown>): Record<string, unknown> {
  const input = args.input;
  return typeof input === 'object' && input !== null && !Array.isArray(input) ? (input as Record<string, unknown>) : {};
}

function repoFromInput(ctx: ExecCtx, input: Record<string, unknown>): RepoRecord {
  const id = typeof input.repositoryId === 'string' ? decodeGlobalId(input.repositoryId) : null;
  const repo = id && id.type === 'Repository' ? ctx.store.reposById.get(id.repoId as number) : undefined;
  if (!repo || repo.owner !== ctx.store.viewer.login) {
    throw new Error(`Could not resolve to a Repository with the global id of '${String(input.repositoryId)}'`);
  }
  return repo;
}

function itemFromInput(ctx: ExecCtx, rawId: unknown, kind: 'issue' | 'pull'): { repo: RepoRecord; item: IssueRecord | PullRecord } {
  const id = typeof rawId === 'string' ? decodeGlobalId(rawId) : null;
  const repo = id ? ctx.store.reposById.get(id.repoId as number) : undefined;
  const item = repo ? repo.items.get(id?.number as number) : undefined;
  if (!repo || !item || item.kind !== kind) {
    throw new Error(`Could not resolve to a ${kind === 'issue' ? 'Issue' : 'PullRequest'} with the global id of '${String(rawId)}'`);
  }
  return { repo, item };
}

RESOLVERS.Mutation = {
  createIssue: (ctx, _source, args) => {
    const input = inputOf(args);
    const repo = repoFromInput(ctx, input);
    const title = typeof input.title === 'string' ? input.title : '';
    if (title === '') throw new Error('Title is required');
    const now = ctx.store.clock.tick();
    const issue: IssueRecord = {
      kind: 'issue',
      number: repo.nextItemNumber,
      title,
      body: typeof input.body === 'string' ? input.body : '',
      state: 'open',
      labels: [],
      comments: [],
      author: ctx.store.viewer.login,
      createdAt: now,
      updatedAt: now,
    };
    repo.nextItemNumber += 1;
    repo.items.set(issue.number, issue);
    return gvalue('CreateIssuePayload', { clientMutationId: input.clientMutationId ?? null, issue: issueValue(repo, issue) });
  },
  closeIssue: (ctx, _source, args) => {
    const input = inputOf(args);
    const { repo, item } = itemFromInput(ctx, input.issueId, 'issue');
    const issue = item as IssueRecord;
    issue.state = 'closed';
    issue.updatedAt = ctx.store.clock.tick();
    return gvalue('CloseIssuePayload', { clientMutationId: input.clientMutationId ?? null, issue: issueValue(repo, issue) });
  },
  reopenIssue: (ctx, _source, args) => {
    const input = inputOf(args);
    const { repo, item } = itemFromInput(ctx, input.issueId, 'issue');
    const issue = item as IssueRecord;
    issue.state = 'open';
    issue.updatedAt = ctx.store.clock.tick();
    return gvalue('ReopenIssuePayload', { clientMutationId: input.clientMutationId ?? null, issue: issueValue(repo, issue) });
  },
  createPullRequest: (ctx, _source, args) => {
    const input = inputOf(args);
    const repo = repoFromInput(ctx, input);
    const title = typeof input.title === 'string' ? input.title : '';
    const head = typeof input.headRefName === 'string' ? input.headRefName : '';
    const base = typeof input.baseRefName === 'string' ? input.baseRefName : '';
    const headSha = repo.refs.get(`refs/heads/${head}`);
    const baseSha = repo.refs.get(`refs/heads/${base}`);
    if (title === '' || !headSha || !baseSha || head === base) {
      throw new Error('Validation failed: title, headRefName and baseRefName must name two different existing branches');
    }
    const duplicate = [...repo.items.values()].some(
      (item) => item.kind === 'pull' && item.state === 'open' && item.headRef === head && item.baseRef === base,
    );
    if (duplicate) throw new Error(`A pull request already exists for ${head} into ${base}`);
    const now = ctx.store.clock.tick();
    const pull: PullRecord = {
      kind: 'pull',
      number: repo.nextItemNumber,
      title,
      body: typeof input.body === 'string' ? input.body : '',
      state: 'open',
      labels: [],
      comments: [],
      author: ctx.store.viewer.login,
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
    return gvalue('CreatePullRequestPayload', { clientMutationId: input.clientMutationId ?? null, pullRequest: pullValue(repo, pull) });
  },
  mergePullRequest: (ctx, _source, args) => {
    const input = inputOf(args);
    const { repo, item } = itemFromInput(ctx, input.pullRequestId, 'pull');
    const pull = item as PullRecord;
    if (pull.state !== 'open' || pull.merged) throw new Error('Pull Request is not mergeable');
    const { headSha, baseSha } = pullShas(repo, pull);
    const baseRef = `refs/heads/${pull.baseRef}`;
    const base = mergeBase(repo.objects, baseSha, headSha);
    if (base === headSha) throw new Error('Pull Request is not mergeable');
    const outcome = threeWayMerge(
      repo.objects,
      commitTree(repo.objects, base),
      commitTree(repo.objects, baseSha),
      commitTree(repo.objects, headSha),
    );
    if (!outcome.ok) throw new Error('Pull Request is not mergeable due to conflicts');
    const headline = typeof input.commitHeadline === 'string' && input.commitHeadline !== '' ? input.commitHeadline : `${pull.title} (#${pull.number})`;
    const message = pull.body ? `${headline}\n\n${pull.body}` : headline;
    const commit = commitOnBranch(ctx.store, repo, baseRef, outcome.tree, message);
    pull.merged = true;
    pull.mergedAt = commit.at;
    pull.mergeCommitSha = commit.sha;
    pull.state = 'closed';
    pull.updatedAt = commit.at;
    return gvalue('MergePullRequestPayload', { clientMutationId: input.clientMutationId ?? null, pullRequest: pullValue(repo, pull) });
  },
  addComment: (ctx, _source, args) => {
    const input = inputOf(args);
    const decoded = typeof input.subjectId === 'string' ? decodeGlobalId(input.subjectId) : null;
    const repo = decoded ? ctx.store.reposById.get(decoded.repoId as number) : undefined;
    const item = repo ? repo.items.get(decoded?.number as number) : undefined;
    if (!repo || !item || (decoded?.type !== 'Issue' && decoded?.type !== 'PullRequest')) {
      throw new Error(`Could not resolve to a node with the global id of '${String(input.subjectId)}'`);
    }
    const now = ctx.store.clock.tick();
    const comment: CommentRecord = {
      id: repo.id * 1_000_000 + repo.nextCommentId,
      body: typeof input.body === 'string' ? input.body : '',
      author: ctx.store.viewer.login,
      createdAt: now,
      updatedAt: now,
    };
    repo.nextCommentId += 1;
    item.comments.push(comment);
    item.updatedAt = now;
    const node = commentValue(repo, item, comment);
    const edge = gvalue('IssueCommentEdge', { node, cursor: cursorFor(item.comments.length - 1) });
    return gvalue('AddCommentPayload', {
      clientMutationId: input.clientMutationId ?? null,
      commentEdge: edge,
      subject: item.kind === 'issue' ? issueValue(repo, item) : pullValue(repo, item),
    });
  },
};

// ------------------------------------------------------------------------------------------------
// Operation execution and the HTTP handler.
// ------------------------------------------------------------------------------------------------

interface GraphQLOperationRequest {
  readonly query?: unknown;
  readonly variables?: unknown;
  readonly operationName?: unknown;
}

interface GraphQLEnvelope {
  readonly data?: Record<string, unknown> | null;
  readonly errors?: readonly ExecError[];
}

function executeOperation(store: ForgeStore, request: GraphQLOperationRequest): GraphQLEnvelope {
  if (typeof request.query !== 'string' || request.query === '') {
    return { errors: [{ message: 'No GraphQL document was given' }] };
  }
  let document: DocumentNode;
  try {
    document = new Parser(request.query).parseDocument();
  } catch (err) {
    return { errors: [{ message: err instanceof Error ? err.message : String(err) }] };
  }
  const variables =
    typeof request.variables === 'object' && request.variables !== null && !Array.isArray(request.variables)
      ? (request.variables as Record<string, unknown>)
      : {};
  let operation: OperationNode | undefined;
  if (typeof request.operationName === 'string' && request.operationName !== '') {
    operation = document.operations.find((entry) => entry.name === request.operationName);
    if (!operation) return { errors: [{ message: `Unknown operation named "${request.operationName}"` }] };
  } else if (document.operations.length === 1) {
    operation = document.operations[0];
  } else {
    operation = document.operations[0];
  }
  if (!operation) return { errors: [{ message: 'No operation could be selected' }] };
  const rootType = operation.operation === 'mutation' ? 'Mutation' : 'Query';
  const ctx: ExecCtx = { store, variables, fragments: document.fragments, errors: [] };
  const data: Record<string, unknown> = {};
  executeSelectionSet(ctx, rootType, gvalue(rootType, {}), operation.selectionSet, [], data);
  const envelope: { data: Record<string, unknown>; errors?: ExecError[] } = { data };
  if (ctx.errors.length > 0) envelope.errors = ctx.errors;
  return envelope;
}

function sendGraphQLJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'x-github-request-id': 'forge-1',
  });
  res.end(JSON.stringify(body));
}

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > 64 * 1024 * 1024) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

/** POST /api/graphql: one operation document, or a JSON array of them answered in order. */
export async function handleGraphQL(store: ForgeStore, req: IncomingMessage, res: ServerResponse, isAuthorized: boolean): Promise<void> {
  if (!isAuthorized) {
    sendGraphQLJson(res, 401, { message: 'Bad credentials', errors: [{ message: 'Bad credentials' }] });
    return;
  }
  if ((req.method ?? 'GET') !== 'POST') {
    sendGraphQLJson(res, 405, { message: 'Method not allowed', errors: [{ message: 'Method not allowed' }] });
    return;
  }
  let parsed: unknown;
  try {
    const raw = await readRequestBody(req);
    parsed = raw.trim() === '' ? {} : (JSON.parse(raw) as unknown);
  } catch {
    sendGraphQLJson(res, 400, { errors: [{ message: 'Problems parsing JSON' }] });
    return;
  }
  if (Array.isArray(parsed)) {
    const envelopes = parsed.map((entry) =>
      executeOperation(store, typeof entry === 'object' && entry !== null ? (entry as GraphQLOperationRequest) : {}),
    );
    sendGraphQLJson(res, 200, envelopes);
    return;
  }
  if (typeof parsed !== 'object' || parsed === null) {
    sendGraphQLJson(res, 400, { errors: [{ message: 'Problems parsing JSON' }] });
    return;
  }
  sendGraphQLJson(res, 200, executeOperation(store, parsed as GraphQLOperationRequest));
}

