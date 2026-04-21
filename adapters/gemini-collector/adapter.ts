/**
 * Gemini Collector (Live) adapter for hstry
 *
 * Parses live synced data from the Gemini Collector desktop app at:
 *   ~/Library/Application Support/com.gemini-collector/accounts/<account_id>/
 */

import { readdir, readFile, stat } from 'fs/promises';
import { join } from 'path';
import { homedir } from 'os';
import type {
  Adapter,
  AdapterInfo,
  Attachment,
  Conversation,
  Message,
  ParseOptions,
} from '../types/index.ts';
import { runAdapter, textOnlyParts, textPart, thinkingPart } from '../types/index.ts';

const DEFAULT_PATHS = [
  join(homedir(), 'Library', 'Application Support', 'com.gemini-collector', 'accounts'), // macOS
  join(homedir(), '.config', 'com.gemini-collector', 'accounts'), // Linux (XDG)
];

// ---------------------------------------------------------------------------
// Raw JSONL types
// ---------------------------------------------------------------------------

interface RawMeta {
  type: 'meta';
  id: string;
  accountId: string;
  title: string;
  createdAt: string;  // ISO 8601
  updatedAt: string;  // ISO 8601
  remoteHash?: string;
}

interface RawAttachment {
  mediaId: string;
  mimeType: string;
}

interface RawMessage {
  type: 'message';
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: string;  // ISO 8601
  attachments?: RawAttachment[];
  model?: string;
  thinking?: string;
}

interface ManifestItem {
  id: string;
  status: 'normal' | 'hidden' | 'lost';
  updatedAt: string;  // ISO 8601
}

interface ConversationsManifest {
  items: ManifestItem[];
}

interface MediaManifest {
  url_to_name: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sortConversations(conversations: Conversation[]): Conversation[] {
  conversations.sort((a, b) => b.createdAt - a.createdAt);
  return conversations;
}

/** Parse ISO 8601 → Unix ms */
function isoToMs(iso: string): number {
  return Date.parse(iso);
}

/**
 * Build a hash→originalFilename reverse map from media_manifest.json.
 * The manifest maps Google download URLs (containing `filename=` query param) to local hash filenames.
 * We reverse it so we can look up original name by hash.
 */
function buildHashToFilename(manifest: MediaManifest): Map<string, string> {
  const map = new Map<string, string>();
  for (const [url, hashName] of Object.entries(manifest.url_to_name)) {
    try {
      // Parse filename= from query string
      const urlObj = new URL(url);
      const filename = urlObj.searchParams.get('filename');
      if (filename) {
        map.set(hashName, filename);
      }
    } catch {
      // URL parse failed — skip
    }
  }
  return map;
}

function convertMessage(raw: RawMessage, hashToFilename: Map<string, string>): Message | null {
  const text = raw.text?.trim() ?? '';
  const thinking = raw.thinking?.trim() ?? '';

  // Skip empty messages
  if (!text && !thinking) return null;

  const role: Message['role'] = raw.role === 'model' ? 'assistant' : 'user';
  const createdAt = isoToMs(raw.timestamp);

  // Parts: include thinking if present
  let parts;
  if (thinking) {
    parts = [thinkingPart(thinking), ...(text ? [textPart(text)] : [])];
  } else {
    parts = textOnlyParts(text);
  }

  // Attachments
  let attachments: Attachment[] | undefined;
  if (raw.attachments && raw.attachments.length > 0) {
    attachments = raw.attachments.map(att => {
      const name = hashToFilename.get(att.mediaId) ?? att.mediaId;
      return {
        type: 'file' as const,
        name,
        mimeType: att.mimeType,
      };
    });
  }

  const msg: Message = {
    role,
    content: text,
    parts,
    createdAt,
    provider: 'google',
  } as Message & { provider: string };

  if (raw.model) (msg as any).model = raw.model;
  if (attachments) msg.attachments = attachments;

  return msg;
}

async function parseJsonlFile(
  filePath: string,
  hashToFilename: Map<string, string>,
  since?: number
): Promise<Conversation | null> {
  const raw = await readFile(filePath, 'utf-8').catch(() => null);
  if (!raw) return null;

  const lines = raw.split('\n').filter(l => l.trim());
  if (lines.length === 0) return null;

  let meta: RawMeta;
  try {
    meta = JSON.parse(lines[0]) as RawMeta;
  } catch {
    return null;
  }

  if (meta.type !== 'meta') return null;

  const updatedAt = isoToMs(meta.updatedAt);
  if (since !== undefined && updatedAt < since) return null;

  const messages: Message[] = [];
  for (let i = 1; i < lines.length; i++) {
    let rawMsg: RawMessage;
    try {
      rawMsg = JSON.parse(lines[i]) as RawMessage;
    } catch {
      continue;
    }
    if (rawMsg.type !== 'message') continue;

    const msg = convertMessage(rawMsg, hashToFilename);
    if (msg) messages.push(msg);
  }

  if (messages.length === 0) return null;

  return {
    externalId: meta.id,
    title: meta.title,
    createdAt: isoToMs(meta.createdAt),
    updatedAt,
    provider: 'google',
    messages,
  } as Conversation & { provider: string };
}

async function loadManifest(path: string): Promise<ConversationsManifest | null> {
  const raw = await readFile(join(path, 'conversations.json'), 'utf-8').catch(() => null);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as ConversationsManifest;
  } catch {
    return null;
  }
}

async function loadMediaManifest(path: string): Promise<Map<string, string>> {
  const raw = await readFile(join(path, 'media_manifest.json'), 'utf-8').catch(() => null);
  if (!raw) return new Map();
  try {
    const manifest = JSON.parse(raw) as MediaManifest;
    return buildHashToFilename(manifest);
  } catch {
    return new Map();
  }
}

async function parseAccountDir(path: string, opts?: ParseOptions): Promise<Conversation[]> {
  const [manifest, hashToFilename] = await Promise.all([
    loadManifest(path),
    loadMediaManifest(path),
  ]);

  const lostIds = new Set<string>();
  if (manifest) {
    for (const item of manifest.items) {
      if (item.status === 'lost') lostIds.add(item.id);
    }
  }

  const convDir = join(path, 'conversations');
  const entries = await readdir(convDir).catch(() => [] as string[]);
  const conversations: Conversation[] = [];

  for (const filename of entries) {
    if (!filename.endsWith('.jsonl')) continue;
    const convId = filename.slice(0, -6);
    if (lostIds.has(convId)) continue;

    const conv = await parseJsonlFile(join(convDir, filename), hashToFilename, opts?.since);
    if (!conv) continue;

    conversations.push(conv);
    if (opts?.limit && conversations.length >= opts.limit) break;
  }

  return conversations;
}

async function parseAccountDirSince(path: string, since: number): Promise<Conversation[]> {
  const [manifest, hashToFilename] = await Promise.all([
    loadManifest(path),
    loadMediaManifest(path),
  ]);

  const eligibleIds = new Set<string>();
  const lostIds = new Set<string>();
  if (manifest) {
    for (const item of manifest.items) {
      if (item.status === 'lost') { lostIds.add(item.id); continue; }
      if (isoToMs(item.updatedAt) >= since) eligibleIds.add(item.id);
    }
  }

  const convDir = join(path, 'conversations');
  const entries = await readdir(convDir).catch(() => [] as string[]);
  const conversations: Conversation[] = [];

  for (const filename of entries) {
    if (!filename.endsWith('.jsonl')) continue;
    const convId = filename.slice(0, -6);
    if (lostIds.has(convId)) continue;
    if (manifest && !eligibleIds.has(convId)) continue;

    const conv = await parseJsonlFile(join(convDir, filename), hashToFilename, since);
    if (!conv) continue;
    conversations.push(conv);
  }

  return conversations;
}

/** True if `path` looks like a single Gemini Collector account directory. */
async function isAccountDir(path: string): Promise<boolean> {
  const [metaStat, convDirStat] = await Promise.all([
    stat(join(path, 'meta.json')).catch(() => null),
    stat(join(path, 'conversations')).catch(() => null),
  ]);
  return (metaStat?.isFile() ?? false) && (convDirStat?.isDirectory() ?? false);
}

/**
 * Accepts either:
 *  - a direct account dir  (.../accounts/<email>/)  → returns [path]
 *  - the accounts parent   (.../accounts/)           → enumerates subdirs, returns all valid account dirs
 */
async function resolveAccountPaths(path: string): Promise<string[]> {
  if (await isAccountDir(path)) return [path];

  const entries = await readdir(path, { withFileTypes: true }).catch(() => []);
  const accountPaths: string[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const candidate = join(path, entry.name);
    if (await isAccountDir(candidate)) accountPaths.push(candidate);
  }
  return accountPaths;
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

const adapter: Adapter = {
  info(): AdapterInfo {
    return {
      name: 'gemini-collector',
      displayName: 'Gemini Collector (Live)',
      version: '1.0.0',
      defaultPaths: DEFAULT_PATHS,
    };
  },

  async detect(path: string): Promise<number | null> {
    // Direct account dir
    if (await isAccountDir(path)) {
      const manifestStat = await stat(join(path, 'conversations.json')).catch(() => null);
      return manifestStat?.isFile() ? 0.95 : 0.7;
    }

    // Parent accounts/ dir — return 0.9 if any subdir is a valid account
    const entries = await readdir(path, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      if (await isAccountDir(join(path, entry.name))) return 0.9;
    }

    return null;
  },

  async parse(path: string, opts?: ParseOptions): Promise<Conversation[]> {
    const accountPaths = await resolveAccountPaths(path);
    const all: Conversation[] = [];
    for (const accountPath of accountPaths) {
      const convs = await parseAccountDir(accountPath, opts);
      all.push(...convs);
      if (opts?.limit && all.length >= opts.limit) break;
    }
    return sortConversations(all);
  },

  supportsIncremental: true,

  async parseSince(path: string, since: number): Promise<Conversation[]> {
    const accountPaths = await resolveAccountPaths(path);
    const all: Conversation[] = [];
    for (const accountPath of accountPaths) {
      const convs = await parseAccountDirSince(accountPath, since);
      all.push(...convs);
    }
    return sortConversations(all);
  },
};

runAdapter(adapter);
