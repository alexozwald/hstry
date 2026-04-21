/**
 * Google AI Studio adapter for hstry
 *
 * Parses Google AI Studio conversation exports — extensionless JSON files
 * named by their conversation title, synced from Google Drive via rclone.
 */

import { readdir, readFile, stat } from 'fs/promises';
import { basename, extname, join } from 'path';
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
  join(homedir(), 'Documents'),
  join(homedir(), 'Downloads'),
  join(homedir(), 'Desktop'),
];

// ---------------------------------------------------------------------------
// Raw data types
// ---------------------------------------------------------------------------

interface RawChunk {
  role?: string;
  text?: string;
  tokenCount?: number;
  isThought?: boolean;
  parts?: Array<{ text?: string }>;
  thoughtSignatures?: unknown[];
  driveDocument?: { id: string };
  driveImage?: { id: string };
}

interface RawData {
  runSettings?: {
    model?: string;
  };
  systemInstruction?: { text?: string } | Record<string, never>;
  chunkedPrompt?: {
    chunks?: RawChunk[];
    pendingInputs?: unknown[];
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sortConversations(conversations: Conversation[]): Conversation[] {
  conversations.sort((a, b) => b.createdAt - a.createdAt);
  return conversations;
}

/** True if filename has no extension (these are the conversation files). */
function isConversationFile(filename: string): boolean {
  return extname(filename) === '';
}

function extractThinkingText(chunk: RawChunk): string {
  const parts = chunk.parts ?? [];
  return parts
    .map(p => p.text ?? '')
    .filter(Boolean)
    .join('\n');
}

function convertFile(
  filePath: string,
  data: RawData,
  mtimeMs: number,
  birthtimeMs: number
): Conversation | null {
  const chunkedPrompt = data.chunkedPrompt;
  if (!chunkedPrompt) return null;

  const chunks = chunkedPrompt.chunks ?? [];

  const title = basename(filePath);

  // Timestamps: use birthtime as createdAt if it looks real (> year 2000 ~= 946684800000ms)
  const createdAt = birthtimeMs > 946684800000 ? birthtimeMs : mtimeMs;
  const updatedAt = mtimeMs;

  // Model: strip "models/" prefix
  const rawModel = data.runSettings?.model ?? '';
  const model = rawModel ? rawModel.replace(/^models\//, '') : undefined;

  const messages: Message[] = [];

  // System instruction → leading system message
  const sysInstr = data.systemInstruction;
  const sysText =
    sysInstr && typeof sysInstr === 'object' && 'text' in sysInstr
      ? (sysInstr.text ?? '').trim()
      : '';
  if (sysText) {
    messages.push({
      role: 'system',
      content: sysText,
      parts: textOnlyParts(sysText),
    });
  }

  // Accumulator state for merging consecutive same-role chunks
  let currentRole: 'user' | 'assistant' | null = null;
  let currentParts: string[] = [];
  let pendingThinking: string | null = null;
  let pendingAttachments: Attachment[] = [];

  function flush(): void {
    if (currentParts.length === 0 && pendingAttachments.length === 0) {
      pendingThinking = null;
      return;
    }

    const role = currentRole ?? 'user';
    const text = currentParts.join('\n\n');

    let parts;
    if (pendingThinking) {
      parts = [thinkingPart(pendingThinking), ...(text ? [textPart(text)] : [])];
    } else {
      parts = textOnlyParts(text);
    }

    const msg: Message = {
      role,
      content: text,
      parts,
    };
    if (model && role === 'assistant') (msg as any).model = model;
    if (pendingAttachments.length > 0) msg.attachments = [...pendingAttachments];

    messages.push(msg);

    currentRole = null;
    currentParts = [];
    pendingThinking = null;
    pendingAttachments = [];
  }

  for (const chunk of chunks) {
    const chunkRole: 'user' | 'assistant' =
      (chunk.role ?? 'user') === 'user' ? 'user' : 'assistant';

    // Thinking chunk
    if (chunk.isThought) {
      if (currentRole !== null && currentRole !== 'assistant') flush();
      currentRole = 'assistant';
      const t = extractThinkingText(chunk);
      if (t) {
        pendingThinking = pendingThinking ? pendingThinking + '\n\n' + t : t;
      }
      continue;
    }

    // Drive attachment chunk
    if (chunk.driveDocument || chunk.driveImage) {
      if (currentRole !== null && currentRole !== chunkRole) flush();
      currentRole = chunkRole;
      const driveId = chunk.driveDocument?.id ?? chunk.driveImage?.id ?? 'unknown';
      const mimeType = chunk.driveDocument
        ? 'application/vnd.google-apps.document'
        : 'image/*';
      pendingAttachments.push({ type: 'file', name: driveId, mimeType });
      continue;
    }

    // Normal text chunk
    const text = chunk.text;
    if (text === undefined || text === null) continue;

    if (currentRole !== null && currentRole !== chunkRole) flush();
    currentRole = chunkRole;
    currentParts.push(text);
  }

  flush();

  if (messages.length === 0) return null;

  const conv: Conversation = {
    title,
    createdAt,
    updatedAt,
    messages,
  };
  if (model) conv.model = model;
  (conv as any).provider = 'google';

  return conv;
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

const adapter: Adapter = {
  info(): AdapterInfo {
    return {
      name: 'google-aistudio',
      displayName: 'Google AI Studio',
      version: '1.0.0',
      defaultPaths: DEFAULT_PATHS,
    };
  },

  async detect(path: string): Promise<number | null> {
    const entries = await readdir(path).catch(() => [] as string[]);
    const candidates = entries.filter(isConversationFile).slice(0, 10);

    for (const name of candidates) {
      const raw = await readFile(join(path, name), 'utf-8').catch(() => null);
      if (!raw) continue;
      try {
        const data = JSON.parse(raw) as Record<string, unknown>;
        if ('chunkedPrompt' in data) return 0.9;
      } catch {
        continue;
      }
    }

    return null;
  },

  async parse(path: string, opts?: ParseOptions): Promise<Conversation[]> {
    const entries = await readdir(path).catch(() => [] as string[]);
    const conversationFiles = entries.filter(isConversationFile);

    const conversations: Conversation[] = [];

    for (const name of conversationFiles) {
      const filePath = join(path, name);
      const fileStat = await stat(filePath).catch(() => null);
      if (!fileStat) continue;

      const mtimeMs = fileStat.mtimeMs;
      const birthtimeMs = fileStat.birthtimeMs;

      if (opts?.since && mtimeMs < opts.since) continue;

      const raw = await readFile(filePath, 'utf-8').catch(() => null);
      if (!raw) continue;

      let data: RawData;
      try {
        data = JSON.parse(raw) as RawData;
      } catch {
        continue;
      }

      if (!data.chunkedPrompt) continue;

      const conv = convertFile(filePath, data, mtimeMs, birthtimeMs);
      if (!conv) continue;

      conversations.push(conv);
      if (opts?.limit && conversations.length >= opts.limit) {
        return sortConversations(conversations);
      }
    }

    return sortConversations(conversations);
  },

  supportsIncremental: true,

  async parseSince(path: string, since: number): Promise<Conversation[]> {
    return this.parse(path, { since });
  },
};

runAdapter(adapter);
