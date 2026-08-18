/**
 * Your own sessions as the training set.
 *
 * This is the thing the harness has that a benchmark does not. dsh guarantees
 * that model-visible means logged, so every turn you have ever taken is a task
 * with a recorded answer beside it -- and that recorded answer is a free
 * reference for the fallback scorer to compare a candidate against.
 *
 * It is also the most intrusive source in the package, so it is off unless
 * asked for, and it never leaves the machine on its own account. What it does
 * do, and what the user must be told once: **replaying a past turn sends that
 * content to the model provider again.** It is the same content that went there
 * when it was typed, but this is a new send, and hiding that would be the kind
 * of quiet decision this whole design keeps refusing to make.
 *
 * @module dsh-plugin-agentdescent/transcripts
 */

import type { EvolutionTask, RolloutResult, TaskSource } from './evolution.js'

/** What this module needs of `ctx.sessionQuery`, narrowed and checked in tests. */
export interface TranscriptSource {
  listSessions(signal?: AbortSignal): Promise<readonly TranscriptSessionRecord[]>
  readSurface(sessionId: string): Promise<TranscriptSurface>
}

export interface TranscriptSessionRecord {
  readonly header: { readonly id: string; readonly createdAt: number; readonly cwd?: string }
}

export interface TranscriptSurface {
  readonly events: readonly any[]
}

export interface TranscriptOptions {
  readonly name?: string
  readonly query: TranscriptSource
  /**
   * Most turns to take from any one session.
   *
   * Without a cap, one long afternoon becomes the whole training set and the
   * artifact learns that afternoon rather than the work.
   */
  readonly maxPerSession?: number
  /** Sessions to consider, newest first. */
  readonly maxSessions?: number
  /** Shortest prompt worth training on. */
  readonly minPromptChars?: number
  /** Drop a turn whose text matches. Defaults to obvious secret shapes. */
  readonly exclude?: readonly RegExp[]
}

const DEFAULT_MAX_PER_SESSION = 5
const DEFAULT_MAX_SESSIONS = 50
const DEFAULT_MIN_PROMPT_CHARS = 12

/**
 * Shapes that must never become a task.
 *
 * Not a secret scanner, and not presented as one -- it is a coarse filter for
 * the cases that would otherwise put a credential into a prompt that gets
 * replayed. A real scan belongs to whoever runs this on a real corpus.
 */
const DEFAULT_EXCLUDE: readonly RegExp[] = [
  /\b(?:api[_-]?key|secret|password|passwd|token|bearer)\b\s*[:=]/i,
  /\b(?:sk|pk)-[A-Za-z0-9]{16,}\b/,
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
]

/** A session this package created. Training on our own rollouts is a loop. */
function isOurs(id: string): boolean {
  return id.startsWith('evolve-')
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .filter((part: any) => part?.type === 'text' && typeof part.text === 'string')
    .map((part: any) => part.text)
    .join('')
}

/**
 * Tasks drawn from what the user actually did.
 *
 * A task is one user turn; its reference is the assistant text that followed,
 * which is what `replayPairwise` compares a candidate against. Newest sessions
 * first, because a skill is judged on the work in front of the user now -- the
 * distribution drifts, and last quarter's project is not the objective.
 */
export function transcriptTaskSource(options: TranscriptOptions): TaskSource {
  const maxPerSession = options.maxPerSession ?? DEFAULT_MAX_PER_SESSION
  const maxSessions = options.maxSessions ?? DEFAULT_MAX_SESSIONS
  const minChars = options.minPromptChars ?? DEFAULT_MIN_PROMPT_CHARS
  const exclude = options.exclude ?? DEFAULT_EXCLUDE

  return {
    name: options.name ?? 'transcripts',
    fetch: async (count: number, signal?: AbortSignal) => {
      const sessions = (await options.query.listSessions(signal))
        .filter((record) => !isOurs(record.header.id))
        .slice(0, maxSessions)
      const tasks: EvolutionTask[] = []
      for (const record of sessions) {
        if (tasks.length >= count) break
        let surface
        try {
          surface = await options.query.readSurface(record.header.id)
        } catch {
          // One unreadable session must not lose the others: a corrupt or
          // half-written log is ordinary, and the corpus is the point.
          continue
        }
        tasks.push(...harvest(record, surface, {
          limit: Math.min(maxPerSession, count - tasks.length), minChars, exclude,
        }))
      }
      return tasks.slice(0, count)
    },
  }
}

interface HarvestOptions {
  readonly limit: number
  readonly minChars: number
  readonly exclude: readonly RegExp[]
}

function harvest(record: TranscriptSessionRecord, surface: TranscriptSurface,
                 options: HarvestOptions): EvolutionTask[] {
  const tasks: EvolutionTask[] = []
  let pending: { prompt: string; index: number } | undefined

  const flush = (answer: string): void => {
    if (pending === undefined || tasks.length >= options.limit) return
    const reference: RolloutResult = { output: answer }
    tasks.push({
      id: `${record.header.id}#${pending.index}`,
      prompt: pending.prompt,
      meta: { reference, sessionId: record.header.id, createdAt: record.header.createdAt },
    })
    pending = undefined
  }

  surface.events.forEach((event: any, index: number) => {
    const type = String(event?.type ?? '')
    if (type === 'user/message') {
      const source = String(event?.data?.message?.source?.kind ?? 'user')
      // Only what a person typed. Injected context, skill invocations and
      // command payloads are the harness talking to itself, and training on
      // them teaches the artifact to answer its own machinery.
      if (source !== 'user') return
      const prompt = textOf(event?.data?.message?.content).trim()
      if (prompt.length < options.minChars) return
      if (options.exclude.some((pattern) => pattern.test(prompt))) return
      pending = { prompt, index }
    } else if (type === 'assistant/message' && pending !== undefined) {
      const answer = textOf(event?.data?.message?.content).trim()
      if (options.exclude.some((pattern) => pattern.test(answer))) {
        pending = undefined            // the answer carries a secret; drop the pair
        return
      }
      flush(answer)
    }
  })
  return tasks
}
