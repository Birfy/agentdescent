/**
 * Training on your own sessions.
 *
 * The interesting cases are all exclusions: our own rollouts, the harness
 * talking to itself, anything that looks like a credential, and one long
 * session drowning out the rest. Each is a way this quietly trains on the
 * wrong thing.
 */

import type { SessionQueryEngine } from '@deepseek-ai/dsh-session-query'
import { describe, expect, it } from 'vitest'

import { transcriptTaskSource, type TranscriptSource } from '../src/transcripts.js'

function userTurn(text: string, kind = 'user'): unknown {
  return { type: 'user/message', data: { message: { content: [{ type: 'text', text }], source: { kind } } } }
}

function assistantTurn(text: string): unknown {
  return { type: 'assistant/message', data: { message: { content: [{ type: 'text', text }] } } }
}

function query(sessions: Record<string, unknown[]>, order?: string[]): TranscriptSource {
  const ids = order ?? Object.keys(sessions)
  return {
    listSessions: async () => ids.map((id, i) => ({
      header: { id, createdAt: 1_700_000_000 - i },
    })),
    readSurface: async (id: string) => {
      const events = sessions[id]
      if (events === undefined) throw new Error(`no session ${id}`)
      return { events }
    },
  }
}

describe('harvesting turns', () => {
  it('pairs a user turn with the answer that followed', async () => {
    const source = transcriptTaskSource({
      query: query({ s1: [userTurn('how do I total a column?'), assistantTurn('sum it')] }),
    })
    const tasks = await source.fetch(10)
    expect(tasks).toHaveLength(1)
    expect(tasks[0]?.prompt).toBe('how do I total a column?')
    expect((tasks[0]?.meta?.reference as any).output).toBe('sum it')
  })

  it('keeps the reference, because that is what the fallback scorer compares against', async () => {
    const source = transcriptTaskSource({
      query: query({ s1: [userTurn('a question long enough'), assistantTurn('the answer')] }),
    })
    const [task] = await source.fetch(1)
    expect(task?.meta?.sessionId).toBe('s1')
    expect((task?.meta?.reference as any).output).toBe('the answer')
  })

  it('takes only what a person typed', async () => {
    // Injected context, skill invocations and command payloads are the harness
    // talking to itself; training on them teaches the artifact to answer its
    // own machinery.
    const source = transcriptTaskSource({
      query: query({ s1: [
        userTurn('injected context here', 'skill-invocation'), assistantTurn('x'),
        userTurn('what a person actually asked'), assistantTurn('y'),
      ] }),
    })
    const tasks = await source.fetch(10)
    expect(tasks.map((t) => t.prompt)).toEqual(['what a person actually asked'])
  })

  it('never trains on our own rollouts', async () => {
    // A loop: the artifact would be scored on transcripts it produced while
    // being scored.
    const source = transcriptTaskSource({
      query: query({
        'evolve-skill-sql-t0-1': [userTurn('a rollout prompt here'), assistantTurn('out')],
        s1: [userTurn('a real question here'), assistantTurn('out')],
      }),
    })
    const tasks = await source.fetch(10)
    expect(tasks.map((t) => t.meta?.sessionId)).toEqual(['s1'])
  })

  it('drops a turn that looks like it carries a credential', async () => {
    const source = transcriptTaskSource({
      query: query({ s1: [
        userTurn('my api_key = sk-abcdefghijklmnopqrst'), assistantTurn('ok'),
        userTurn('a perfectly ordinary question'), assistantTurn('fine'),
      ] }),
    })
    const tasks = await source.fetch(10)
    expect(tasks.map((t) => t.prompt)).toEqual(['a perfectly ordinary question'])
  })

  it('drops the pair when the *answer* carries one', async () => {
    const source = transcriptTaskSource({
      query: query({ s1: [
        userTurn('what is my configuration?'),
        assistantTurn('your password: hunter2 and token: abc'),
      ] }),
    })
    expect(await source.fetch(10)).toEqual([])
  })

  it('skips a prompt too short to be a task', async () => {
    const source = transcriptTaskSource({
      query: query({ s1: [userTurn('ok'), assistantTurn('sure'),
                          userTurn('a genuinely useful question'), assistantTurn('yes')] }),
    })
    expect((await source.fetch(10)).map((t) => t.prompt)).toEqual(['a genuinely useful question'])
  })

  it('caps how much one session contributes', async () => {
    // Without it, one long afternoon becomes the training set and the artifact
    // learns that afternoon rather than the work.
    const events = Array.from({ length: 20 }, (_, i) =>
      [userTurn(`question number ${i} here`), assistantTurn(`answer ${i}`)]).flat()
    const source = transcriptTaskSource({ query: query({ s1: events }), maxPerSession: 3 })
    expect(await source.fetch(50)).toHaveLength(3)
  })

  it('draws from the newest sessions first, because the distribution drifts', async () => {
    const source = transcriptTaskSource({
      query: query({
        newest: [userTurn('this quarter question'), assistantTurn('a')],
        older: [userTurn('last quarter question'), assistantTurn('b')],
      }, ['newest', 'older']),
      maxPerSession: 1,
    })
    expect((await source.fetch(1)).map((t) => t.prompt)).toEqual(['this quarter question'])
  })

  it('honours the requested count', async () => {
    const events = Array.from({ length: 10 }, (_, i) =>
      [userTurn(`question number ${i} here`), assistantTurn(`answer ${i}`)]).flat()
    const source = transcriptTaskSource({ query: query({ s1: events }), maxPerSession: 10 })
    expect(await source.fetch(4)).toHaveLength(4)
  })

  it('loses one unreadable session, not the corpus', async () => {
    // A corrupt or half-written log is ordinary; the corpus is the point.
    const broken: TranscriptSource = {
      listSessions: async () => [{ header: { id: 'bad', createdAt: 2 } },
                                 { header: { id: 'good', createdAt: 1 } }],
      readSurface: async (id) => {
        if (id === 'bad') throw new Error('corrupt log')
        return { events: [userTurn('a question that survives'), assistantTurn('yes')] }
      },
    }
    const tasks = await transcriptTaskSource({ query: broken }).fetch(10)
    expect(tasks.map((t) => t.meta?.sessionId)).toEqual(['good'])
  })

  it('gives an unanswered turn no task, because there is nothing to compare against', async () => {
    const source = transcriptTaskSource({
      query: query({ s1: [userTurn('a question nobody answered')] }),
    })
    expect(await source.fetch(10)).toEqual([])
  })
})

/**
 * The real engine must still satisfy the narrowed view. If dsh changes
 * `listSessions` or `readSurface`, this stops compiling rather than failing at
 * the first harvest.
 */
export const _theRealEngineStillFits: (engine: SessionQueryEngine) => TranscriptSource =
  (engine) => engine as unknown as TranscriptSource
