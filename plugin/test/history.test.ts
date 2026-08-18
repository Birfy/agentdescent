/**
 * Evolution's record, in the session log.
 *
 * Before this a run's history lived in the Python ledger and in memory, and a
 * reload emptied the second. The point of moving it into the log is that it
 * survives, replays and exports -- so the tests append against a real session
 * store and read the events back out of it.
 */

import { Context } from '@deepseek-ai/cordis'
import SessionStore from '@deepseek-ai/dsh-session'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { EvolutionRegistry, type CommitRecord, type EvolutionEngine, type RunStatus } from '../src/evolution.js'
import { evolutionHistory, recordHistory, type HistorySession } from '../src/history.js'

const COMMIT: CommitRecord = {
  commitId: 'c1', runId: 'run-1', artifact: 'skill:sql', blastRadius: 0.2,
  version: 3, heldOutBefore: 0.6, heldOutAfter: 0.82,
}

function status(phase: RunStatus['phase'], extra: Partial<RunStatus> = {}): RunStatus {
  return {
    runId: 'run-1', artifact: 'skill:sql', phase, round: 4,
    accepted: 1, rejected: 6, tokensSpent: 900, ...extra,
  }
}

function engine() {
  const listeners = new Set<(s: RunStatus) => void>()
  const fake: EvolutionEngine & { emit(s: RunStatus): void } = {
    name: 'fake',
    start: async () => 'run-1',
    status: () => undefined,
    cancel: async () => {},
    head: () => undefined,
    history: async () => [],
    onPhase: (listener) => {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    emit: (s) => { for (const listener of [...listeners]) listener(s) },
  }
  return fake
}

describe('recording evolution in the log', () => {
  let ctx: Context
  let evolution: EvolutionRegistry
  let fake: ReturnType<typeof engine>

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    fake = engine()
    evolution.registerEngine(fake)
  })

  /** A session that records what was appended, standing in for a live one. */
  function recorder(): HistorySession & { appended: Array<{ type: string; data: any }> } {
    const appended: Array<{ type: string; data: any }> = []
    return { appended, append: (type, data) => { appended.push({ type, data: data as any }); return data } }
  }

  it('writes a commit into the log', () => {
    const session = recorder()
    recordHistory(ctx, { evolution, session })
    evolution.commit(COMMIT)
    expect(session.appended).toHaveLength(1)
    expect(session.appended[0]?.type).toBe('evolution/commit')
    expect(session.appended[0]?.data).toMatchObject({
      artifact: 'skill:sql', version: 3, heldOutAfter: 0.82,
    })
  })

  it('writes a run ending, with what it cost', () => {
    const session = recorder()
    recordHistory(ctx, { evolution, session })
    fake.emit(status('done'))
    expect(session.appended).toHaveLength(1)
    expect(session.appended[0]?.type).toBe('evolution/run-ended')
    expect(session.appended[0]?.data).toMatchObject({
      phase: 'done', accepted: 1, rejected: 6, tokensSpent: 900,
    })
  })

  it('records why a run failed, which is the part worth keeping', () => {
    const session = recorder()
    recordHistory(ctx, { evolution, session })
    fake.emit(status('failed', { error: 'the sidecar died' }))
    expect(session.appended[0]?.data).toMatchObject({
      phase: 'failed', error: 'the sidecar died',
    })
  })

  it('does not record a run that is still going', () => {
    // A log entry per round would bury the two facts worth keeping.
    const session = recorder()
    recordHistory(ctx, { evolution, session })
    fake.emit(status('running'))
    fake.emit(status('starting'))
    expect(session.appended).toEqual([])
  })

  it('contains an append that fails, because the commit already happened', () => {
    // Losing the record of a change is bad; losing the change is worse. This
    // must never propagate into the path that announced the commit.
    const warn = vi.spyOn(ctx.logger, 'warn').mockImplementation(() => {})
    const broken: HistorySession = {
      append: () => { throw new Error('disk full') },
    }
    recordHistory(ctx, { evolution, session: broken })
    expect(() => evolution.commit(COMMIT)).not.toThrow()
    expect(warn).toHaveBeenCalled()
  })

  it('stops recording when disposed', () => {
    const session = recorder()
    const stop = recordHistory(ctx, { evolution, session })
    stop()
    evolution.commit(COMMIT)
    expect(session.appended).toEqual([])
  })

  it('picks its own events out of a mixed log', () => {
    const log = [
      { type: 'user/message', data: {} },
      { type: 'evolution/commit', data: COMMIT },
      { type: 'assistant/message', data: {} },
      { type: 'evolution/run-ended', data: { runId: 'run-1' } },
    ]
    expect(evolutionHistory(log).map((e) => e.type))
      .toEqual(['evolution/commit', 'evolution/run-ended'])
  })
})

describe('against a real session', () => {
  it('appends an event the log keeps and gives back', async () => {
    // The whole point is durability, so this goes through the real store rather
    // than a recorder: a typed event the log accepts, and returns on read.
    const ctx = new Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(EvolutionRegistry)
    const session = ctx.sessions.create()
    ctx.evolution.registerEngine(engine())
    recordHistory(ctx, { evolution: ctx.evolution, session: session as never })

    ctx.evolution.commit(COMMIT)

    const recorded = evolutionHistory(session.events as never)
    expect(recorded).toHaveLength(1)
    expect((recorded[0]?.data as any).commitId).toBe('c1')
  })

  it('records a terminal phase on the same session', async () => {
    const ctx = new Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(EvolutionRegistry)
    const session = ctx.sessions.create()
    const fake = engine()
    ctx.evolution.registerEngine(fake)
    recordHistory(ctx, { evolution: ctx.evolution, session: session as never })

    fake.emit(status('running'))
    fake.emit(status('done'))

    const recorded = evolutionHistory(session.events as never)
    expect(recorded.map((e) => e.type)).toEqual(['evolution/run-ended'])
    expect((recorded[0]?.data as any)).toMatchObject({ phase: 'done', tokensSpent: 900 })
  })
})
