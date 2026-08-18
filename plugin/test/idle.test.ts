/**
 * Starting a run when nobody is looking, and getting out of the way when they
 * come back.
 *
 * Every rule here exists because of a specific way this goes wrong, and each
 * one is tested by the wrong behaviour rather than the right one: competing
 * with the user for the model they are waiting on, spending their tokens after
 * they return, and burning money overnight on a laptop left open.
 */

import { Context } from '@deepseek-ai/cordis'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { EvolutionRegistry, type EvolutionEngine, type EvolutionSpec } from '../src/evolution.js'
import { idleTrigger } from '../src/idle.js'

const SPEC: EvolutionSpec = { artifact: 'skill:sql', taskSource: 'manual', scorer: 'gold' }

function engine(): EvolutionEngine & { started: EvolutionSpec[]; cancelled: string[] } {
  const started: EvolutionSpec[] = []
  const cancelled: string[] = []
  return {
    name: 'fake', started, cancelled,
    start: async (spec) => { started.push(spec); return `run-${started.length}` },
    status: () => undefined,
    cancel: async (runId) => { cancelled.push(runId) },
    head: () => undefined,
    history: async () => [],
  }
}

describe('the idle trigger', () => {
  let ctx: Context
  let evolution: EvolutionRegistry
  let fake: ReturnType<typeof engine>
  let clock: number

  beforeEach(async () => {
    vi.useFakeTimers()
    clock = 1_000_000
    ctx = new Context()
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    evolution.registerArtifact({
      id: 'skill:sql', blastRadius: 0.2,
      load: async () => ({}), bind: () => () => {}, publish: async () => () => {},
    })
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    fake = engine()
    evolution.registerEngine(fake)
  })

  function arm(overrides = {}) {
    return idleTrigger(ctx, {
      evolution, runs: [SPEC], afterMs: 1000, minIntervalMs: 10_000,
      now: () => clock, ...overrides,
    })
  }

  function status(id: string, state: 'idle' | 'running') {
    ;(ctx as any).emit('agent/status', { agent: { id }, status: state })
  }

  it('starts a run once the harness has been quiet long enough', async () => {
    arm()
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started).toHaveLength(1)
    expect(fake.started[0]?.artifact).toBe('skill:sql')
  })

  it('runs barrier-free, so it can stop at a rollout boundary', async () => {
    arm()
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started[0]?.asynchronous).toBe(true)
  })

  it('waits while any agent is busy, not just the one it saw last', async () => {
    // Watching a single agent would start a training run while the user is
    // mid-task in another session, competing for the model they are waiting on.
    arm()
    status('user-a', 'running')
    status('user-b', 'running')
    await vi.advanceTimersByTimeAsync(5000)
    expect(fake.started).toEqual([])

    status('user-a', 'idle')
    await vi.advanceTimersByTimeAsync(5000)
    expect(fake.started).toEqual([])            // b is still working

    status('user-b', 'idle')
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started).toHaveLength(1)
  })

  it('stops a running evolution the moment the user comes back', async () => {
    arm()
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started).toHaveLength(1)
    status('user-a', 'running')
    await vi.advanceTimersByTimeAsync(0)
    expect(fake.cancelled).toEqual(['run-1'])
  })

  it('ignores its own rollout children as activity', async () => {
    // Counting them would make the trigger cancel the run it just started.
    arm()
    await vi.advanceTimersByTimeAsync(1000)
    status('evolve-skill-sql-t0-1', 'running')
    await vi.advanceTimersByTimeAsync(0)
    expect(fake.cancelled).toEqual([])
  })

  it('keeps a minimum gap between automatic runs', async () => {
    // Without it, a machine left alone overnight starts a run every time the
    // previous one ends -- a plausible way to spend a lot of money by leaving a
    // laptop open.
    arm()
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started).toHaveLength(1)

    status('user', 'running')                   // clears the active run
    status('user', 'idle')
    clock += 5000                               // less than minIntervalMs
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started).toHaveLength(1)

    status('user', 'running')
    status('user', 'idle')
    clock += 10_000                             // now past it
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started).toHaveLength(2)
  })

  it('starts nothing when nothing was configured to run', async () => {
    arm({ runs: [] })
    await vi.advanceTimersByTimeAsync(10_000)
    expect(fake.started).toEqual([])
  })

  it('rotates through the configured runs instead of only ever the first', async () => {
    const second: EvolutionSpec = { ...SPEC, artifact: 'skill:other' }
    evolution.registerArtifact({
      id: 'skill:other', blastRadius: 0.2,
      load: async () => ({}), bind: () => () => {}, publish: async () => () => {},
    })
    arm({ runs: [SPEC, second] })
    await vi.advanceTimersByTimeAsync(1000)
    status('user', 'running'); status('user', 'idle')
    clock += 20_000
    await vi.advanceTimersByTimeAsync(1000)
    expect(fake.started.map((s) => s.artifact)).toEqual(['skill:sql', 'skill:other'])
  })

  it('survives a start that fails, and tries again later', async () => {
    const warn = vi.spyOn(ctx.logger, 'warn').mockImplementation(() => {})
    arm({ runs: [{ ...SPEC, scorer: 'missing' }] })
    await vi.advanceTimersByTimeAsync(1000)
    expect(warn).toHaveBeenCalled()
    status('user', 'running'); status('user', 'idle')
    clock += 20_000
    await vi.advanceTimersByTimeAsync(1000)
    expect(warn).toHaveBeenCalledTimes(2)       // still armed, not wedged
  })

  it('stops watching when disposed', async () => {
    const off = arm()
    off()
    await vi.advanceTimersByTimeAsync(10_000)
    expect(fake.started).toEqual([])
  })
})
