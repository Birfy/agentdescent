/**
 * An evolution run as a background job.
 *
 * The point is that a run joins the vocabulary the harness already has -- a
 * model can list it, read it and kill it with the same tools it uses for a long
 * bash command. So the tests are about the lifecycle contract the registry
 * demands: a job that settles exactly once, on an edge rather than a poll, with
 * the right one of three outcomes.
 */

import { Context } from '@deepseek-ai/cordis'
import { beforeEach, describe, expect, it } from 'vitest'

import {
  EvolutionRegistry, type EvolutionEngine, type RunStatus,
} from '../src/evolution.js'
import { startAsJob, type JobHost } from '../src/jobs.js'

const SPEC = { artifact: 'skill:sql', taskSource: 'manual', scorer: 'gold' }

function status(phase: RunStatus['phase'], extra: Partial<RunStatus> = {}): RunStatus {
  return {
    runId: 'run-1', artifact: 'skill:sql', phase, round: 1,
    accepted: 2, rejected: 3, tokensSpent: 40, ...extra,
  }
}

/** An engine whose phases the test drives by hand. */
function engine(withPhases = true) {
  const listeners = new Set<(s: RunStatus) => void>()
  const cancelled: string[] = []
  const fake: EvolutionEngine & { emit(s: RunStatus): void; cancelled: string[] } = {
    name: 'fake',
    cancelled,
    start: async () => 'run-1',
    status: () => undefined,
    cancel: async (runId) => { cancelled.push(runId) },
    head: () => undefined,
    history: async () => [],
    emit: (s) => { for (const listener of [...listeners]) listener(s) },
  }
  if (withPhases) {
    fake.onPhase = (listener) => {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    }
  }
  return fake
}

/** A stand-in registry that records what a producer handed it. */
function host(): JobHost & { started: any[]; hooks: any[] } {
  const started: any[] = []
  const hooks: any[] = []
  return {
    started, hooks,
    start: (spec: any) => {
      started.push(spec)
      hooks.push(spec.run())
      return `evolution-${started.length}`
    },
  }
}

describe('an evolution run as a job', () => {
  let ctx: Context
  let evolution: EvolutionRegistry

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    evolution.registerArtifact({
      id: 'skill:sql', blastRadius: 0.2,
      load: async () => ({}), bind: () => () => {}, publish: async () => () => {},
    })
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
  })

  it('registers under its own kind, labelled with what it is evolving', async () => {
    evolution.registerEngine(engine())
    const jobs = host()
    const { jobId } = await startAsJob(ctx, SPEC, { evolution, jobs })
    expect(jobId).toBe('evolution-1')
    expect(jobs.started[0].kind).toBe('evolution')
    expect(jobs.started[0].label).toContain('skill:sql')
  })

  it('settles when the run reaches a terminal phase', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })

    fake.emit(status('running'))
    fake.emit(status('done'))
    await expect(jobs.hooks[0].done).resolves.toMatchObject({ status: 'completed' })
  })

  it('reports a cancelled run as killed, not failed', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    fake.emit(status('cancelled'))
    await expect(jobs.hooks[0].done).resolves.toMatchObject({ status: 'killed' })
  })

  it('reports a spent budget as completed, because the ceiling working is not a fault', async () => {
    // Calling it failed would teach a model to raise the budget.
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    fake.emit(status('budget-exhausted'))
    await expect(jobs.hooks[0].done).resolves
      .toMatchObject({ status: 'completed', detail: 'token budget spent' })
  })

  it('carries the failure reason through', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    fake.emit(status('failed', { error: 'the sidecar died' }))
    await expect(jobs.hooks[0].done).resolves
      .toMatchObject({ status: 'failed', detail: 'the sidecar died' })
  })

  it('points a staged commit at the place it is waiting', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    fake.emit(status('awaiting-review'))
    await expect(jobs.hooks[0].done).resolves
      .toMatchObject({ status: 'completed', detail: expect.stringContaining('/evolve pending') })
  })

  it('hands a reader the progress since they last looked', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    fake.emit(status('running', { round: 1 }))
    fake.emit(status('running', { round: 2 }))
    const first = jobs.hooks[0].readOutput()
    expect(first).toContain('round 1')
    expect(first).toContain('round 2')
    expect(jobs.hooks[0].readOutput()).toBe('')   // one consuming cursor
  })

  it('cancels the run when the job is killed', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    jobs.hooks[0].cancel('user asked')
    await new Promise((resolve) => setImmediate(resolve))
    expect(fake.cancelled).toEqual(['run-1'])
  })

  it('ignores phases belonging to another run', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    fake.emit(status('done', { runId: 'someone-else' }))
    let settled = false
    void jobs.hooks[0].done.then(() => { settled = true })
    await new Promise((resolve) => setImmediate(resolve))
    expect(settled).toBe(false)
  })

  it('settles once, not once per terminal phase', async () => {
    const fake = engine()
    evolution.registerEngine(fake)
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    fake.emit(status('done'))
    fake.emit(status('failed', { error: 'later noise' }))
    await expect(jobs.hooks[0].done).resolves.toMatchObject({ status: 'completed' })
  })

  it('fails immediately against an engine that reports no phases', async () => {
    // A job that never settles is worse than one that never started: the
    // registry waits on it forever and nothing says why.
    evolution.registerEngine(engine(false))
    const jobs = host()
    await startAsJob(ctx, SPEC, { evolution, jobs })
    await expect(jobs.hooks[0].done).resolves
      .toMatchObject({ status: 'failed', detail: expect.stringContaining('no phase changes') })
  })
})
