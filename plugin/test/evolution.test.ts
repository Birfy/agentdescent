/**
 * The evolution seam, against a real Cordis context.
 *
 * The seam's whole job is to be the part that does *not* change when the
 * optimiser does, so the tests mount fakes through the public registration API
 * rather than reaching into the service.
 */

import { Context } from '@deepseek-ai/cordis'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  EvolutionRegistry,
  type ArtifactAdapter,
  type ArtifactState,
  type CommitRecord,
  type EvolutionEngine,
  type EvolutionSpec,
  type RunStatus,
  type Scorer,
  type TaskSource,
} from '../src/evolution.js'

function adapter(id: string, blastRadius = 0.2): ArtifactAdapter {
  return {
    id,
    blastRadius,
    load: async () => ({ 'SKILL.md': 'body' }),
    bind: () => () => {},
    publish: async () => () => {},
  }
}

function scorer(name: string, value = 1): Scorer {
  return { name, score: async () => value }
}

function taskSource(name: string): TaskSource {
  return { name, fetch: async (n) => Array.from({ length: n }, (_, i) => ({ id: `t${i}`, prompt: 'q' })) }
}

function engine(name = 'fake'): EvolutionEngine & { started: EvolutionSpec[] } {
  const started: EvolutionSpec[] = []
  const status: RunStatus = {
    runId: 'run-1', artifact: 'skill:sql', phase: 'running',
    round: 0, accepted: 0, rejected: 0, tokensSpent: 0,
  }
  return {
    name,
    started,
    start: async (spec) => { started.push(spec); return 'run-1' },
    status: () => status,
    cancel: async () => {},
    head: () => ({ artifact: 'skill:sql', version: 3, heldOut: 0.7, state: {} }),
    history: async () => [],
  }
}

const SPEC: EvolutionSpec = { artifact: 'skill:sql', taskSource: 'manual', scorer: 'gold' }

describe('the evolution seam', () => {
  let ctx: Context
  let evolution: EvolutionRegistry

  beforeEach(async () => {
    ctx = new Context()
    // `ctx.plugin()` returns an awaitable fiber; the service is published when
    // it settles, not when the call returns.
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
  })

  it('publishes itself on the context under `evolution`', () => {
    expect(evolution).toBeInstanceOf(EvolutionRegistry)
  })

  describe('without an engine', () => {
    it('says what to mount instead of failing obscurely', async () => {
      await expect(evolution.start(SPEC)).rejects.toThrow(/dsh-evolution-agentdescent/)
      expect(() => evolution.status('run-1')).toThrow(/no evolution engine/)
      expect(() => evolution.head('skill:sql')).toThrow(/no evolution engine/)
    })

    it('still accepts registrations, because they are what an engine consumes', () => {
      expect(() => evolution.registerArtifact(adapter('skill:sql'))).not.toThrow()
      expect(evolution.listArtifacts()).toEqual(['skill:sql'])
    })
  })

  describe('the engine is sole', () => {
    it('refuses a second one at mount rather than at the first run', () => {
      evolution.registerEngine(engine('first'))
      expect(() => evolution.registerEngine(engine('second')))
        .toThrow(/one ledger takes one optimiser/)
    })

    it('frees the slot when the first is disposed', () => {
      const dispose = evolution.registerEngine(engine('first'))
      dispose()
      expect(() => evolution.registerEngine(engine('second'))).not.toThrow()
    })
  })

  describe('registries', () => {
    it('rejects a duplicate id, name, or source', () => {
      evolution.registerArtifact(adapter('skill:sql'))
      evolution.registerScorer(scorer('gold'))
      evolution.registerTaskSource(taskSource('manual'))
      expect(() => evolution.registerArtifact(adapter('skill:sql'))).toThrow(/already registered/)
      expect(() => evolution.registerScorer(scorer('gold'))).toThrow(/already registered/)
      expect(() => evolution.registerTaskSource(taskSource('manual'))).toThrow(/already registered/)
    })

    it('unregisters on dispose and lets the name be reused', () => {
      const dispose = evolution.registerScorer(scorer('gold'))
      expect(evolution.scorer('gold')).toBeDefined()
      dispose()
      expect(evolution.scorer('gold')).toBeUndefined()
      expect(() => evolution.registerScorer(scorer('gold'))).not.toThrow()
    })

    it('lists names sorted, for surfaces that show the choices', () => {
      evolution.registerScorer(scorer('replay-pairwise'))
      evolution.registerScorer(scorer('command'))
      evolution.registerScorer(scorer('gold'))
      expect(evolution.listScorers()).toEqual(['command', 'gold', 'replay-pairwise'])
    })
  })

  describe('start()', () => {
    beforeEach(() => {
      evolution.registerArtifact(adapter('skill:sql'))
      evolution.registerScorer(scorer('gold'))
      evolution.registerTaskSource(taskSource('manual'))
    })

    it('reaches the engine with the spec once everything resolves', async () => {
      const fake = engine()
      evolution.registerEngine(fake)
      await expect(evolution.start(SPEC)).resolves.toBe('run-1')
      expect(fake.started).toEqual([SPEC])
    })

    it('names the unknown thing and what is registered', async () => {
      const fake = engine()
      evolution.registerEngine(fake)
      await expect(evolution.start({ ...SPEC, scorer: 'typo' }))
        .rejects.toThrow(/unknown scorer "typo"\. Registered: gold/)
      await expect(evolution.start({ ...SPEC, artifact: 'skill:absent' }))
        .rejects.toThrow(/unknown artifact "skill:absent"/)
      await expect(evolution.start({ ...SPEC, taskSource: 'nope' }))
        .rejects.toThrow(/unknown task source "nope"/)
      // A rejected spec must not reach the engine at all.
      expect(fake.started).toEqual([])
    })

    it('resolves names before the engine sees them, so the message is engine-independent', async () => {
      // No engine mounted: the *engine* error wins, because there is nothing to
      // validate a spec for. Registration mistakes surface once one exists.
      await expect(evolution.start({ ...SPEC, scorer: 'typo' }))
        .rejects.toThrow(/no evolution engine/)
    })
  })

  describe('commits', () => {
    const record: CommitRecord = {
      commitId: 'c1', runId: 'run-1', artifact: 'skill:sql', blastRadius: 0.2,
      version: 4, heldOutBefore: 0.6, heldOutAfter: 0.75,
    }

    it('reaches every listener', () => {
      const seen: CommitRecord[] = []
      evolution.onCommit((r) => seen.push(r))
      evolution.onCommit((r) => seen.push(r))
      evolution.commit(record)
      expect(seen).toEqual([record, record])
    })

    it('stops after dispose', () => {
      const seen: CommitRecord[] = []
      const off = evolution.onCommit((r) => seen.push(r))
      off()
      evolution.commit(record)
      expect(seen).toEqual([])
    })

    it('contains a throwing listener, because the commit already happened', () => {
      const warn = vi.spyOn(ctx.logger, 'warn').mockImplementation(() => {})
      const seen: CommitRecord[] = []
      evolution.onCommit(() => { throw new Error('render failed') })
      evolution.onCommit((r) => seen.push(r))
      expect(() => evolution.commit(record)).not.toThrow()
      expect(seen).toEqual([record])          // the later listener is not starved
      expect(warn).toHaveBeenCalled()
    })
  })

  describe('the adapter contract', () => {
    it('binds a candidate to a scope and unbinds on dispose', async () => {
      let bound: ArtifactState | undefined
      const scoped: ArtifactAdapter = {
        ...adapter('skill:sql'),
        bind: (_scope, state) => { bound = state; return () => { bound = undefined } },
      }
      evolution.registerArtifact(scoped)
      const undo = evolution.artifact('skill:sql')!.bind(ctx, { 'SKILL.md': 'candidate' })
      expect(bound).toEqual({ 'SKILL.md': 'candidate' })
      undo()
      expect(bound).toBeUndefined()
    })

    it('carries a blast radius, which is what picks the governance layer', () => {
      evolution.registerArtifact(adapter('skill:sql', 0.2))
      evolution.registerArtifact(adapter('plugin:mine', 0.6))
      expect(evolution.artifact('skill:sql')!.blastRadius).toBeLessThan(0.5)
      expect(evolution.artifact('plugin:mine')!.blastRadius).toBeGreaterThanOrEqual(0.5)
    })
  })
})
