/**
 * `/evolve` and the model-facing tools.
 *
 * The behaviour worth pinning is the refusal: with more than one scorer
 * registered and none named, this must not choose. A run measured against the
 * wrong objective produces a confidently wrong artifact and nothing downstream
 * can tell.
 */

import { Context } from '@deepseek-ai/cordis'
import CommandRegistry from '@deepseek-ai/dsh-commands'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRegistry from '@deepseek-ai/dsh-tools'
import { beforeEach, describe, expect, it } from 'vitest'

import { CommitQueue } from '../src/approval.js'
import {
  EvolutionRegistry, type ArtifactAdapter, type ArtifactState, type EvolutionEngine,
} from '../src/evolution.js'
import {
  describeChoices, parseEvolveInput, registerCommand, registerTools,
} from '../src/surface.js'

function adapter(id: string): ArtifactAdapter {
  return {
    id, blastRadius: 0.2,
    load: async () => ({}),
    bind: () => () => {},
    publish: async () => () => {},
  }
}

function engine(): EvolutionEngine & { started: unknown[]; cancelled: string[] } {
  const started: unknown[] = []
  const cancelled: string[] = []
  return {
    name: 'fake', started, cancelled,
    start: async (spec) => { started.push(spec); return 'run-1' },
    status: (runId) => (runId === 'run-1'
      ? { runId, artifact: 'skill:sql', phase: 'running' as const, round: 2,
          accepted: 1, rejected: 4, tokensSpent: 900 }
      : undefined),
    cancel: async (runId) => { cancelled.push(runId) },
    head: () => undefined,
    history: async () => [],
  }
}

describe('resolving what to evolve', () => {
  let ctx: Context
  let evolution: EvolutionRegistry

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    evolution.registerArtifact(adapter('skill:sql'))
  })

  it('resolves a single registration without being told', () => {
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    expect(parseEvolveInput(evolution, 'skill:sql')).toEqual({
      artifact: 'skill:sql', scorer: 'gold', taskSource: 'manual',
    })
  })

  it('refuses to choose between several scorers', () => {
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerScorer({ name: 'command', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    expect(() => parseEvolveInput(evolution, 'skill:sql'))
      .toThrow(/will not choose for you/)
  })

  it('takes the objective when it is named', () => {
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerScorer({ name: 'command', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    expect(parseEvolveInput(evolution, 'skill:sql command manual').scorer).toBe('command')
  })

  it('names an unknown artifact and lists what exists', () => {
    expect(() => parseEvolveInput(evolution, 'skill:ghost'))
      .toThrow(/unknown artifact "skill:ghost"/)
    expect(() => parseEvolveInput(evolution, 'skill:ghost')).toThrow(/skill:sql/)
  })

  it('asks for an artifact when given nothing', () => {
    expect(() => parseEvolveInput(evolution, '   ')).toThrow(/needs an artifact/)
  })

  it('describes empty registries honestly rather than as an empty list', () => {
    expect(describeChoices(evolution)).toContain('(none registered)')
  })
})

describe('/evolve', () => {
  let ctx: Context
  let evolution: EvolutionRegistry
  let fake: ReturnType<typeof engine>

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(CommandRegistry)
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    evolution.registerArtifact(adapter('skill:sql'))
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    fake = engine()
    evolution.registerEngine(fake)
    registerCommand(ctx)
  })

  async function run(input: string) {
    const definition = ctx.commands.find(undefined as never, 'evolve')
    return await definition!.handler({ name: 'evolve', rawInput: input } as never)
  }

  it('registers itself for discovery', () => {
    expect(ctx.commands.list(undefined as never).map((c) => c.name)).toContain('evolve')
  })

  it('starts a run and hands back how to follow it', async () => {
    const result = await run('skill:sql')
    expect(result.kind).toBe('success')
    expect(result.text).toContain('run-1')
    expect(fake.started).toEqual([{ artifact: 'skill:sql', scorer: 'gold', taskSource: 'manual' }])
  })

  it('reports a run rather than making the user parse ids', async () => {
    const result = await run('status run-1')
    expect(result.text).toContain('round 2')
    expect(result.text).toContain('1 accepted / 4 rejected')
  })

  it('says so for a run it does not have', async () => {
    expect(await run('status run-9')).toEqual({ kind: 'error', text: 'no run "run-9"' })
  })

  it('cancels', async () => {
    expect((await run('cancel run-1')).kind).toBe('success')
    expect(fake.cancelled).toEqual(['run-1'])
  })

  it('lists the choices when asked for nothing in particular', async () => {
    const result = await run('')
    expect(result.text).toContain('skill:sql')
    expect(result.text).toContain('gold')
  })

  it('turns a refusal into an error result instead of throwing at the UI', async () => {
    evolution.registerScorer({ name: 'command', score: async () => 1 })
    const result = await run('skill:sql')
    expect(result.kind).toBe('error')
    expect(result.text).toMatch(/will not choose for you/)
  })
})

describe('the model-facing tools', () => {
  let ctx: Context
  let evolution: EvolutionRegistry
  let fake: ReturnType<typeof engine>

  beforeEach(async () => {
    ctx = new Context()
    // ToolRegistry injects systemPrompt: a tool's schema joins prompt assembly,
    // so the registry cannot exist without somewhere to contribute it.
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRegistry)
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    evolution.registerArtifact(adapter('skill:sql'))
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    fake = engine()
    evolution.registerEngine(fake)
    registerTools(ctx)
  })

  it('offers the model a way to start and follow its own training run', () => {
    const names = ctx.tools.schemas().map((schema) => schema.name)
    expect(names).toContain('evolve_start')
    expect(names).toContain('evolve_status')
  })

  it('starts a run from a tool call', async () => {
    const tool = ctx.tools.get('evolve_start')
    const result = await (tool as any).execute({ artifact: 'skill:sql' }, {} as never)
    expect(JSON.stringify(result)).toContain('run-1')
    expect(fake.started).toHaveLength(1)
  })

})


describe('/evolve review', () => {
  let ctx: Context
  let evolution: EvolutionRegistry
  let queue: CommitQueue
  let published: ArtifactState[]

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(CommandRegistry)
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    published = []
    evolution.registerArtifact({
      id: 'plugin:mine', blastRadius: 0.6,
      load: async () => ({}),
      bind: () => () => {},
      publish: async (state) => { published.push(state); return () => {} },
    })
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    evolution.registerEngine(engine())
    queue = new CommitQueue()
    registerCommand(ctx, { queue })
  })

  async function run(input: string) {
    const definition = ctx.commands.find(undefined as never, 'evolve')
    return await definition!.handler({ name: 'evolve', rawInput: input } as never)
  }

  async function stage() {
    await queue.submit(evolution.artifact('plugin:mine')!, { 'src/index.ts': 'evolved' }, 2, {
      commitId: 'c1', runId: 'run-1', artifact: 'plugin:mine', blastRadius: 0.6,
      version: 2, heldOutBefore: 0.5, heldOutAfter: 0.8,
    })
  }

  it('says plainly when nothing is waiting', async () => {
    expect((await run('pending')).text).toContain('nothing waiting')
  })

  it('lists a staged commit with what it would buy', async () => {
    await stage()
    const result = await run('pending')
    expect(result.text).toContain('plugin:mine')
    expect(result.text).toContain('0.500 -> 0.800')
  })

  it('publishes only when a person approves', async () => {
    await stage()
    expect(published).toEqual([])
    const result = await run('approve pending-1')
    expect(result.kind).toBe('success')
    expect(published).toEqual([{ 'src/index.ts': 'evolved' }])
  })

  it('rejects without publishing, and says the ledger still has it', async () => {
    await stage()
    const result = await run('reject pending-1')
    expect(result.text).toContain('re-proposed')
    expect(published).toEqual([])
  })

  it('names an id it does not hold rather than doing nothing', async () => {
    expect((await run('approve pending-99')).kind).toBe('error')
    expect((await run('reject pending-99')).kind).toBe('error')
  })

  it('asks for an id when given none', async () => {
    expect((await run('approve')).text).toContain('<id>')
  })
})
