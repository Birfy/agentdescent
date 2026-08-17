/**
 * The engine provider: the seam's `start`/`status`/`head` on one side, and the
 * callbacks the optimiser drives on the other.
 *
 * Most of these run against a fake sidecar -- a second Peer -- because the
 * behaviour under test is this side's routing. The last one spawns the real
 * Python engine, which is the only way to know the two halves actually agree.
 */

import { spawn } from 'node:child_process'
import { PassThrough } from 'node:stream'

import { Context } from '@deepseek-ai/cordis'
import SkillRegistry from '@deepseek-ai/dsh-skill'
import { beforeEach, describe, expect, it } from 'vitest'

import { Peer } from '../src/bridge.js'
import { startEngine, type Sidecar } from '../src/engine.js'
import {
  EvolutionRegistry,
  type ArtifactAdapter,
  type ArtifactState,
  type CommitRecord,
} from '../src/evolution.js'

const START = '---\ndescription: total a column\n---\n\nCOLUMN: id\n'

/** A sidecar made of streams, plus the peer standing in for the optimiser. */
function fakeSidecar(): { sidecar: Sidecar; engineSide: Peer; killed: () => boolean } {
  const toEngine = new PassThrough()
  const toHarness = new PassThrough()
  let dead = false
  const engineSide = new Peer(toEngine, toHarness)
  engineSide.handle('handshake', () => ({ protocol: 1, engine: 'fake', version: '0.0.0' }))
  return {
    sidecar: { stdin: toEngine, stdout: toHarness, kill: () => { dead = true } },
    engineSide,
    killed: () => dead,
  }
}

function adapter(id: string, published: ArtifactState[]): ArtifactAdapter {
  return {
    id,
    blastRadius: 0.2,
    load: async () => ({ 'SKILL.md': START }),
    bind: () => () => {},
    publish: async (state) => { published.push(state); return () => {} },
  }
}

describe('the engine provider', () => {
  let ctx: Context
  let evolution: EvolutionRegistry
  let published: ArtifactState[]

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(SkillRegistry)
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    published = []
    evolution.registerArtifact(adapter('skill:total', published))
    evolution.registerScorer({ name: 'gold', score: async () => 1 })
    evolution.registerTaskSource({
      name: 'manual',
      fetch: async (n) => Array.from({ length: n }, (_, i) => ({ id: `t${i}`, prompt: 'q' })),
    })
  })

  async function start(sidecar: Sidecar, overrides = {}) {
    return await startEngine(ctx, {
      evolution,
      rollout: async (_artifact, state) => ({
        output: state['SKILL.md']?.includes('amount') === true ? 'correct' : 'wrong',
        finishReason: 'completed',
        steps: 1,
        tokens: 10,
      }),
      complete: async () => 'no edit',
      spawnSidecar: () => sidecar,
      ...overrides,
    })
  }

  it('handshakes and reports the engine it reached', async () => {
    const { sidecar } = fakeSidecar()
    const engine = await start(sidecar)
    expect(engine.name).toBe('fake@0.0.0')
  })

  it('refuses a protocol mismatch instead of negotiating', async () => {
    // Two versions that half-understand each other produce measurements that
    // look fine and are not -- worse than a process that will not start.
    const { sidecar, engineSide, killed } = fakeSidecar()
    engineSide.handle('handshake', () => ({ protocol: 99, engine: 'fake', version: '9.9' }))
    await expect(start(sidecar)).rejects.toThrow(/protocol mismatch/)
    expect(killed()).toBe(true)
  })

  it('forwards run.start and returns the run id', async () => {
    const { sidecar, engineSide } = fakeSidecar()
    const specs: unknown[] = []
    engineSide.handle('run.start', (spec) => { specs.push(spec); return 'run-7' })
    const engine = await start(sidecar)
    const spec = { artifact: 'skill:total', taskSource: 'manual', scorer: 'gold' }
    await expect(engine.start(spec)).resolves.toBe('run-7')
    expect(specs).toEqual([spec])
  })

  describe('the callbacks the optimiser drives', () => {
    it('loads an artifact through its adapter', async () => {
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar)
      await expect(engineSide.call('artifact.load', { artifact: 'skill:total' }))
        .resolves.toEqual({ 'SKILL.md': START })
    })

    it('names an unknown artifact rather than loading nothing', async () => {
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar)
      await expect(engineSide.call('artifact.load', { artifact: 'skill:absent' }))
        .rejects.toThrow(/unknown artifact "skill:absent"/)
    })

    it('fetches tasks from the named source', async () => {
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar)
      const tasks = await engineSide.call<unknown[]>('tasks.fetch', { source: 'manual', count: 3 })
      expect(tasks).toHaveLength(3)
    })

    it('executes a rollout against the candidate state it was handed', async () => {
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar)
      const wrong = await engineSide.call<{ output: string }>('rollout.execute', {
        artifact: 'skill:total', state: { 'SKILL.md': START }, task: { id: 't0', prompt: 'q' },
      })
      const right = await engineSide.call<{ output: string }>('rollout.execute', {
        artifact: 'skill:total', state: { 'SKILL.md': 'COLUMN: amount' }, task: { id: 't0', prompt: 'q' },
      })
      expect(wrong.output).toBe('wrong')
      expect(right.output).toBe('correct')
    })

    it('scores through the registered scorer', async () => {
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar)
      await expect(engineSide.call('reward.score', {
        scorer: 'gold', task: { id: 't0', prompt: 'q' }, run: { output: 'x' },
      })).resolves.toBe(1)
    })

    it('refuses to guess when a score names no scorer and several exist', async () => {
      evolution.registerScorer({ name: 'command', score: async () => 0 })
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar)
      await expect(engineSide.call('reward.score', {
        task: { id: 't0', prompt: 'q' }, run: { output: 'x' },
      })).rejects.toThrow(/must send one/)
    })

    it('runs completions on this side, so the engine never holds a key', async () => {
      const prompts: string[] = []
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar, {
        complete: async (prompt: string) => { prompts.push(prompt); return 'reflected' },
      })
      await expect(engineSide.call('llm.complete', { prompt: 'propose an edit' }))
        .resolves.toEqual({ text: 'reflected' })
      expect(prompts).toEqual(['propose an edit'])
    })
  })

  describe('publication', () => {
    const commit: CommitRecord = {
      commitId: 'c1', runId: 'run-1', artifact: 'skill:total', blastRadius: 0.2,
      version: 1, heldOutBefore: 0, heldOutAfter: 1,
    }

    it('reaches the adapter, updates the head, and announces the commit', async () => {
      const { sidecar, engineSide } = fakeSidecar()
      const engine = await start(sidecar)
      const announced: CommitRecord[] = []
      evolution.onCommit((record) => announced.push(record))

      await engineSide.call('artifact.publish', {
        artifact: 'skill:total', state: { 'SKILL.md': 'COLUMN: amount\n' },
        version: 1, commit,
      })

      expect(published).toEqual([{ 'SKILL.md': 'COLUMN: amount\n' }])
      expect(engine.head('skill:total')?.version).toBe(1)
      expect(announced).toEqual([commit])
    })

    it('refuses to publish an artifact nothing registered', async () => {
      const { sidecar, engineSide } = fakeSidecar()
      await start(sidecar)
      await expect(engineSide.call('artifact.publish', {
        artifact: 'skill:ghost', state: {}, version: 1,
      })).rejects.toThrow(/unknown artifact/)
      expect(published).toEqual([])
    })
  })

  it('projects progress notifications into a status callers can poll', async () => {
    const { sidecar, engineSide } = fakeSidecar()
    const engine = await start(sidecar)
    expect(engine.status('run-1')).toBeUndefined()
    engineSide.notify('log.progress', {
      runId: 'run-1', artifact: 'skill:total', round: 2,
      accepted: 1, rejected: 3, tokensSpent: 500,
    })
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(engine.status('run-1')).toMatchObject({
      runId: 'run-1', round: 2, accepted: 1, rejected: 3, tokensSpent: 500,
    })
  })
})

/**
 * The real thing: this plugin's bridge against the real Python engine.
 *
 * Everything above proves this side routes correctly. Only this proves the two
 * halves speak the same protocol -- the one claim a fake on either side can
 * never make.
 */
describe('against the real Python sidecar', () => {
  const repoRoot = new URL('../..', import.meta.url).pathname

  const available = new Promise<boolean>((resolve) => {
    const probe = spawn('python3', ['-c', 'import agentdescent.dsh.daemon'],
                        { cwd: repoRoot, stdio: 'ignore' })
    probe.on('exit', (code) => resolve(code === 0))
    probe.on('error', () => resolve(false))
  })

  it('handshakes, runs, and publishes an improved artifact', async () => {
    if (!(await available)) return                 // no engine importable here

    const ctx = new Context()
    await ctx.plugin(SkillRegistry)
    await ctx.plugin(EvolutionRegistry)
    const evolution = ctx.evolution
    const published: ArtifactState[] = []
    evolution.registerArtifact(adapter('skill:total', published))
    evolution.registerScorer({
      name: 'gold',
      score: async (_task, run) => (run.output === 'correct' ? 1 : 0),
    })
    evolution.registerTaskSource({
      name: 'manual',
      fetch: async (n) => Array.from({ length: Math.min(n, 4) },
                                     (_, i) => ({ id: `t${i}`, prompt: 'total the column' })),
    })

    const edit = `<EDITS>${JSON.stringify({
      rationale: 'id is a row number',
      edits: [{ path: 'SKILL.md', content: '---\ndescription: d\n---\n\nCOLUMN: amount\n' }],
    })}</EDITS>`

    const engine = await startEngine(ctx, {
      evolution,
      rollout: async (_artifact, state) => ({
        output: state['SKILL.md']?.includes('COLUMN: amount') === true ? 'correct' : 'wrong',
        finishReason: 'completed', steps: 1, tokens: 0,
      }),
      complete: async () => edit,
      python: 'python3',
      cwd: repoRoot,
    })

    expect(engine.name).toMatch(/^agentdescent@/)
    const runId = await engine.start({
      artifact: 'skill:total', taskSource: 'manual', scorer: 'gold',
      rounds: 3, workers: 1,
    })

    const deadline = Date.now() + 90_000
    while (Date.now() < deadline && published.length === 0) {
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    expect(published.length, `run ${runId} published nothing`).toBe(1)
    expect(published[0]?.['SKILL.md']).toContain('COLUMN: amount')
    expect(engine.head('skill:total')?.version).toBe(1)
  }, 120_000)
})
