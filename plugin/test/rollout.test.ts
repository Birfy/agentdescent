/**
 * Rollouts that run inside the harness.
 *
 * The agent loop itself is not mounted here -- it needs a model adapter and a
 * whole profile -- so the child is a fake that emits the same events a real one
 * does. What is under test is the part this module owns: the order of binding
 * and submitting, the interval it claims, what it projects from the log, and
 * that nothing is left registered or running afterwards.
 *
 * The narrowed view of `ctx.agents` is checked against the real interface at
 * the bottom, so this staying green while dsh's API moves is not possible.
 */

import { Context } from '@deepseek-ai/cordis'
import type { AgentRegistry } from '@deepseek-ai/dsh-agent'
import { beforeEach, describe, expect, it } from 'vitest'

import { EvolutionRegistry, type ArtifactAdapter, type ArtifactState } from '../src/evolution.js'
import { inProcessRollout, type RolloutAgentHandle, type RolloutHost } from '../src/rollout.js'

const TASK = { id: 't0', prompt: 'total the column' }

/** A child that emits what a real one emits, on command. */
class FakeAgent {
  readonly status: 'idle' | 'running' = 'idle'
  disposed = false
  submitted: unknown[] = []

  constructor(readonly id: string, readonly ctx: Context,
              private readonly script: (agent: FakeAgent) => void | Promise<void>) {}

  followup(message: unknown): void {
    this.submitted.push(message)
    void Promise.resolve().then(async () => await this.script(this))
  }

  emitStatus(status: 'idle' | 'running'): void {
    // Cast at the emit: these are dsh's own event names, and a test standing in
    // for the loop is not the place to reconstruct their exact payload types.
    ;(this.ctx as any).emit('agent/status', { agent: this, status })
  }

  emitEvent(event: unknown): void {
    ;(this.ctx as any).emit('session/event', { id: this.id }, event)
  }

  /** running -> events -> idle, the shape a real turn has. */
  async turn(events: unknown[]): Promise<void> {
    this.emitStatus('running')
    for (const event of events) this.emitEvent(event)
    this.emitStatus('idle')
  }
}

describe('in-process rollouts', () => {
  let ctx: Context
  let evolution: EvolutionRegistry
  let bindings: Array<{ state: ArtifactState; live: boolean }>
  let created: FakeAgent[]

  function host(script: (agent: FakeAgent) => void | Promise<void>): RolloutHost {
    return {
      create: async (options: any): Promise<RolloutAgentHandle> => {
        const agent = new FakeAgent(String(options.sessionId), ctx.isolate('skills'), script)
        created.push(agent)
        return {
          agent: agent as never,
          dispose: async () => { agent.disposed = true },
        }
      },
    }
  }

  function useHost(h: RolloutHost): void {
    Object.defineProperty(ctx, 'agents', { value: h, configurable: true })
  }

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(EvolutionRegistry)
    evolution = ctx.evolution
    bindings = []
    created = []
    const adapter: ArtifactAdapter = {
      id: 'skill:total',
      blastRadius: 0.2,
      load: async () => ({ 'SKILL.md': 'start' }),
      bind: (_scope, state) => {
        const record = { state, live: true }
        bindings.push(record)
        return () => { record.live = false }
      },
      publish: async () => () => {},
    }
    evolution.registerArtifact(adapter)
  })

  it('binds the candidate before submitting the turn', async () => {
    // Prompt assembly reads the registries at the first step. Binding after the
    // turn races the loop, and when it loses the child answers from the old
    // artifact -- a rollout that silently scored the wrong thing.
    const order: string[] = []
    useHost(host(async (agent) => {
      order.push('turn')
      await agent.turn([assistantMessage('done')])
    }))
    const originalBind = evolution.artifact('skill:total')!.bind
    Object.defineProperty(evolution.artifact('skill:total')!, 'bind', {
      value: (scope: Context, state: ArtifactState) => {
        order.push('bind')
        return originalBind(scope, state)
      },
    })

    const rollout = inProcessRollout(ctx, { evolution })
    await rollout('skill:total', { 'SKILL.md': 'candidate' }, TASK)
    expect(order).toEqual(['bind', 'turn'])
  })

  it('hands each rollout its own candidate state', async () => {
    useHost(host(async (agent) => { await agent.turn([assistantMessage('ok')]) }))
    const rollout = inProcessRollout(ctx, { evolution })
    await Promise.all([
      rollout('skill:total', { 'SKILL.md': 'A' }, { id: 'a', prompt: 'q' }),
      rollout('skill:total', { 'SKILL.md': 'B' }, { id: 'b', prompt: 'q' }),
    ])
    expect(bindings.map((b) => b.state['SKILL.md']).sort()).toEqual(['A', 'B'])
    expect(created).toHaveLength(2)
    expect(created[0]?.id).not.toBe(created[1]?.id)
  })

  it('owns the interval from its submission to the next idle', async () => {
    useHost(host(async (agent) => {
      await agent.turn([assistantMessage('first')])
    }))
    const rollout = inProcessRollout(ctx, { evolution })
    const result = await rollout('skill:total', {}, TASK)
    expect(result.output).toBe('first')
  })

  it('ignores another agent reaching idle', async () => {
    // Two rollouts run concurrently; a status event from the wrong child must
    // not end this one's interval early and score a half-finished turn.
    useHost(host(async (agent) => {
      ;(ctx as any).emit('agent/status', { agent: { id: 'someone-else' }, status: 'idle' })
      await agent.turn([assistantMessage('mine')])
    }))
    const rollout = inProcessRollout(ctx, { evolution })
    expect((await rollout('skill:total', {}, TASK)).output).toBe('mine')
  })

  it('unbinds and disposes the child when the turn is done', async () => {
    useHost(host(async (agent) => { await agent.turn([assistantMessage('done')]) }))
    await inProcessRollout(ctx, { evolution })('skill:total', {}, TASK)
    expect(bindings.every((b) => !b.live)).toBe(true)
    expect(created[0]?.disposed).toBe(true)
  })

  it('unbinds and disposes even when the rollout times out', async () => {
    // A hung child must not hold a worker forever, and must not leave its
    // candidate registered where the next rollout would see it.
    useHost(host(() => { /* never reaches idle */ }))
    const rollout = inProcessRollout(ctx, { evolution, timeoutMs: 50 })
    await expect(rollout('skill:total', {}, TASK)).rejects.toThrow(/did not reach idle/)
    expect(bindings.every((b) => !b.live)).toBe(true)
    expect(created[0]?.disposed).toBe(true)
  })

  it('refuses an artifact nothing registered', async () => {
    useHost(host(async (agent) => { await agent.turn([]) }))
    await expect(inProcessRollout(ctx, { evolution })('skill:ghost', {}, TASK))
      .rejects.toThrow(/unknown artifact/)
  })

  describe('what it projects from the log', () => {
    async function run(events: unknown[]) {
      useHost(host(async (agent) => { await agent.turn(events) }))
      return await inProcessRollout(ctx, { evolution })('skill:total', {}, TASK)
    }

    it('prefers a committed assistant message', async () => {
      expect((await run([
        { type: 'assistant/chunk', data: { chunk: { type: 'text-delta', text: 'stream' } } },
        assistantMessage('committed'),
      ])).output).toBe('committed')
    })

    it('falls back to the streamed deltas when nothing committed', async () => {
      expect((await run([
        { type: 'assistant/chunk', data: { chunk: { type: 'text-delta', text: 'par' } } },
        { type: 'assistant/chunk', data: { chunk: { type: 'text-delta', text: 'tial' } } },
      ])).output).toBe('partial')
    })

    it('counts steps and tool errors, which is what an efficiency scorer reads', async () => {
      const result = await run([
        { type: 'step/start', data: {} },
        { type: 'tool/result', data: { result: { isError: true } } },
        { type: 'step/start', data: {} },
        { type: 'tool/result', data: { result: { isError: false } } },
        assistantMessage('done'),
      ])
      expect(result.steps).toBe(2)
      expect(result.toolErrors).toBe(1)
    })

    it('carries how the turn ended, so a broken turn is not scored as an answer', async () => {
      const result = await run([
        assistantMessage('half an answer'),
        { type: 'turn/end', data: { reason: { kind: 'max-tokens' }, usage: { totalTokens: 40 } } },
      ])
      expect(result.finishReason).toBe('max-tokens')
      expect(result.tokens).toBe(40)
    })
  })
})

function assistantMessage(text: string): unknown {
  return { type: 'assistant/message', data: { message: { content: [{ type: 'text', text }] } } }
}

/**
 * The real registry must still satisfy the slice `rollout.ts` narrowed itself
 * to. If dsh changes `create` or `AgentHandle`, this stops compiling -- which is
 * the point: a narrowed view that drifts is a runtime failure wearing a green
 * test suite.
 */
export const _theRealRegistryStillFits: (registry: AgentRegistry) => RolloutHost =
  (registry) => registry as unknown as RolloutHost
