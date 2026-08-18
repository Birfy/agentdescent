/**
 * Rollouts that run **inside** this harness.
 *
 * The alternative is a stripped-down replica: spawn something that looks like
 * the harness, measure a candidate against it, and ship the winner into the real
 * one. That measures the replica. Here the candidate is registered against one
 * child agent's own context and the child runs with this harness's real tools,
 * sandbox, approval policy and prompt assembly -- so the number the aggregator
 * accepts on is a number about the thing that will actually serve the user.
 *
 * The concurrency story falls out of the same fact. A registration scoped to
 * `agent.ctx` lands in that agent's layer, and a nearer layer wins a name
 * outright, so N workers can hold N different candidates for the same skill in
 * one process at once -- no containers, no second harness.
 *
 * @module dsh-plugin-agentdescent/rollout
 */

import type { Context } from '@deepseek-ai/cordis'

import type {
  ArtifactId, ArtifactState, EvolutionRegistry, EvolutionTask, RolloutResult,
} from './evolution.js'
import type { RolloutRunner } from './engine.js'

/** The lifecycle state an agent reports on every `agent/status`. */
export type AgentStatus = 'idle' | 'running'

/**
 * The slice of a live agent a rollout touches.
 *
 * Narrowed deliberately, and checked against the real interface below: a rollout
 * needs somewhere to scope a candidate, a way to submit one turn, and a way to
 * know when the work it owns is finished. Anything more would be this module
 * reaching into the loop.
 */
export interface RolloutAgent {
  readonly id: string
  readonly ctx: Context
  readonly status: AgentStatus
  followup(message: any): void
}

/** An owned agent plus the disposer that stops and drains it. */
export interface RolloutAgentHandle {
  readonly agent: RolloutAgent
  dispose(): Promise<void>
}

/** What this module needs of `ctx.agents`. */
export interface RolloutHost {
  create(options: any): Promise<RolloutAgentHandle>
}

export interface InProcessRolloutOptions {
  readonly evolution: EvolutionRegistry
  /** Builds the child's user message. Defaults to a plain text turn. */
  readonly message?: (task: EvolutionTask) => unknown
  /** Names each child session. Defaults to `evolve-<artifact>-<task>-<n>`. */
  readonly sessionId?: (artifact: ArtifactId, task: EvolutionTask, seq: number) => string
  readonly cwd?: string
  /** Give-up for one rollout. A hung child must not hold a worker forever. */
  readonly timeoutMs?: number
}

const DEFAULT_TIMEOUT_MS = 900_000

let seq = 0

/**
 * Build a {@link RolloutRunner} that forks a child agent per rollout.
 *
 * Order matters and is the easy thing to get wrong: the candidate is bound
 * **before** the turn is submitted, because prompt assembly reads the registries
 * at the first step. Binding after `followup()` races the loop, and when it
 * loses, the child answers from the old artifact and the rollout silently scores
 * the wrong thing.
 */
export function inProcessRollout(ctx: Context, options: InProcessRolloutOptions): RolloutRunner {
  const { evolution } = options
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS

  return async function rollout(artifact: ArtifactId, state: ArtifactState,
                                task: EvolutionTask): Promise<RolloutResult> {
    const adapter = evolution.artifact(artifact)
    if (adapter === undefined) throw new Error(`unknown artifact "${artifact}"`)

    const id = options.sessionId?.(artifact, task, (seq += 1))
      ?? `evolve-${artifact.replace(/[^A-Za-z0-9_.-]+/g, '-')}-${task.id}-${(seq += 1)}`
    const host = ctx.agents as unknown as RolloutHost
    const handle = await host.create({
      sessionId: id,
      ...(options.cwd === undefined ? {} : { meta: { cwd: options.cwd } }),
    })
    const { agent } = handle

    const collected = new Collector()
    const stopWatching = ctx.on('session/event', (session: any, event: any) => {
      if (session?.id === agent.id) collected.absorb(event)
    })
    // Bind before the turn: see the note above.
    const unbind = adapter.bind(agent.ctx, state)

    try {
      const settled = ownedInterval(ctx, agent, timeoutMs)
      agent.followup(options.message?.(task) ?? textMessage(task.prompt))
      await settled
      return collected.result()
    } finally {
      unbind()
      stopWatching()
      // Disposing reaches quiescence, so the child cannot outlive the rollout
      // that owns it and leak a running agent per measurement.
      await handle.dispose().catch(() => {})
    }
  }
}

/**
 * Resolve once the agent has finished the work this rollout submitted.
 *
 * The interval is "from our submission to the next whole-agent idle", the same
 * ownership the Python SDK's `Session.run()` defines -- so a rollout measured
 * in-process and one measured through the SDK mean the same thing, and their
 * rewards are comparable. It is subscribed *before* the turn is submitted,
 * because an agent that finishes quickly would otherwise reach idle before
 * anyone was listening.
 */
function ownedInterval(ctx: Context, agent: RolloutAgent, timeoutMs: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    let started = false
    const finish = (fn: () => void) => {
      clearTimeout(timer)
      off()
      fn()
    }
    const timer = setTimeout(() => {
      finish(() => reject(new Error(
        `rollout on ${agent.id} did not reach idle within ${timeoutMs}ms`)))
    }, timeoutMs)
    timer.unref?.()
    const off = ctx.on('agent/status', (payload: any) => {
      if (payload?.agent?.id !== agent.id) return
      if (payload.status === 'running') started = true
      else if (payload.status === 'idle' && started) finish(resolve)
    })
  })
}

/** A plain single-text user turn. */
function textMessage(text: string): unknown {
  return { content: [{ type: 'text', text }], source: { kind: 'user' } }
}

/**
 * Projects a rollout's outcome from the session log.
 *
 * Everything a scorer reads comes from here, and it comes from the log rather
 * than from the driver because the log is what dsh guarantees: model-visible
 * means logged. A field derived any other way could disagree with the
 * transcript the same run is judged against.
 */
class Collector {
  private text = ''
  private chunks = ''
  private steps = 0
  private toolErrors = 0
  private tokens = 0
  private finishReason: string | undefined

  absorb(event: any): void {
    switch (event?.type) {
      case 'assistant/message': {
        const content = event.data?.message?.content
        const rendered = renderText(content)
        if (rendered.length > 0) this.text = rendered
        break
      }
      case 'assistant/chunk': {
        const chunk = event.data?.chunk
        if (chunk?.type === 'text-delta' && typeof chunk.text === 'string') {
          this.chunks += chunk.text
        }
        break
      }
      case 'step/start':
        this.steps += 1
        break
      case 'tool/result':
        // A failed tool call is signal, not an error to raise: "the candidate
        // made the agent misuse a tool" is exactly what an efficiency scorer is
        // looking for.
        if (event.data?.result?.isError === true || event.data?.isError === true) {
          this.toolErrors += 1
        }
        break
      case 'turn/end':
        this.finishReason = typeof event.data?.reason?.kind === 'string'
          ? event.data.reason.kind
          : this.finishReason
        this.tokens += Number(event.data?.usage?.totalTokens ?? 0) || 0
        break
      default:
        break
    }
  }

  result(): RolloutResult {
    return {
      // A committed assistant message is authoritative; the accumulated deltas
      // are the fallback for a turn that streamed but never committed one.
      output: (this.text.length > 0 ? this.text : this.chunks).trim(),
      ...(this.finishReason === undefined ? {} : { finishReason: this.finishReason }),
      steps: this.steps,
      toolErrors: this.toolErrors,
      tokens: this.tokens,
    }
  }
}

function renderText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .filter((part: any) => part?.type === 'text' && typeof part.text === 'string')
    .map((part: any) => part.text)
    .join('')
}
