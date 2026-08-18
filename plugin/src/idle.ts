/**
 * Evolving while nobody is looking.
 *
 * A harness that only improves when someone types `/evolve` improves about as
 * often as people remember to type it. The target is a harness that gets better
 * because it is used, so a run starts on its own -- but only when the person is
 * demonstrably not waiting on anything, and it gets out of the way the instant
 * they come back.
 *
 * "Idle" here is whole-harness idle, not agent idle: with several sessions
 * open, one going quiet says nothing about whether the user is busy.
 *
 * @module dsh-plugin-agentdescent/idle
 */

import type { Context } from '@deepseek-ai/cordis'

import type { EvolutionRegistry, EvolutionSpec, RunId } from './evolution.js'

export interface IdleTriggerOptions {
  readonly evolution: EvolutionRegistry
  /** What to run when the harness goes quiet. Nothing starts without one. */
  readonly runs: readonly EvolutionSpec[]
  /** How long everything must stay quiet first. */
  readonly afterMs?: number
  /** Smallest gap between two automatic runs, however idle it stays. */
  readonly minIntervalMs?: number
  /** Injectable for tests; defaults to wall-clock. */
  readonly now?: () => number
}

const DEFAULT_AFTER_MS = 5 * 60_000
const DEFAULT_MIN_INTERVAL_MS = 60 * 60_000

/**
 * Start a run once the harness has been quiet for a while.
 *
 * Three rules, and each exists because of a way this goes wrong:
 *
 * - **Every agent must be idle.** Watching one agent would start a training run
 *   while the user is mid-task in another session, competing with them for the
 *   very model they are waiting on.
 * - **Activity cancels the pending start, and stops a running one.** The user
 *   coming back is the signal that the machine is theirs again. A run that kept
 *   going would spend their tokens and their concurrency while they work.
 * - **A minimum interval.** Without it, a machine left alone overnight starts a
 *   run every time the previous one finishes, which is a plausible way to spend
 *   a lot of money by leaving a laptop open.
 */
export function idleTrigger(ctx: Context, options: IdleTriggerOptions): () => void {
  const afterMs = options.afterMs ?? DEFAULT_AFTER_MS
  const minIntervalMs = options.minIntervalMs ?? DEFAULT_MIN_INTERVAL_MS
  const now = options.now ?? (() => Date.now())
  const busy = new Set<string>()
  let timer: ReturnType<typeof setTimeout> | undefined
  let lastStarted = -Infinity
  let active: RunId | undefined
  let next = 0

  const disarm = (): void => {
    if (timer !== undefined) clearTimeout(timer)
    timer = undefined
  }

  const arm = (): void => {
    disarm()
    if (options.runs.length === 0) return
    timer = setTimeout(() => { void start() }, afterMs)
    timer.unref?.()
  }

  const start = async (): Promise<void> => {
    if (busy.size > 0 || active !== undefined) return
    if (now() - lastStarted < minIntervalMs) return
    const spec = options.runs[next % options.runs.length]
    next += 1
    if (spec === undefined) return
    lastStarted = now()
    try {
      // Asynchronous by default here: a background run should be interruptible
      // at a rollout boundary, not hold a barrier open across the user's return.
      active = await options.evolution.start({ asynchronous: true, ...spec })
      ctx.logger.info(`idle: started ${active} on ${spec.artifact}`)
    } catch (error) {
      active = undefined
      ctx.logger.warn(`idle: could not start a run on ${spec.artifact}: ${String(error)}`)
    }
  }

  const stopForUser = (): void => {
    const running = active
    active = undefined
    if (running === undefined) return
    ctx.logger.info(`idle: stopping ${running}; the harness is in use again`)
    void options.evolution.cancel(running, 'the harness is in use again').catch(() => {})
  }

  const off = ctx.on('agent/status', (payload: any) => {
    const id = String(payload?.agent?.id ?? '')
    if (id.length === 0) return
    // A rollout's own child agents are activity we caused; counting them would
    // make the trigger cancel its own run the moment it started one.
    if (id.startsWith('evolve-')) return
    if (payload.status === 'running') {
      busy.add(id)
      disarm()
      stopForUser()
    } else {
      busy.delete(id)
      if (busy.size === 0) arm()
    }
  })

  arm()
  return () => {
    disarm()
    off()
  }
}
