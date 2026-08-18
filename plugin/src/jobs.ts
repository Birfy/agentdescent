/**
 * An evolution run as a background job.
 *
 * Registering with `ctx.jobs` costs one adapter and buys the whole background
 * vocabulary the harness already has: the model can list a run with `job_list`,
 * read where it got to, and stop it with `job_kill`, using the same tools it
 * uses for a long bash command. Without this a run is visible only through
 * `/evolve status`, which a model cannot reach and a UI has to special-case.
 *
 * @module dsh-plugin-agentdescent/jobs
 */

import type { Context } from '@deepseek-ai/cordis'

import type { EvolutionRegistry, EvolutionSpec, RunId, RunStatus } from './evolution.js'

/** The phases a run does not come back from. */
const TERMINAL = new Set(['done', 'failed', 'cancelled', 'budget-exhausted',
                          'awaiting-review'])

/** How a terminal phase maps onto the registry's three outcomes. */
function outcomeOf(status: RunStatus): { status: 'completed' | 'killed' | 'failed'; detail: string } {
  switch (status.phase) {
    case 'cancelled':
      return { status: 'killed', detail: 'cancelled' }
    case 'failed':
      return { status: 'failed', detail: status.error ?? 'failed' }
    case 'budget-exhausted':
      // Not a failure: the budget is a ceiling, and reaching it is the ceiling
      // working. Reporting it as failed would teach a model to raise it.
      return { status: 'completed', detail: 'token budget spent' }
    case 'awaiting-review':
      return { status: 'completed', detail: 'waiting for review — /evolve pending' }
    default:
      return { status: 'completed', detail: `${status.accepted} accepted` }
  }
}

/** One line of progress, for a reader that wants to know where a run got to. */
function render(status: RunStatus): string {
  return `round ${status.round}: ${status.accepted} accepted, ` +
    `${status.rejected} rejected, ${status.tokensSpent} tokens (${status.phase})\n`
}

export interface JobsBridgeOptions {
  readonly evolution: EvolutionRegistry
  /** Narrowed `ctx.jobs`; supplied so tests can drive it. */
  readonly jobs?: JobHost
}

/** What this module needs of `ctx.jobs`. */
export interface JobHost {
  start(spec: {
    kind: string
    label: string
    run: () => {
      cancel: (reason?: string) => void
      done: Promise<{ status: 'completed' | 'killed' | 'failed'; detail?: string; output?: string }>
      readOutput?: () => string
    }
  }): string
}

/**
 * Start a run and register it as a job.
 *
 * The producer's `cancel` must be synchronous, and the seam's is not, so the
 * cancellation is issued and not awaited -- which matches what cancellation
 * means here anyway: the run stops at its next rollout boundary, so "requested"
 * is the only honest synchronous answer.
 */
export async function startAsJob(ctx: Context, spec: EvolutionSpec,
                                 options: JobsBridgeOptions): Promise<{ runId: RunId; jobId: string }> {
  const { evolution } = options
  const host = options.jobs ?? (ctx as unknown as { jobs: JobHost }).jobs
  const runId = await evolution.start(spec)

  let settle: (outcome: { status: 'completed' | 'killed' | 'failed'; detail?: string }) => void
  const done = new Promise<{ status: 'completed' | 'killed' | 'failed'; detail?: string }>(
    (resolve) => { settle = resolve })
  const pending: string[] = []

  const jobId = host.start({
    kind: 'evolution',
    label: `evolve ${spec.artifact} (${spec.scorer})`,
    run: () => {
      // Phase edges, not polling: a run that finished between two polls would
      // otherwise leave the job running until someone happened to look.
      const off = evolution.onPhase?.((status) => {
        if (status.runId !== runId) return
        pending.push(render(status))
        if (TERMINAL.has(status.phase)) {
          off?.()
          settle(outcomeOf(status))
        }
      })
      if (off === undefined) {
        // An engine with no phase edges cannot tell us when it ended, and a job
        // that never settles is worse than one that never started.
        settle({ status: 'failed', detail: 'this engine reports no phase changes' })
      }
      return {
        cancel: (reason?: string) => { void evolution.cancel(runId, reason).catch(() => {}) },
        done,
        readOutput: () => pending.splice(0).join(''),
      }
    },
  })
  return { runId, jobId }
}
