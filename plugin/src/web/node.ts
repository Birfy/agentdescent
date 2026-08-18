/**
 * The Chat row an evolution run gets in the Web Client.
 *
 * A run is invisible in the transcript until something renders it: the durable
 * `evolution/*` events are in the log, but the Chat view shows what a
 * Conversation Node Definition assembles. This is that assembly -- one row per
 * run, correlating its commits and its ending into a single piece of state.
 *
 * It is deliberately all pure functions over events, which is the client
 * engine's own contract and also why the interesting half is testable without a
 * browser: `match` / `start` / `apply` are a state machine, and only the small
 * renderer beside it needs a DOM.
 *
 * @module dsh-plugin-agentdescent/web/node
 */

import type { EvolutionCommitEvent, EvolutionRunEvent } from '../history.js'

/** What one row knows about a run. */
export interface EvolutionNodeState {
  readonly runId: string
  readonly artifact: string
  /** Commits in log order. A run may accept several before it ends. */
  readonly commits: readonly EvolutionCommitEvent[]
  /** Set once the run ends; `undefined` while it is still going. */
  readonly ended: EvolutionRunEvent | undefined
}

/** The shape the row renders from -- everything a reader needs, precomputed. */
export interface EvolutionNodeView {
  readonly runId: string
  readonly artifact: string
  readonly phase: string
  readonly commitCount: number
  /** Held-out reward before the first commit and after the last, when any. */
  readonly heldOutBefore: number | undefined
  readonly heldOutAfter: number | undefined
  readonly tokensSpent: number | undefined
  readonly error: string | undefined
}

export const COMMIT_EVENT = 'evolution/commit'
export const ENDED_EVENT = 'evolution/run-ended'

/**
 * Correlate one event into a run's row, or decline it.
 *
 * The business id is the **run**, not the event, so a commit and the ending that
 * follows land on the same row. Either kind may open it: the client engine has
 * to assemble a window loaded from the middle, and a user scrolled back far
 * enough sees an ending whose commits are outside the loaded page.
 */
export function matchEvolutionEvent(event: { type?: string; data?: unknown }):
    { id: string; role: 'start' | 'update' } | null {
  if (event?.type !== COMMIT_EVENT && event?.type !== ENDED_EVENT) return null
  const runId = runIdOf(event)
  if (runId === undefined || runId.length === 0) return null
  return { id: runId, role: 'start' }
}

/** The row a start match creates, already carrying its first event. */
export function startEvolutionNode(event: { type?: string; data?: unknown }): EvolutionNodeState {
  return applyEvolutionEvent(
    { runId: runIdOf(event) ?? '', artifact: '', commits: [], ended: undefined }, event)
}

/**
 * Fold one event into a row.
 *
 * Deterministic under replay in ascending `seq`, which the engine requires: a
 * commit appends, an ending replaces, and neither reads anything outside the
 * event it was handed. Re-applying a commit is idempotent by `commitId`,
 * because overlapping pages replay events the row has already seen.
 */
export function applyEvolutionEvent(state: EvolutionNodeState,
                                    event: { type?: string; data?: unknown }): EvolutionNodeState {
  if (event?.type === COMMIT_EVENT) {
    const commit = event.data as EvolutionCommitEvent
    if (state.commits.some((seen) => seen.commitId === commit.commitId)) return state
    return {
      ...state,
      runId: state.runId || commit.runId,
      artifact: state.artifact || commit.artifact,
      commits: [...state.commits, commit],
    }
  }
  if (event?.type === ENDED_EVENT) {
    const ended = event.data as EvolutionRunEvent
    return {
      ...state,
      runId: state.runId || ended.runId,
      artifact: state.artifact || ended.artifact,
      ended,
    }
  }
  return state
}

/**
 * Everything the row displays, derived once.
 *
 * `heldOutBefore` comes from the *first* commit and `heldOutAfter` from the
 * last, so a run that accepted three changes reports the distance it actually
 * travelled rather than only its last hop.
 */
export function viewOfEvolutionNode(state: EvolutionNodeState): EvolutionNodeView {
  const first = state.commits[0]
  const last = state.commits[state.commits.length - 1]
  return {
    runId: state.runId,
    artifact: state.artifact,
    // No ending recorded means it is still going, as far as a reader can tell.
    phase: state.ended?.phase ?? 'running',
    commitCount: state.commits.length,
    heldOutBefore: first?.heldOutBefore,
    heldOutAfter: last?.heldOutAfter,
    tokensSpent: state.ended?.tokensSpent,
    error: state.ended?.error,
  }
}

/** One line of prose for the row, and the only place the wording lives. */
export function summariseEvolutionNode(view: EvolutionNodeView): string {
  if (view.phase === 'failed') {
    return `Evolving ${view.artifact} failed: ${view.error ?? 'no reason recorded'}`
  }
  if (view.phase === 'awaiting-review') {
    return `Evolving ${view.artifact} produced a change waiting for review — /evolve pending`
  }
  if (view.commitCount === 0) {
    const ending = view.phase === 'running' ? 'is running' : `ended (${view.phase})`
    return `Evolving ${view.artifact} ${ending}; nothing accepted yet`
  }
  const delta = view.heldOutBefore !== undefined && view.heldOutAfter !== undefined
    ? ` — held-out ${view.heldOutBefore.toFixed(3)} → ${view.heldOutAfter.toFixed(3)}`
    : ''
  const changes = view.commitCount === 1 ? '1 change' : `${view.commitCount} changes`
  return `Evolved ${view.artifact}: ${changes} accepted${delta}`
}

function runIdOf(event: { data?: unknown }): string | undefined {
  const runId = (event?.data as { runId?: unknown } | undefined)?.runId
  return typeof runId === 'string' ? runId : undefined
}
