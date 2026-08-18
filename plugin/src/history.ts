/**
 * Writing what evolution did into the session log.
 *
 * Until now a run's record lived in two places, and neither survives: the
 * Python ledger, which the harness cannot read without asking, and in-memory
 * listeners, which a reload empties. So a user who restarted lost every trace
 * that their harness had changed under them.
 *
 * dsh's own answer to that is the one this uses: durable session state means a
 * `SessionEventMap` entry, appended to a session's append-only log. That makes
 * a commit part of the transcript -- it replays, it exports, and any UI that
 * ever renders evolution renders it from here rather than from a live feed it
 * had to be present for.
 *
 * The events are **log-only**: they are not model-visible, so they add no
 * tokens and cannot move the request prefix. What changed is a fact about the
 * harness, not something the model needs told mid-conversation.
 *
 * @module dsh-plugin-agentdescent/history
 */

import type { Context } from '@deepseek-ai/cordis'

import type { ArtifactId, CommitRecord, EvolutionRegistry, RunId, RunStatus } from './evolution.js'

/** What a durable commit record carries. */
export interface EvolutionCommitEvent {
  readonly commitId: string
  readonly runId: RunId
  readonly artifact: ArtifactId
  readonly blastRadius: number
  readonly version: number
  readonly heldOutBefore: number
  readonly heldOutAfter: number
}

/** What a durable run-ended record carries. */
export interface EvolutionRunEvent {
  readonly runId: RunId
  readonly artifact: ArtifactId
  readonly phase: string
  readonly accepted: number
  readonly rejected: number
  readonly tokensSpent: number
  readonly error?: string
}

declare module '@deepseek-ai/dsh-session' {
  interface SessionEventMap {
    /** An artifact of this harness changed. Log-only. */
    'evolution/commit': EvolutionCommitEvent
    /** A run reached a terminal phase. Log-only. */
    'evolution/run-ended': EvolutionRunEvent
  }
}

/** The slice of a session this module writes to. */
export interface HistorySession {
  append(type: 'evolution/commit' | 'evolution/run-ended', data: unknown): unknown
}

export interface HistoryOptions {
  readonly evolution: EvolutionRegistry
  /** The session that gets the record -- normally whoever ran `/evolve`. */
  readonly session: HistorySession
}

/**
 * Record commits and run endings on a session.
 *
 * The session is the caller's rather than a log of its own, because the record
 * belongs to the person whose harness changed: it shows up in their transcript,
 * beside the work that prompted it, and exports with it.
 *
 * A failing append is contained. The commit has already happened, and losing
 * the *record* of a change is bad while losing the change would be worse --
 * so this never propagates into the commit path that called it.
 */
export function recordHistory(ctx: Context, options: HistoryOptions): () => void {
  const { evolution, session } = options
  const disposers: Array<() => void> = []

  disposers.push(evolution.onCommit((record: CommitRecord) => {
    try {
      session.append('evolution/commit', {
        commitId: record.commitId,
        runId: record.runId,
        artifact: record.artifact,
        blastRadius: record.blastRadius,
        version: record.version,
        heldOutBefore: record.heldOutBefore,
        heldOutAfter: record.heldOutAfter,
      })
    } catch (error) {
      ctx.logger.warn(`could not record commit ${record.commitId}: ${String(error)}`)
    }
  }))

  const off = evolution.onPhase?.((status: RunStatus) => {
    if (!TERMINAL.has(status.phase)) return
    try {
      session.append('evolution/run-ended', {
        runId: status.runId,
        artifact: status.artifact,
        phase: status.phase,
        accepted: status.accepted,
        rejected: status.rejected,
        tokensSpent: status.tokensSpent,
        ...(status.error === undefined ? {} : { error: status.error }),
      })
    } catch (error) {
      ctx.logger.warn(`could not record the end of ${status.runId}: ${String(error)}`)
    }
  })
  if (off !== undefined) disposers.push(off)

  return () => { for (const dispose of disposers.reverse()) dispose() }
}

const TERMINAL = new Set(['done', 'failed', 'cancelled', 'budget-exhausted',
                          'awaiting-review'])

/** Read the evolution records out of a log, for a reader or a renderer. */
export function evolutionHistory(events: readonly { type: string; data: unknown }[]):
    Array<{ type: string; data: unknown }> {
  return events.filter((event) => event.type === 'evolution/commit'
                               || event.type === 'evolution/run-ended')
}
