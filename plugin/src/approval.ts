/**
 * Where an L1 commit waits for a human.
 *
 * The obvious implementation is `ctx.approval.request()`, and it does not work
 * here: that seam requires an **open agent turn**, because it appends its
 * audit pair to that agent's session log. An evolution run lands its commits on
 * its own timeline -- minutes after `/evolve` returned, quite possibly while
 * nobody is talking to the harness at all -- so there is no turn to attach the
 * question to, and asking would throw.
 *
 * So a high-blast-radius commit is **staged** rather than asked about. It waits
 * in a queue, `/evolve pending` shows it, and a human approves or rejects it
 * from a command -- which is itself an open turn, and the point at which the
 * person is actually present to decide.
 *
 * The queue is deliberately not durable. A commit that survived a restart would
 * be a change approved against a harness that no longer exists in the same
 * shape, and the ledger still holds the state, so re-running is cheap and
 * re-deciding is correct.
 *
 * @module dsh-plugin-agentdescent/approval
 */

import type { ArtifactAdapter, ArtifactId, ArtifactState, CommitRecord } from './evolution.js'

/** A commit held back because its blast radius puts it above the auto-merge line. */
export interface PendingCommit {
  readonly id: string
  readonly artifact: ArtifactId
  readonly blastRadius: number
  readonly state: ArtifactState
  readonly commit: CommitRecord | undefined
  /** Files this commit would write, for a reviewer who wants the shape first. */
  readonly paths: readonly string[]
}

/** What `artifact.publish` reports back to the engine. */
export type PublishOutcome =
  | { readonly status: 'published'; readonly version: number }
  | { readonly status: 'pending'; readonly pendingId: string; readonly reason: string }

export interface CommitQueueOptions {
  /**
   * The highest blast radius that merges without a human.
   *
   * 0.2 is a skill: local, reversible, judged on held-out score alone. 0.6 is a
   * harness -- a preset, a tool set, a plugin's source -- where being wrong
   * costs more than one wrong answer.
   */
  readonly autoMergeMaxBlastRadius?: number
}

const DEFAULT_AUTO_MERGE_MAX = 0.2

/**
 * Holds L1 commits until a person decides.
 *
 * Staging is *not* a rejection and not a failure: the run succeeded and produced
 * something the gate accepted. It is the difference between "this is ready" and
 * "this is live", which for anything that can change how the harness behaves is
 * a difference a person owns.
 */
export class CommitQueue {
  private readonly pending = new Map<string, PendingCommit>()
  private readonly autoMergeMax: number
  private seq = 0

  constructor(options: CommitQueueOptions = {}) {
    this.autoMergeMax = options.autoMergeMaxBlastRadius ?? DEFAULT_AUTO_MERGE_MAX
  }

  /** Whether this artifact merges without asking. */
  mergesAutomatically(blastRadius: number): boolean {
    return blastRadius <= this.autoMergeMax
  }

  /**
   * Publish now, or stage for review.
   *
   * The outcome goes back to the engine so it can tell the two apart. An engine
   * that recorded a head for a staged commit would report a version the harness
   * is not serving -- and the next run would start from a state that never
   * shipped.
   */
  async submit(adapter: ArtifactAdapter, state: ArtifactState, version: number,
               commit: CommitRecord | undefined): Promise<PublishOutcome> {
    if (this.mergesAutomatically(adapter.blastRadius)) {
      await adapter.publish(state)
      return { status: 'published', version }
    }
    const id = `pending-${(this.seq += 1)}`
    this.pending.set(id, {
      id,
      artifact: adapter.id,
      blastRadius: adapter.blastRadius,
      state,
      commit,
      paths: Object.keys(state).sort(),
    })
    return {
      status: 'pending',
      pendingId: id,
      reason: `blast radius ${adapter.blastRadius} is above the ${this.autoMergeMax} ` +
              'auto-merge line; a human approves this one',
    }
  }

  list(): PendingCommit[] {
    return [...this.pending.values()]
  }

  get(id: string): PendingCommit | undefined {
    return this.pending.get(id)
  }

  /** Approve and publish. Returns what was published, or undefined if unknown. */
  async approve(id: string, publish: (held: PendingCommit) => Promise<unknown>):
      Promise<PendingCommit | undefined> {
    const held = this.pending.get(id)
    if (held === undefined) return undefined
    // Removed only after the publish succeeds: a failed write must leave the
    // commit reviewable rather than silently dropping a change someone approved.
    await publish(held)
    this.pending.delete(id)
    return held
  }

  reject(id: string): PendingCommit | undefined {
    const held = this.pending.get(id)
    if (held !== undefined) this.pending.delete(id)
    return held
  }
}

/** One line per pending commit, for `/evolve pending`. */
export function renderPending(held: readonly PendingCommit[]): string {
  if (held.length === 0) return 'nothing waiting for review.'
  return held.map((entry) => {
    const delta = entry.commit === undefined
      ? ''
      : ` held-out ${entry.commit.heldOutBefore.toFixed(3)} -> ` +
        `${entry.commit.heldOutAfter.toFixed(3)}`
    return `${entry.id}  ${entry.artifact}  (L1, blast ${entry.blastRadius})${delta}\n` +
           `    ${entry.paths.length} file(s): ${entry.paths.slice(0, 6).join(', ')}` +
           (entry.paths.length > 6 ? ', …' : '')
  }).join('\n')
}
