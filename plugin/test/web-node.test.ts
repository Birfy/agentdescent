/**
 * The Chat row's assembly.
 *
 * The client engine's contract is that a Definition is pure over events, which
 * is exactly what makes this testable with no browser in sight. The cases worth
 * pinning are the awkward ones the engine actually produces: a window loaded
 * from the middle, pages that overlap, and a run still going.
 */

import { describe, expect, it } from 'vitest'

import type { EvolutionCommitEvent, EvolutionRunEvent } from '../src/history.js'
import {
  applyEvolutionEvent, COMMIT_EVENT, ENDED_EVENT, matchEvolutionEvent,
  startEvolutionNode, summariseEvolutionNode, viewOfEvolutionNode,
} from '../src/web/node.js'

function commit(over: Partial<EvolutionCommitEvent> = {}) {
  const data: EvolutionCommitEvent = {
    commitId: 'c1', runId: 'run-1', artifact: 'skill:sql', blastRadius: 0.2,
    version: 2, heldOutBefore: 0.60, heldOutAfter: 0.72, ...over,
  }
  return { type: COMMIT_EVENT, data }
}

function ended(over: Partial<EvolutionRunEvent> = {}) {
  const data: EvolutionRunEvent = {
    runId: 'run-1', artifact: 'skill:sql', phase: 'done',
    accepted: 1, rejected: 5, tokensSpent: 4200, ...over,
  }
  return { type: ENDED_EVENT, data }
}

function fold(...events: Array<{ type: string; data: unknown }>) {
  let state = startEvolutionNode(events[0]!)
  for (const event of events.slice(1)) state = applyEvolutionEvent(state, event)
  return state
}

describe('correlating events into a row', () => {
  it('keys on the run, so a commit and its ending share a row', () => {
    expect(matchEvolutionEvent(commit())?.id).toBe('run-1')
    expect(matchEvolutionEvent(ended())?.id).toBe('run-1')
  })

  it('declines anything that is not ours', () => {
    expect(matchEvolutionEvent({ type: 'user/message', data: {} })).toBeNull()
    expect(matchEvolutionEvent({ type: COMMIT_EVENT, data: {} })).toBeNull()
    expect(matchEvolutionEvent({})).toBeNull()
  })

  it('lets either kind open the row', () => {
    // A window loaded from the middle has to assemble: a user scrolled back far
    // enough sees an ending whose commits are outside the loaded page.
    expect(matchEvolutionEvent(commit())?.role).toBe('start')
    expect(matchEvolutionEvent(ended())?.role).toBe('start')
  })
})

describe('folding a row', () => {
  it('collects the commits a run accepted', () => {
    const state = fold(commit({ commitId: 'c1' }), commit({ commitId: 'c2' }), ended())
    expect(state.commits.map((c) => c.commitId)).toEqual(['c1', 'c2'])
    expect(state.ended?.phase).toBe('done')
  })

  it('is idempotent per commit, because overlapping pages replay', () => {
    const state = fold(commit({ commitId: 'c1' }), commit({ commitId: 'c1' }))
    expect(state.commits).toHaveLength(1)
  })

  it('assembles from an ending alone', () => {
    const state = fold(ended())
    expect(state.runId).toBe('run-1')
    expect(state.artifact).toBe('skill:sql')
    expect(state.commits).toEqual([])
  })

  it('ignores an event it does not own', () => {
    const state = fold(commit())
    expect(applyEvolutionEvent(state, { type: 'assistant/message', data: {} })).toBe(state)
  })
})

describe('what the row shows', () => {
  it('reports the distance travelled, not just the last hop', () => {
    // Three commits: before comes from the first, after from the last.
    const state = fold(
      commit({ commitId: 'c1', heldOutBefore: 0.50, heldOutAfter: 0.61 }),
      commit({ commitId: 'c2', heldOutBefore: 0.61, heldOutAfter: 0.70 }),
      commit({ commitId: 'c3', heldOutBefore: 0.70, heldOutAfter: 0.83 }),
      ended({ accepted: 3 }))
    const view = viewOfEvolutionNode(state)
    expect(view.heldOutBefore).toBe(0.50)
    expect(view.heldOutAfter).toBe(0.83)
    expect(view.commitCount).toBe(3)
    expect(summariseEvolutionNode(view)).toContain('3 changes accepted')
    expect(summariseEvolutionNode(view)).toContain('0.500 → 0.830')
  })

  it('says a run with no ending is still running', () => {
    const view = viewOfEvolutionNode(fold(commit()))
    expect(view.phase).toBe('running')
    expect(view.tokensSpent).toBeUndefined()
  })

  it('does not dress up a run that accepted nothing', () => {
    // The honest common case on a small held-out set, and it must not read as
    // a success.
    const view = viewOfEvolutionNode(fold(ended({ accepted: 0 })))
    expect(summariseEvolutionNode(view)).toContain('nothing accepted yet')
  })

  it('carries a failure reason, which is what a reader is looking for', () => {
    const view = viewOfEvolutionNode(fold(ended({ phase: 'failed', error: 'the sidecar died' })))
    expect(summariseEvolutionNode(view)).toContain('the sidecar died')
  })

  it('points a staged change at where it is waiting', () => {
    const view = viewOfEvolutionNode(fold(commit(), ended({ phase: 'awaiting-review' })))
    expect(summariseEvolutionNode(view)).toContain('/evolve pending')
  })

  it('says something sensible when a failure recorded no reason', () => {
    const view = viewOfEvolutionNode(fold(ended({ phase: 'failed' })))
    expect(summariseEvolutionNode(view)).toContain('no reason recorded')
  })

  it('keeps what the run cost', () => {
    expect(viewOfEvolutionNode(fold(ended())).tokensSpent).toBe(4200)
  })
})
