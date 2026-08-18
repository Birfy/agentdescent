/**
 * Where an L1 commit waits for a human.
 *
 * The behaviour that matters is what does *not* happen: a staged commit is not
 * published, does not become a head, and is not announced as a commit. Getting
 * that wrong is invisible from inside a run -- the engine would report a version
 * the harness is not serving, and the next run would start from a state that
 * never shipped.
 */

import { Context } from '@deepseek-ai/cordis'
import { beforeEach, describe, expect, it } from 'vitest'

import { CommitQueue, renderPending } from '../src/approval.js'
import { EvolutionRegistry, type ArtifactAdapter, type ArtifactState, type CommitRecord } from '../src/evolution.js'

function adapter(id: string, blastRadius: number, published: ArtifactState[]): ArtifactAdapter {
  return {
    id, blastRadius,
    load: async () => ({}),
    bind: () => () => {},
    publish: async (state) => { published.push(state); return () => {} },
  }
}

const COMMIT: CommitRecord = {
  commitId: 'c1', runId: 'run-1', artifact: 'plugin:mine', blastRadius: 0.6,
  version: 2, heldOutBefore: 0.5, heldOutAfter: 0.8,
}

describe('the auto-merge line', () => {
  it('lets a skill through and holds a harness change', () => {
    const queue = new CommitQueue()
    expect(queue.mergesAutomatically(0.2)).toBe(true)
    expect(queue.mergesAutomatically(0.6)).toBe(false)
  })

  it('is configurable, because where the line sits is a deployment choice', () => {
    const strict = new CommitQueue({ autoMergeMaxBlastRadius: 0 })
    expect(strict.mergesAutomatically(0.2)).toBe(false)
  })
})

describe('submitting a commit', () => {
  let published: ArtifactState[]
  let queue: CommitQueue

  beforeEach(() => {
    published = []
    queue = new CommitQueue()
  })

  it('publishes a low-blast-radius change immediately', async () => {
    const outcome = await queue.submit(adapter('skill:sql', 0.2, published),
                                       { 'SKILL.md': 'evolved' }, 3, undefined)
    expect(outcome).toEqual({ status: 'published', version: 3 })
    expect(published).toHaveLength(1)
    expect(queue.list()).toEqual([])
  })

  it('stages a harness change without publishing anything', async () => {
    const outcome = await queue.submit(adapter('plugin:mine', 0.6, published),
                                       { 'src/index.ts': 'evolved' }, 2, COMMIT)
    expect(outcome.status).toBe('pending')
    expect(published).toEqual([])              // nothing reached disk
    expect(queue.list()).toHaveLength(1)
  })

  it('explains why it was held, in terms a reviewer can act on', async () => {
    const outcome = await queue.submit(adapter('plugin:mine', 0.6, published), {}, 1, COMMIT)
    expect(outcome.status === 'pending' && outcome.reason).toMatch(/blast radius 0.6/)
    expect(outcome.status === 'pending' && outcome.reason).toMatch(/human approves/)
  })

  it('records what the change would write, so review does not need the diff first', async () => {
    await queue.submit(adapter('plugin:mine', 0.6, published),
                       { 'src/b.ts': 'x', 'src/a.ts': 'y' }, 1, COMMIT)
    expect(queue.list()[0]?.paths).toEqual(['src/a.ts', 'src/b.ts'])
  })
})

describe('deciding', () => {
  let published: ArtifactState[]
  let queue: CommitQueue

  beforeEach(async () => {
    published = []
    queue = new CommitQueue()
    await queue.submit(adapter('plugin:mine', 0.6, published),
                       { 'src/index.ts': 'evolved' }, 2, COMMIT)
  })

  it('publishes on approval and clears the queue', async () => {
    const held = await queue.approve('pending-1', async (entry) => {
      published.push(entry.state)
    })
    expect(held?.artifact).toBe('plugin:mine')
    expect(published).toEqual([{ 'src/index.ts': 'evolved' }])
    expect(queue.list()).toEqual([])
  })

  it('keeps the commit reviewable when publishing fails', async () => {
    // Dropping it here would silently discard a change someone approved.
    await expect(queue.approve('pending-1', async () => {
      throw new Error('disk full')
    })).rejects.toThrow(/disk full/)
    expect(queue.list()).toHaveLength(1)
  })

  it('drops it on rejection, and the ledger still has it', () => {
    expect(queue.reject('pending-1')?.artifact).toBe('plugin:mine')
    expect(queue.list()).toEqual([])
  })

  it('says so for an id it does not hold', async () => {
    expect(queue.reject('pending-99')).toBeUndefined()
    expect(await queue.approve('pending-99', async () => {})).toBeUndefined()
  })
})

describe('what a reviewer sees', () => {
  it('says plainly when nothing is waiting', () => {
    expect(renderPending([])).toContain('nothing waiting')
  })

  it('shows the artifact, the layer, and what the change bought', async () => {
    const queue = new CommitQueue()
    await queue.submit(adapter('plugin:mine', 0.6, []), { 'src/index.ts': 'x' }, 2, COMMIT)
    const rendered = renderPending(queue.list())
    expect(rendered).toContain('plugin:mine')
    expect(rendered).toContain('L1')
    expect(rendered).toContain('0.500 -> 0.800')
  })
})

describe('the registry is untouched by a staged commit', () => {
  it('announces nothing, because nothing was committed', async () => {
    const ctx = new Context()
    await ctx.plugin(EvolutionRegistry)
    const announced: CommitRecord[] = []
    ctx.evolution.onCommit((record) => announced.push(record))

    const queue = new CommitQueue()
    await queue.submit(adapter('plugin:mine', 0.6, []), {}, 2, COMMIT)
    expect(announced).toEqual([])
  })
})
