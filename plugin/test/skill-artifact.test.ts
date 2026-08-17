/**
 * The `skill:<name>` adapter, against the real `ctx.skills` registry.
 *
 * Testing this against a stub would prove nothing: every claim worth making
 * here -- that a candidate is visible to one agent only, that a published skill
 * beats the one on disk, that republishing does not collide -- is a claim about
 * the registry's layering and rank rules, not about our code in isolation.
 */

import { Context } from '@deepseek-ai/cordis'
import SkillRegistry from '@deepseek-ai/dsh-skill'
import type { SkillCandidate, SkillProvider } from '@deepseek-ai/dsh-skill'
import { beforeEach, describe, expect, it } from 'vitest'

import { ENTRYPOINT, EVOLVED_SKILL_RANK, parseSkill, skillArtifact } from '../src/artifacts/skill.js'

const BODY = `---
name: sql
description: Write SQL for our warehouse
---

Use snake_case table names.
`

/** A stand-in for the filesystem provider, at its nearest root's rank. */
function onDisk(name: string, body: string, rank = 100): SkillProvider {
  const candidate: SkillCandidate = {
    name,
    description: 'from disk',
    invocation: { modelInvocable: true, userInvocable: true },
    source: 'project-dsh',
    provider: 'filesystem',
    rank,
    locator: name,
  }
  return {
    name: 'filesystem',
    list: async () => [candidate],
    get: async () => ({ ...candidate, content: body }),
  }
}

describe('parseSkill', () => {
  it('reads the routing fields and strips the frontmatter off the body', () => {
    const parsed = parseSkill(BODY)
    expect(parsed.description).toBe('Write SQL for our warehouse')
    expect(parsed.content.trim()).toBe('Use snake_case table names.')
  })

  it('falls back to the first line when there is no frontmatter', () => {
    expect(parseSkill('# Totalling a column\n\nbody').description).toBe('Totalling a column')
  })

  it('declines a block scalar rather than describing the skill as ">"', () => {
    const parsed = parseSkill('---\ndescription: >\n  folded\n---\n\nbody\n')
    expect(parsed.description).not.toBe('>')
    expect(parsed.description).toBe('An evolved skill.')
  })

  it('survives a body with no content at all', () => {
    expect(parseSkill('').description).toBe('An evolved skill.')
  })
})

describe('the skill artifact', () => {
  let ctx: Context

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(SkillRegistry)
  })

  function artifact(state: Record<string, string> = { [ENTRYPOINT]: BODY }) {
    return skillArtifact(ctx, { name: 'sql', load: async () => state })
  }

  it('is addressed as skill:<name> and merges on score, without an oracle', () => {
    const adapter = artifact()
    expect(adapter.id).toBe('skill:sql')
    expect(adapter.blastRadius).toBeLessThan(0.5)
  })

  it('publishes a state the registry then serves', async () => {
    const adapter = artifact()
    await adapter.publish({ [ENTRYPOINT]: BODY })
    const listed = await ctx.skills.list({})
    expect(listed.map((s) => s.name)).toContain('sql')
    const loaded = await ctx.skills.get('sql', {})
    expect(loaded?.content.trim()).toBe('Use snake_case table names.')
  })

  it('beats the copy on disk, because the evolved body came from it', async () => {
    ctx.skills.registerProvider(() => onDisk('sql', 'the old body'))
    await artifact().publish({ [ENTRYPOINT]: BODY })
    const loaded = await ctx.skills.get('sql', {})
    // Losing this tie leaves the ledger and the running harness disagreeing
    // with no error anywhere: an accepted commit, and every later rollout still
    // reading the old body.
    expect(loaded?.content.trim()).toBe('Use snake_case table names.')
    expect(EVOLVED_SKILL_RANK).toBeLessThan(100)
  })

  it('republishes without colliding with its own previous registration', async () => {
    const adapter = artifact()
    await adapter.publish({ [ENTRYPOINT]: '---\ndescription: v1\n---\n\nfirst\n' })
    await adapter.publish({ [ENTRYPOINT]: '---\ndescription: v2\n---\n\nsecond\n' })
    const loaded = await ctx.skills.get('sql', {})
    expect(loaded?.content.trim()).toBe('second')
    expect((await ctx.skills.list({})).filter((s) => s.name === 'sql')).toHaveLength(1)
  })

  it('never leaves the skill missing while it is being replaced', async () => {
    const adapter = artifact()
    await adapter.publish({ [ENTRYPOINT]: BODY })
    // The new provider is registered before the old one is disposed, so no
    // lookup in that window can observe an absent skill.
    const during = adapter.publish({ [ENTRYPOINT]: '---\ndescription: v2\n---\n\nnext\n' })
    expect((await ctx.skills.list({})).some((s) => s.name === 'sql')).toBe(true)
    await during
    expect((await ctx.skills.list({})).some((s) => s.name === 'sql')).toBe(true)
  })

  it('withdraws the skill when the publication is disposed', async () => {
    const adapter = artifact()
    const dispose = await adapter.publish({ [ENTRYPOINT]: BODY })
    dispose()
    expect((await ctx.skills.list({})).some((s) => s.name === 'sql')).toBe(false)
  })

  it('binds two candidates at once without either seeing the other', async () => {
    // The claim the in-process rollout design rests on: N workers, one process,
    // each seeing its own candidate.
    const adapter = artifact()
    const a = ctx.isolate('skills')
    const b = ctx.isolate('skills')
    await a.plugin(SkillRegistry)
    await b.plugin(SkillRegistry)

    adapter.bind(a, { [ENTRYPOINT]: '---\ndescription: A\n---\n\ncandidate A\n' })
    adapter.bind(b, { [ENTRYPOINT]: '---\ndescription: B\n---\n\ncandidate B\n' })

    expect((await a.skills.get('sql', {}))?.content.trim()).toBe('candidate A')
    expect((await b.skills.get('sql', {}))?.content.trim()).toBe('candidate B')
  })

  it('unbinds a candidate when its rollout ends', async () => {
    const adapter = artifact()
    const scope = ctx.isolate('skills')
    await scope.plugin(SkillRegistry)
    const undo = adapter.bind(scope, { [ENTRYPOINT]: BODY })
    expect((await scope.skills.list({})).some((s) => s.name === 'sql')).toBe(true)
    undo()
    expect((await scope.skills.list({})).some((s) => s.name === 'sql')).toBe(false)
  })

  it('loads its starting state through the supplied loader', async () => {
    const adapter = artifact({ [ENTRYPOINT]: BODY, 'references/style.md': 'more' })
    expect(Object.keys(await adapter.load()).sort()).toEqual([ENTRYPOINT, 'references/style.md'])
  })
})
