/**
 * The library artifact, against the real skill registry.
 *
 * `skill:<name>` refines one skill; this is the half that can add one nobody
 * wrote and drop one that never helped. Deletion is the case worth guarding:
 * a per-skill provider design makes additions work and deletions silently not.
 */

import { Context } from '@deepseek-ai/cordis'
import SkillRegistry from '@deepseek-ai/dsh-skill'
import { beforeEach, describe, expect, it } from 'vitest'

import { parseLibrary, skillLibraryArtifact } from '../src/artifacts/library.js'

function skill(name: string, description = name, body = 'do the thing'): Record<string, string> {
  return { [`${name}/SKILL.md`]: `---\ndescription: ${description}\n---\n\n${body}\n` }
}

const LIBRARY = { ...skill('sql', 'write SQL'), ...skill('csv', 'total a column') }

describe('reading a library tree', () => {
  it('finds a skill per top-level directory holding SKILL.md', () => {
    expect(parseLibrary(LIBRARY).map((s) => s.name)).toEqual(['csv', 'sql'])
  })

  it('treats nested files as resources, not skills', () => {
    const withResource = { ...LIBRARY, 'sql/references/style.md': 'snake_case' }
    expect(parseLibrary(withResource).map((s) => s.name)).toEqual(['csv', 'sql'])
  })

  it('does not promote a stray root file into the catalogue', () => {
    // A README becoming a model-facing skill is the kind of thing nobody
    // notices until the model starts citing it.
    expect(parseLibrary({ ...LIBRARY, 'README.md': 'about this library' })
      .map((s) => s.name)).toEqual(['csv', 'sql'])
  })

  it('ignores a dotted directory', () => {
    expect(parseLibrary({ ...LIBRARY, ...skill('.system') }).map((s) => s.name))
      .toEqual(['csv', 'sql'])
  })

  it('is empty for an empty library rather than throwing', () => {
    expect(parseLibrary({})).toEqual([])
  })
})

describe('the library artifact', () => {
  let ctx: Context

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(SkillRegistry)
  })

  function artifact(state: Record<string, string> = LIBRARY) {
    return skillLibraryArtifact(ctx, { name: 'user', load: async () => state })
  }

  it('is addressed as skills:<name> and merges on score', () => {
    expect(artifact().id).toBe('skills:user')
    expect(artifact().blastRadius).toBeLessThan(0.5)
  })

  it('serves every skill in the library at once', async () => {
    await artifact().publish(LIBRARY)
    const listed = await ctx.skills.list({})
    expect(listed.map((s) => s.name).sort()).toEqual(['csv', 'sql'])
    expect((await ctx.skills.get('sql', {}))?.description).toBe('write SQL')
  })

  it('serves a skill the run invented, which is the point of evolving a library', async () => {
    const adapter = artifact()
    await adapter.publish(LIBRARY)
    await adapter.publish({ ...LIBRARY, ...skill('regex', 'write careful regexes') })
    const listed = await ctx.skills.list({})
    expect(listed.map((s) => s.name).sort()).toEqual(['csv', 'regex', 'sql'])
  })

  it('stops serving a skill the run removed', async () => {
    // The failure a per-skill provider design produces: the model keeps seeing
    // a skill the ledger says was deleted, so the held-out score no longer
    // describes what the harness does.
    const adapter = artifact()
    await adapter.publish(LIBRARY)
    await adapter.publish(skill('sql', 'write SQL'))
    const listed = await ctx.skills.list({})
    expect(listed.map((s) => s.name)).toEqual(['sql'])
    expect(await ctx.skills.get('csv', {})).toBeUndefined()
  })

  it('never leaves the library empty while it is being replaced', async () => {
    const adapter = artifact()
    await adapter.publish(LIBRARY)
    const during = adapter.publish({ ...LIBRARY, ...skill('new') })
    expect((await ctx.skills.list({})).length).toBeGreaterThan(0)
    await during
    expect((await ctx.skills.list({})).length).toBe(3)
  })

  it('withdraws the whole library when the publication is disposed', async () => {
    const dispose = await artifact().publish(LIBRARY)
    dispose()
    expect(await ctx.skills.list({})).toEqual([])
  })

  it('binds a candidate library to one agent without the others seeing it', async () => {
    const adapter = artifact()
    const a = ctx.isolate('skills')
    const b = ctx.isolate('skills')
    await a.plugin(SkillRegistry)
    await b.plugin(SkillRegistry)

    adapter.bind(a, { ...LIBRARY, ...skill('only-a', 'candidate A') })
    adapter.bind(b, skill('sql', 'write SQL'))

    expect((await a.skills.list({})).map((s) => s.name).sort()).toEqual(['csv', 'only-a', 'sql'])
    expect((await b.skills.list({})).map((s) => s.name)).toEqual(['sql'])
  })

  it('carries a skill body through to the model', async () => {
    await artifact().publish(skill('sql', 'write SQL', 'Use snake_case table names.'))
    expect((await ctx.skills.get('sql', {}))?.content.trim())
      .toBe('Use snake_case table names.')
  })
})
