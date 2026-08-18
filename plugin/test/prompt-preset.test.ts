/**
 * The two L1-adjacent artifacts: a prompt section, and a whole preset.
 *
 * A prompt section is in every request, so the case worth testing is *when* a
 * commit lands rather than that it lands. A preset decides what an agent may
 * do, so the case worth testing is what an empty or trimmed policy means.
 */

import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRegistry from '@deepseek-ai/dsh-tools'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { beforeEach, describe, expect, it } from 'vitest'

import { parseToolPolicy, presetArtifact, PERSONA_FILE, TOOLS_FILE } from '../src/artifacts/preset.js'
import { EVOLVED_SECTION_ORDER, promptArtifact, SECTION_FILE, whenIdle } from '../src/artifacts/prompt.js'

describe('the prompt section artifact', () => {
  let ctx: Context

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(SystemPrompt)
  })

  function artifact(overrides = {}) {
    return promptArtifact(ctx, {
      name: 'house-style', load: async () => ({ [SECTION_FILE]: 'be terse' }), ...overrides,
    })
  }

  it('is addressed as prompt:<name> and sits after the persona', () => {
    expect(artifact().id).toBe('prompt:house-style')
    expect(EVOLVED_SECTION_ORDER).toBeGreaterThan(0)     // after the persona
    expect(EVOLVED_SECTION_ORDER).toBeLessThan(100)      // before tool guidance
  })

  it('publishes the section into the prompt', async () => {
    await artifact().publish({ [SECTION_FILE]: 'always cite the table' })
    const assembled = await ctx.systemPrompt.assemble({} as never)
    expect(JSON.stringify(assembled)).toContain('always cite the table')
  })

  it('replaces its own previous section instead of colliding with it', async () => {
    // A duplicate section name throws, so the two can never briefly coexist --
    // which is why this adapter disposes before registering.
    const adapter = artifact()
    await adapter.publish({ [SECTION_FILE]: 'first' })
    await adapter.publish({ [SECTION_FILE]: 'second' })
    const assembled = JSON.stringify(await ctx.systemPrompt.assemble({} as never))
    expect(assembled).toContain('second')
    expect(assembled).not.toContain('first')
  })

  it('waits for a turn boundary before swapping the prefix', async () => {
    // The whole reason this adapter is different: swapping mid-turn rewrites
    // the request prefix and every live session's KV cache goes with it.
    let release: () => void = () => {}
    const gate = new Promise<void>((resolve) => { release = resolve })
    const adapter = artifact({ atTurnBoundary: () => gate })

    let landed = false
    const publishing = adapter.publish({ [SECTION_FILE]: 'new text' }).then(() => { landed = true })
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(landed).toBe(false)
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never))).not.toContain('new text')

    release()
    await publishing
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never))).toContain('new text')
  })

  it('publishes immediately when there is no boundary to wait for', async () => {
    // Headless and one-shot compositions have no live session whose cache
    // could be lost; delaying there would be ceremony.
    await artifact().publish({ [SECTION_FILE]: 'immediate' })
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never))).toContain('immediate')
  })

  it('keeps a candidate to one agent, shadowing the committed section', async () => {
    const adapter = artifact()
    await adapter.publish({ [SECTION_FILE]: 'committed' })
    const child = ctx.isolate('systemPrompt')
    await child.plugin(SystemPrompt)
    adapter.bind(child, { [SECTION_FILE]: 'candidate' })

    expect(JSON.stringify(await child.systemPrompt.assemble({} as never))).toContain('candidate')
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never))).toContain('committed')
  })

  it('does not lose half a section split across files', async () => {
    await artifact().publish({ 'a.md': 'first half', 'b.md': 'second half' })
    const assembled = JSON.stringify(await ctx.systemPrompt.assemble({} as never))
    expect(assembled).toContain('first half')
    expect(assembled).toContain('second half')
  })

  it('withdraws the section when the publication is disposed', async () => {
    const dispose = await artifact().publish({ [SECTION_FILE]: 'temporary' })
    dispose()
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never)))
      .not.toContain('temporary')
  })

  it('whenIdle resolves at once when nothing is running', async () => {
    await expect(whenIdle(ctx, () => false)()).resolves.toBeUndefined()
  })

  it('whenIdle waits until the last agent goes quiet', async () => {
    let busy = true
    const waiting = whenIdle(ctx, () => busy)()
    let done = false
    void waiting.then(() => { done = true })
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(done).toBe(false)
    busy = false
    ;(ctx as any).emit('agent/status', { agent: { id: 'a' }, status: 'idle' })
    await waiting
    expect(done).toBe(true)
  })
})

describe('reading a tool policy', () => {
  it('allows bare names and denies prefixed ones', () => {
    expect(parseToolPolicy('bash\nread\n-web_search\n')).toEqual({
      allow: ['bash', 'read'], deny: ['web_search'],
    })
  })

  it('ignores comments and blank lines', () => {
    expect(parseToolPolicy('# only these\n\nbash\n')).toEqual({ allow: ['bash'], deny: [] })
  })

  it('reads an empty file as no restriction, not as no tools', () => {
    // The other reading turns a reflector that trimmed the file into an agent
    // that can do nothing, which looks like a broken harness rather than a bad
    // proposal.
    expect(parseToolPolicy('   \n\n')).toEqual({ allow: [], deny: [] })
  })
})

describe('the preset artifact', () => {
  let ctx: Context

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRegistry)
    for (const name of ['bash', 'web_search', 'read']) {
      ctx.tools.register(defineTool({
        name, description: name, parameters: {},
        output: { schema: { type: 'string' }, render: () => [] },
        async execute() { return name },
      }))
    }
  })

  function artifact(state: Record<string, string>) {
    return presetArtifact(ctx, { name: 'careful', load: async () => state })
  }

  it('is L1, because it decides what an agent may do', () => {
    expect(artifact({}).blastRadius).toBeGreaterThanOrEqual(0.5)
    expect(artifact({}).id).toBe('preset:careful')
  })

  it('publishes the persona globally, which is what a deployment persona is', async () => {
    await artifact({}).publish({ [PERSONA_FILE]: 'You never guess.' })
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never)))
      .toContain('You never guess.')
  })

  it('does NOT mask tools globally, because dsh refuses an unscoped restriction', async () => {
    // Not our shortcut: tools.restrict() throws on a plain context, because a
    // global mask would hit every agent in the process -- including the ones
    // not using this preset. The committed policy is applied per agent instead.
    const adapter = artifact({})
    await expect(adapter.publish({ [TOOLS_FILE]: '-web_search\n' })).resolves.toBeTypeOf('function')
    expect(ctx.tools.schemas().map((s) => s.name)).toContain('web_search')
  })

  it('keeps the committed state for a composition to apply per agent', async () => {
    const adapter = artifact({}) as any
    await adapter.publish({ [TOOLS_FILE]: '-web_search\n' })
    expect(adapter.committedState()).toEqual({ [TOOLS_FILE]: '-web_search\n' })
  })

  it('leaves every tool alone when the policy is empty', () => {
    const child = ctx.isolate('tools')
    artifact({}).bind(child, { [TOOLS_FILE]: '# nothing here\n' })
    expect(ctx.tools.schemas()).toHaveLength(3)
  })

  it('demands an agent context for the tool half, and says so', () => {
    // A preset's mask is per-agent by dsh's own rule. Binding somewhere without
    // an agent scope is a wiring mistake, and the message names the fix rather
    // than leaving the registry's more general one to be interpreted.
    expect(() => artifact({}).bind(ctx, { [TOOLS_FILE]: '-web_search\n' }))
      .toThrow(/per-agent, so it needs that agent's own context \(agent\.ctx\)/)
  })

  it('does not half-apply a preset whose tool policy is refused', async () => {
    // The persona would otherwise stay behind after the tool half threw.
    expect(() => artifact({}).bind(ctx, { [PERSONA_FILE]: 'orphan', [TOOLS_FILE]: '-web_search\n' }))
      .toThrow()
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never))).not.toContain('orphan')
  })

  it('lifts the persona it applied when unbound', async () => {
    const undo = artifact({}).bind(ctx, { [PERSONA_FILE]: 'candidate persona' })
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never)))
      .toContain('candidate persona')
    undo()
    expect(JSON.stringify(await ctx.systemPrompt.assemble({} as never)))
      .not.toContain('candidate persona')
  })
})
