/**
 * The row a profile actually lists.
 *
 * Every other test here exercises a part. This checks the composition: that
 * mounting it registers what `/evolve` needs, that configured artifacts appear,
 * and -- the one that matters at boot -- that a sidecar which fails to start
 * leaves a harness that says so rather than one that accepts `/evolve` and
 * silently does nothing.
 */

import { Context } from '@deepseek-ai/cordis'
import CommandRegistry from '@deepseek-ai/dsh-commands'
import SkillRegistry from '@deepseek-ai/dsh-skill'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRegistry from '@deepseek-ai/dsh-tools'
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { EvolutionRegistry } from '../src/evolution.js'
import { apply, inject, name, REFLECTION_ARTIFACT } from '../src/plugin.js'

function skillDir(): string {
  const root = mkdtempSync(join(tmpdir(), 'dsh-skill-'))
  mkdirSync(join(root, 'references'), { recursive: true })
  writeFileSync(join(root, 'SKILL.md'), '---\ndescription: total\n---\n\nbody\n')
  writeFileSync(join(root, 'references', 'rules.md'), 'COLUMN: id\n')
  return root
}

function pluginDir(): string {
  const root = mkdtempSync(join(tmpdir(), 'dsh-plug-'))
  mkdirSync(join(root, 'src'), { recursive: true })
  writeFileSync(join(root, 'package.json'), '{"name":"dsh-plugin-mine"}')
  writeFileSync(join(root, 'src', 'index.ts'), 'export const name = "mine"\n')
  return root
}

describe('the bundle row', () => {
  let ctx: Context

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRegistry)
    await ctx.plugin(SkillRegistry)
    await ctx.plugin(CommandRegistry)
    await ctx.plugin(EvolutionRegistry)
    // `ctx.agents` is supplied by the harness; nothing here starts a turn.
    Object.defineProperty(ctx, 'agents', { value: { create: async () => { throw new Error('unused') } }, configurable: true })
  })

  it('declares what it needs before it is mounted', () => {
    expect(name).toBe('dsh-plugin-agentdescent')
    for (const service of ['evolution', 'agents', 'skills', 'commands', 'tools']) {
      expect(inject).toContain(service)
    }
  })

  it('registers the surface a user and a model reach for', () => {
    apply(ctx, { spawnGuard: true } as never)
    expect(ctx.commands.list(undefined as never).map((c) => c.name)).toContain('evolve')
    const tools = ctx.tools.schemas().map((schema) => schema.name)
    expect(tools).toContain('evolve_start')
    expect(tools).toContain('evolve_status')
  })

  it('ships rung A and the fallback, and leaves rung B to whoever knows the objective', () => {
    apply(ctx)
    const scorers = ctx.evolution.listScorers()
    expect(scorers).toContain('gold')
    expect(scorers).toContain('replay-pairwise')
    // `command` and `judge` need the user's own check or rubric, so mounting
    // them with invented defaults would be inventing their objective.
    expect(scorers).not.toContain('command')
  })

  it('registers the artifact reflection borrows, before anything can ask for one', () => {
    // Without this the engine's first reflection fails with "unknown artifact",
    // and it fails mid-run rather than at boot.
    apply(ctx)
    expect(ctx.evolution.artifact(REFLECTION_ARTIFACT)).toBeDefined()
    expect(ctx.evolution.artifact(REFLECTION_ARTIFACT)?.blastRadius).toBe(0)
  })

  it('never lets reflection masquerade as a publishable artifact', async () => {
    apply(ctx)
    await expect(ctx.evolution.artifact(REFLECTION_ARTIFACT)!.publish({}))
      .rejects.toThrow(/not an artifact/)
  })

  it('exposes configured skills and plugins at their own layers', async () => {
    apply(ctx, { skills: [{ name: 'total', dir: skillDir() }],
                 plugins: [{ name: 'dsh-plugin-mine', dir: pluginDir() }] })
    expect(ctx.evolution.listArtifacts()).toContain('skill:total')
    expect(ctx.evolution.listArtifacts()).toContain('plugin:dsh-plugin-mine')
    expect(ctx.evolution.artifact('skill:total')?.blastRadius).toBeLessThan(0.5)
    expect(ctx.evolution.artifact('plugin:dsh-plugin-mine')?.blastRadius).toBeGreaterThanOrEqual(0.5)
  })

  it('loads a configured skill as a tree, not just its body', async () => {
    apply(ctx, { skills: [{ name: 'total', dir: skillDir() }] })
    const tree = await ctx.evolution.artifact('skill:total')!.load()
    expect(Object.keys(tree).sort()).toEqual(['SKILL.md', 'references/rules.md'])
  })

  it('says so loudly when the sidecar cannot start', async () => {
    // The failure that would otherwise be silent: a harness that booted, accepts
    // /evolve, and never runs anything.
    const error = vi.spyOn(ctx.logger, 'error').mockImplementation(() => {})
    apply(ctx, { python: 'definitely-not-a-python-on-this-machine' })
    await vi.waitFor(() => { expect(error).toHaveBeenCalled() }, { timeout: 5000 })
    expect(String(error.mock.calls[0]?.[0])).toMatch(/did not start/)
  })

  it('leaves /evolve reporting no engine rather than pretending', async () => {
    const error = vi.spyOn(ctx.logger, 'error').mockImplementation(() => {})
    apply(ctx, { python: 'definitely-not-a-python-on-this-machine',
                 skills: [{ name: 'total', dir: skillDir() }] })
    // Tasks are the user's, so the bundle ships no source; register one to get
    // past resolution and reach the engine.
    ctx.evolution.registerTaskSource({ name: 'manual', fetch: async () => [] })
    await vi.waitFor(() => { expect(error).toHaveBeenCalled() }, { timeout: 5000 })
    const definition = ctx.commands.find(undefined as never, 'evolve')
    const result = await definition!.handler(
      { name: 'evolve', rawInput: 'skill:total gold manual' } as never)
    expect(result.kind).toBe('error')
    expect(result.text).toMatch(/no evolution engine/)
  })
})

describe('saying what to train on, in configuration', () => {
  let ctx: Context

  beforeEach(async () => {
    ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRegistry)
    await ctx.plugin(SkillRegistry)
    await ctx.plugin(CommandRegistry)
    await ctx.plugin(EvolutionRegistry)
    Object.defineProperty(ctx, 'agents', {
      value: { create: async () => { throw new Error('unused') } }, configurable: true,
    })
  })

  function dataset(lines: string[]): string {
    const file = join(mkdtempSync(join(tmpdir(), 'dsh-cfg-')), 'tasks.jsonl')
    writeFileSync(file, lines.join('\n'), 'utf8')
    return file
  }

  it('registers a declared dataset as a task source', async () => {
    const file = dataset([JSON.stringify({ prompt: 'q1', gold: 'a1' }), 'a bare question'])
    apply(ctx, { datasets: [{ name: 'sql-regressions', file }] })
    expect(ctx.evolution.listTaskSources()).toContain('sql-regressions')
    const tasks = await ctx.evolution.taskSource('sql-regressions')!.fetch(10)
    expect(tasks.map((t) => t.prompt)).toEqual(['q1', 'a bare question'])
  })

  it('registers a declared objective as a scorer, under the name given', () => {
    apply(ctx, {
      objectives: [{ name: 'runs-on-my-db', kind: 'command', run: ['node', '-e', 'process.exit(0)'] }],
    })
    expect(ctx.evolution.listScorers()).toContain('runs-on-my-db')
  })

  it('makes /evolve able to name both without any TypeScript', async () => {
    // The whole point: the two decisions that are the user's are reachable from
    // a config file.
    const file = dataset([JSON.stringify({ prompt: 'q', gold: 'a' })])
    apply(ctx, {
      skills: [{ name: 'total', dir: skillDir() }],
      datasets: [{ name: 'mine', file }],
      objectives: [{ name: 'my-check', kind: 'command', run: ['node', '-e', 'process.exit(0)'] }],
    })
    const definition = ctx.commands.find(undefined as never, 'evolve')
    const listed = await definition!.handler({ name: 'evolve', rawInput: '' } as never)
    expect(listed.text).toContain('skill:total')
    expect(listed.text).toContain('my-check')
    expect(listed.text).toContain('mine')
  })

  it('refuses to mount on a misconfigured objective, naming it', () => {
    // A run that quietly used a different objective than the one intended is
    // not something anyone can detect afterwards.
    expect(() => apply(ctx, { objectives: [{ name: 'typo', kind: 'vibes' } as never] }))
      .toThrow(/objective "typo".*unknown kind/s)
  })

  it('refuses to mount on a dataset that is not there', () => {
    expect(() => apply(ctx, { datasets: [{ name: 'gone', file: '/no/such/file.jsonl' }] }))
      .toThrow(/cannot be read/)
  })
})
