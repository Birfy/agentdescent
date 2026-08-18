/**
 * The `plugin:<pkg>` artifact.
 *
 * Two behaviours carry this file, and both are refusals: the governance surface
 * cannot be written, and a code candidate cannot be bound to one agent. Neither
 * failure would announce itself if it were allowed through -- one removes the
 * reviewer, the other reports "nothing improved" from a measurement that never
 * varied.
 */

import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Context } from '@deepseek-ai/cordis'
import { beforeEach, describe, expect, it } from 'vitest'

import {
  FROZEN_PLUGIN_PATHS, isFrozen, PLUGIN_BLAST_RADIUS, pluginArtifact,
} from '../src/artifacts/plugin.js'

function scaffold(): string {
  const root = mkdtempSync(join(tmpdir(), 'dsh-plugin-'))
  mkdirSync(join(root, 'src'), { recursive: true })
  mkdirSync(join(root, 'test'), { recursive: true })
  mkdirSync(join(root, 'node_modules', 'left-pad'), { recursive: true })
  mkdirSync(join(root, 'lib'), { recursive: true })
  writeFileSync(join(root, 'package.json'), '{"name":"dsh-plugin-mine"}')
  writeFileSync(join(root, 'cordis.patch.yml'), '- insert: []\n')
  writeFileSync(join(root, 'src', 'index.ts'), 'export const name = "mine"\n')
  writeFileSync(join(root, 'test', 'index.test.ts'), 'it("works", () => {})\n')
  writeFileSync(join(root, 'node_modules', 'left-pad', 'index.js'), 'module.exports = 1\n')
  writeFileSync(join(root, 'lib', 'index.js'), 'built output\n')
  return root
}

describe('what counts as the governance surface', () => {
  it('freezes the files that can rewrite the rules or move the goalposts', () => {
    expect(isFrozen('cordis.patch.yml')).toBe(true)   // decides which rows mount
    expect(isFrozen('package.json')).toBe(true)       // install-time scripts
    expect(isFrozen('test/index.test.ts')).toBe(true) // the gate
    expect(isFrozen('src/foo.spec.ts')).toBe(true)
  })

  it('leaves ordinary source writable', () => {
    expect(isFrozen('src/index.ts')).toBe(false)
    expect(isFrozen('README.md')).toBe(false)
  })

  it('mirrors the engine-side list rather than trusting it', () => {
    // Duplicated on purpose: a rule enforced only where the change is proposed
    // is enforced only as long as that side is correct.
    expect(FROZEN_PLUGIN_PATHS).toContain('cordis.patch.yml')
    expect(FROZEN_PLUGIN_PATHS).toContain('package.json')
  })
})

describe('the plugin artifact', () => {
  let ctx: Context
  let root: string

  beforeEach(() => {
    ctx = new Context()
    root = scaffold()
  })

  function artifact() {
    return pluginArtifact(ctx, { name: 'dsh-plugin-mine', packageDir: root })
  }

  it('is L1, so every merge goes through the oracle', () => {
    expect(artifact().blastRadius).toBe(PLUGIN_BLAST_RADIUS)
    expect(PLUGIN_BLAST_RADIUS).toBeGreaterThanOrEqual(0.5)
  })

  it('is addressed as plugin:<name>', () => {
    expect(artifact().id).toBe('plugin:dsh-plugin-mine')
  })

  it('loads the source and leaves out what is not source', async () => {
    const tree = await artifact().load()
    expect(Object.keys(tree).sort()).toEqual([
      'cordis.patch.yml', 'package.json', 'src/index.ts', 'test/index.test.ts',
    ])
    expect(tree['src/index.ts']).toContain('export const name')
  })

  it('refuses to bind, because a code candidate cannot be scoped to one agent', () => {
    // A no-op bind would run every rollout against the code already on disk,
    // score every candidate identically, and report that nothing improved --
    // a wrong answer wearing a valid-looking measurement.
    expect(() => artifact().bind(ctx, {})).toThrow(/separate process/)
  })

  it('publishes ordinary source to disk, where HMR picks it up', async () => {
    await artifact().publish({ 'src/index.ts': 'export const name = "evolved"\n' })
    expect(readFileSync(join(root, 'src', 'index.ts'), 'utf8')).toContain('evolved')
  })

  it('creates directories a candidate adds', async () => {
    await artifact().publish({ 'src/deep/new.ts': 'export const added = true\n' })
    expect(readFileSync(join(root, 'src', 'deep', 'new.ts'), 'utf8')).toContain('added')
  })

  it('refuses loudly when a state carries the governance surface', async () => {
    // Silently dropping them would leave the ledger believing it published
    // something it did not.
    await expect(artifact().publish({
      'src/index.ts': 'fine\n',
      'cordis.patch.yml': '- insert: [{id: approval, name: nothing}]\n',
    })).rejects.toThrow(/refusing to publish the governance surface/)
    expect(readFileSync(join(root, 'cordis.patch.yml'), 'utf8')).toBe('- insert: []\n')
    expect(readFileSync(join(root, 'src', 'index.ts'), 'utf8')).toContain('export const name = "mine"')
  })

  it('names every refused path, not just the first', async () => {
    await expect(artifact().publish({
      'package.json': '{}', 'cordis.patch.yml': 'x', 'test/index.test.ts': 'it.skip()',
    })).rejects.toThrow(/cordis\.patch\.yml.*package\.json.*test\/index\.test\.ts/s)
  })

  it('restores what was there when the publication is undone', async () => {
    const before = readFileSync(join(root, 'src', 'index.ts'), 'utf8')
    const undo = await artifact().publish({ 'src/index.ts': 'evolved\n' })
    undo()
    expect(readFileSync(join(root, 'src', 'index.ts'), 'utf8')).toBe(before)
  })

  it('skips a file too large to reason about as a diff', async () => {
    writeFileSync(join(root, 'src', 'huge.bin'), 'x'.repeat(2048))
    const tree = await pluginArtifact(ctx, {
      name: 'dsh-plugin-mine', packageDir: root, maxFileBytes: 1024,
    }).load()
    expect(Object.keys(tree)).not.toContain('src/huge.bin')
    expect(Object.keys(tree)).toContain('src/index.ts')
  })
})
