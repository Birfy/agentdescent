/**
 * The `plugin:<pkg>` artifact: a dsh plugin's own source as an evolvable tree.
 *
 * This is the L1 case. A skill is text the agent reads and a bad one costs a
 * wrong answer; a plugin is code the harness executes, and a bad one costs the
 * harness. So the blast radius forces every merge through the oracle, the
 * governance surface cannot be written at all, and `bind` refuses outright.
 *
 * @module dsh-plugin-agentdescent/artifacts/plugin
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'

import type { Context } from '@deepseek-ai/cordis'

import type { ArtifactAdapter, ArtifactId, ArtifactState } from '../evolution.js'

/**
 * Paths a plugin may never rewrite while being evolved.
 *
 * The mirror of `agentdescent.dsh.plugin.FROZEN_PLUGIN_PATHS`, and deliberately
 * duplicated rather than imported across the bridge. The engine checks its copy
 * and this checks its own, because a governance rule enforced only on the side
 * that proposes the change is enforced only as long as that side is correct.
 *
 * - `cordis.patch.yml` decides which rows mount, including the approval gate,
 *   the sandbox backend and this engine's own. Editing it is editing the
 *   reviewer.
 * - `package.json` carries `prepare`/`postinstall` -- code that runs outside the
 *   agent's sandbox -- and the `dsh.bundle` pointer.
 * - the tests are the gate; a candidate that can edit them can pass itself.
 */
export const FROZEN_PLUGIN_PATHS: readonly string[] = [
  'cordis.patch.yml',
  'cordis.patch.yaml',
  'package.json',
]

const FROZEN_PREFIXES: readonly string[] = ['test/', 'tests/']
const FROZEN_SUFFIXES: readonly string[] = ['.test.ts', '.spec.ts']

/** Never source: build output, dependencies, history. */
const IGNORED: readonly string[] = ['node_modules', 'lib', 'dist', '.git', 'coverage']

/** L1: a plugin is harness, so every merge is forced through the oracle. */
export const PLUGIN_BLAST_RADIUS = 0.6

/** Whether a relative path is part of the governance surface. */
export function isFrozen(path: string): boolean {
  const normalised = path.split(sep).join('/')
  return FROZEN_PLUGIN_PATHS.includes(normalised)
    || FROZEN_PREFIXES.some((prefix) => normalised.startsWith(prefix))
    || FROZEN_SUFFIXES.some((suffix) => normalised.endsWith(suffix))
}

export interface PluginArtifactOptions {
  readonly id?: ArtifactId
  /** The package name, as the profile depends on it. */
  readonly name: string
  /** The checkout to read and write. Must be a source directory, not a tarball. */
  readonly packageDir: string
  /** Bytes above which a file is treated as an asset and left out of the tree. */
  readonly maxFileBytes?: number
}

const DEFAULT_MAX_FILE_BYTES = 256 * 1024

/**
 * An {@link ArtifactAdapter} for one plugin package.
 *
 * `bind` throws, and that is the whole point of it existing. A skill candidate
 * can be scoped to one agent because a skill is data the registry serves; a
 * *code* candidate cannot, because Node has already loaded the module and one
 * process holds one version of it. A `bind` that quietly did nothing would be
 * far worse than an error: every rollout would run the code currently on disk,
 * every candidate would score identically, and the run would report "nothing
 * improved" — a wrong conclusion that looks like a valid measurement. Code
 * candidates need a separate process, which is what the out-of-harness executor
 * is for.
 */
export function pluginArtifact(ctx: Context, options: PluginArtifactOptions): ArtifactAdapter {
  const root = resolve(options.packageDir)
  const id = options.id ?? `plugin:${options.name}`
  const maxBytes = options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES

  return {
    id,
    blastRadius: PLUGIN_BLAST_RADIUS,

    load: async () => readTree(root, maxBytes),

    bind(): () => void {
      throw new Error(
        `${id} cannot be bound to one agent: it is code, and this process has ` +
        'already loaded it. Measuring a code candidate needs a separate ' +
        'process — use the out-of-harness executor. (A no-op bind would run ' +
        'every rollout against the code already on disk and report that nothing ' +
        'improved, which is a wrong answer wearing a valid-looking measurement.)')
    },

    async publish(state: ArtifactState): Promise<() => void> {
      const written = Object.keys(state).filter((path) => !isFrozen(path)).sort()
      const refused = Object.keys(state).filter((path) => isFrozen(path)).sort()
      if (refused.length > 0) {
        // Loud, not silent. A commit that quietly dropped these would leave the
        // ledger believing it published something it did not.
        throw new Error(
          `${id}: refusing to publish the governance surface (${refused.join(', ')}). ` +
          'Those files decide which rows mount, what runs at install time, and ' +
          'what the tests check — they are frozen on both sides of the bridge.')
      }
      const previous = new Map<string, string | undefined>()
      for (const path of written) {
        const target = join(root, path)
        previous.set(path, existsSync(target) ? readFileSync(target, 'utf8') : undefined)
        mkdirSync(dirname(target), { recursive: true })
        writeFileSync(target, state[path] ?? '', 'utf8')
      }
      ctx.logger.info(`${id}: wrote ${written.length} file(s) to ${root}`)
      // The disposer restores what was there. HMR reloads from disk either way,
      // so undoing the write is undoing the deployment.
      return () => {
        for (const [path, body] of previous) {
          if (body !== undefined) writeFileSync(join(root, path), body, 'utf8')
        }
      }
    },
  }
}

/** Read a package directory as `{relative path -> content}`. */
function readTree(root: string, maxBytes: number): ArtifactState {
  const tree: Record<string, string> = {}
  const walk = (directory: string): void => {
    let entries: string[]
    try {
      entries = readdirSync(directory)
    } catch {
      return
    }
    for (const entry of entries) {
      if (IGNORED.includes(entry)) continue
      const full = join(directory, entry)
      let info
      try {
        info = statSync(full)
      } catch {
        continue
      }
      if (info.isDirectory()) {
        walk(full)
        continue
      }
      // A tree is the unit of aggregation, and an artifact carrying a binary is
      // an artifact whose diffs cannot be reasoned about. Assets stay on disk.
      if (info.size > maxBytes) continue
      try {
        tree[relative(root, full).split(sep).join('/')] = readFileSync(full, 'utf8')
      } catch {
        continue                              // unreadable or not text
      }
    }
  }
  walk(root)
  return tree
}
