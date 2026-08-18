/**
 * The mountable plugin: one `apply(ctx)` that composes everything else here.
 *
 * Every other module in this package is a part -- the seam, the engine, the
 * adapters, the scorers, the surface. This is the row a profile actually lists,
 * and the only place that decides defaults.
 *
 * @module dsh-plugin-agentdescent/plugin
 */

import { Context } from '@deepseek-ai/cordis'

import { CommitQueue } from './approval.js'
import { pluginArtifact } from './artifacts/plugin.js'
import { skillArtifact } from './artifacts/skill.js'
import { startEngine } from './engine.js'
import type { EvolutionSpec } from './evolution.js'
import { idleTrigger } from './idle.js'
import { inProcessRollout } from './rollout.js'
import { efficiency, goldScorer, replayPairwiseScorer } from './scorers.js'
import { registerCommand, registerTools } from './surface.js'

export const name = 'dsh-plugin-agentdescent'

/** Mounted after these, because every one of them is used on the first run. */
export const inject = ['evolution', 'agents', 'skills', 'commands', 'tools']

export interface Config {
  /** Interpreter for the optimiser sidecar. */
  readonly python?: string
  /** Where the sidecar runs. Must be somewhere `agentdescent` imports from. */
  readonly cwd?: string
  /** Skills to expose as `skill:<name>` artifacts. */
  readonly skills?: readonly { readonly name: string; readonly dir: string }[]
  /** Plugin checkouts to expose as `plugin:<name>` artifacts (L1). */
  readonly plugins?: readonly { readonly name: string; readonly dir: string }[]
  /**
   * The highest blast radius that merges without a person.
   *
   * The default keeps skills automatic and everything harness-shaped in the
   * review queue, which is the line the governance design draws.
   */
  readonly autoMergeMaxBlastRadius?: number
  readonly tokenBudget?: number
  readonly rolloutTimeoutMs?: number
  /**
   * Runs to start on their own once the harness goes quiet.
   *
   * Empty by default. A harness that started spending tokens the first time you
   * walked away, without being asked, would be a surprise -- and the kind that
   * shows up on a bill.
   */
  readonly idleRuns?: readonly EvolutionSpec[]
  readonly idleAfterMs?: number
  readonly idleMinIntervalMs?: number
}

/**
 * Compose the evolution capability into this harness.
 *
 * The sidecar is started **lazily is not an option**: the engine has to be
 * registered before `/evolve` can resolve anything, and a command that reports
 * "no engine" until someone happens to run something twice is worse than a
 * slower boot.
 */
export function apply(ctx: Context, config: Config = {}): void {
  const queue = new CommitQueue(
    config.autoMergeMaxBlastRadius === undefined
      ? {}
      : { autoMergeMaxBlastRadius: config.autoMergeMaxBlastRadius })

  for (const skill of config.skills ?? []) {
    ctx.evolution.registerArtifact(skillArtifact(ctx, {
      name: skill.name,
      load: async () => await readSkillDir(skill.dir),
    }))
  }
  for (const plugin of config.plugins ?? []) {
    ctx.evolution.registerArtifact(pluginArtifact(ctx, {
      name: plugin.name, packageDir: plugin.dir,
    }))
  }

  // Reflection borrows an artifact id so its turn takes the same logged,
  // budgeted, sandboxed path as a rollout. Registered here, before the engine
  // can ask for a completion.
  registerReflectionArtifact(ctx)

  // Rung A and the fallback ship mounted; rung B needs the user's own command
  // or rubric, so it is registered by whoever knows what the objective is.
  ctx.evolution.registerScorer(goldScorer())
  ctx.evolution.registerScorer(efficiency(goldScorer({ name: 'gold-efficient' })))
  ctx.evolution.registerScorer(replayPairwiseScorer())

  registerCommand(ctx, { queue })
  registerTools(ctx)

  if ((config.idleRuns ?? []).length > 0) {
    idleTrigger(ctx, {
      evolution: ctx.evolution,
      runs: config.idleRuns ?? [],
      ...(config.idleAfterMs === undefined ? {} : { afterMs: config.idleAfterMs }),
      ...(config.idleMinIntervalMs === undefined
        ? {}
        : { minIntervalMs: config.idleMinIntervalMs }),
    })
  }

  const rollout = inProcessRollout(ctx, {
    evolution: ctx.evolution,
    ...(config.rolloutTimeoutMs === undefined ? {} : { timeoutMs: config.rolloutTimeoutMs }),
  })

  // The engine is async to start, and a failure here must be loud: a harness
  // that booted with a dead sidecar would accept `/evolve` and never run it.
  void startEngine(ctx, {
    evolution: ctx.evolution,
    queue,
    rollout,
    complete: async (prompt) => (await rollout(
      REFLECTION_ARTIFACT, {}, { id: 'reflect', prompt })).output,
    ...(config.python === undefined ? {} : { python: config.python }),
    ...(config.cwd === undefined ? {} : { cwd: config.cwd }),
  }).then(
    (engine) => { ctx.evolution.registerEngine(engine) },
    (error: unknown) => {
      ctx.logger.error(
        `the evolution engine did not start: ${String(error)}. /evolve will report ` +
        'that no engine is mounted rather than silently doing nothing.')
    },
  )
}

/**
 * The artifact id reflection borrows.
 *
 * The engine holds no credentials by design, so every completion it needs comes
 * back here -- and running it through the ordinary rollout path means a
 * reflection is logged, budgeted and sandboxed exactly like any other model use
 * in this harness, rather than being a second unpoliced way to reach the model.
 * The adapter is a no-op so nothing pretends to evolve.
 */
export const REFLECTION_ARTIFACT = 'internal:reflection'

/** Register the no-op adapter reflection borrows. Call once at mount. */
export function registerReflectionArtifact(ctx: Context): () => void {
  return ctx.evolution.registerArtifact({
    id: REFLECTION_ARTIFACT,
    blastRadius: 0,
    load: async () => ({}),
    bind: () => () => {},
    publish: async () => {
      throw new Error('internal:reflection is not an artifact; it cannot be published')
    },
  })
}

/** Read a skill directory as `{relative path -> content}`. */
async function readSkillDir(dir: string): Promise<Record<string, string>> {
  const { readdirSync, readFileSync, statSync } = await import('node:fs')
  const { join, relative, sep } = await import('node:path')
  const tree: Record<string, string> = {}
  const walk = (directory: string): void => {
    for (const entry of readdirSync(directory)) {
      const full = join(directory, entry)
      if (statSync(full).isDirectory()) walk(full)
      else tree[relative(dir, full).split(sep).join('/')] = readFileSync(full, 'utf8')
    }
  }
  walk(dir)
  return tree
}

export default apply
