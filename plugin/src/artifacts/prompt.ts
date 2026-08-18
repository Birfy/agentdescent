/**
 * The `prompt:<section>` artifact: an evolvable piece of the system prompt.
 *
 * A skill is read on demand; a prompt section is in **every request**. That
 * makes it the highest-leverage artifact here and the one with a cost nothing
 * else has: swapping it rewrites the request prefix, and every live session's
 * KV cache goes with it.
 *
 * So publishing waits for a turn boundary. dsh treats prompt-prefix stability as
 * a first-class concern -- every package's README has a "KV Cache effect"
 * section -- and a commit that landed mid-turn would invalidate the cache of
 * every session in flight, to save a wait of seconds. Rollouts bind freely:
 * a child agent has no cache worth keeping.
 *
 * @module dsh-plugin-agentdescent/artifacts/prompt
 */

import type { Context } from '@deepseek-ai/cordis'

import type { ArtifactAdapter, ArtifactId, ArtifactState } from '../evolution.js'

/** The file in the tree whose text is the section. */
export const SECTION_FILE = 'SECTION.md'

/**
 * Where an evolved section sits in the prompt.
 *
 * After the deployment persona (0) and before tool guidance (100-199): it is
 * the user's own standing instruction, so it should read as coming after who
 * the agent is and before how its tools work.
 */
export const EVOLVED_SECTION_ORDER = 50

export interface PromptArtifactOptions {
  readonly id?: ArtifactId
  /** The section name. Must be unique among registered sections. */
  readonly name: string
  load: () => Promise<ArtifactState>
  readonly order?: number
  readonly blastRadius?: number
  /**
   * Resolves at the next turn boundary. Publishing awaits it, so a commit lands
   * between turns rather than inside one.
   *
   * Omitted, publishing is immediate -- correct for a headless or single-shot
   * composition where there is no live session whose cache could be lost.
   */
  readonly atTurnBoundary?: () => Promise<void>
}

function textOf(state: ArtifactState): string {
  // One file by convention, but tolerate a tree: a section is text, and a
  // reflector that split it across two files should not silently lose half.
  const entries = Object.entries(state).sort(([a], [b]) => a.localeCompare(b))
  const named = entries.find(([path]) => path === SECTION_FILE)
  if (named !== undefined) return named[1]
  return entries.map(([, body]) => body).join('\n\n')
}

/**
 * An {@link ArtifactAdapter} for one system-prompt section.
 *
 * `bind` scopes a candidate to one agent, where a same-named section shadows
 * the global one -- so a rollout measures the candidate prompt while every
 * other session keeps the committed one.
 */
export function promptArtifact(ctx: Context, options: PromptArtifactOptions): ArtifactAdapter {
  const { name } = options
  const id = options.id ?? `prompt:${name}`
  const order = options.order ?? EVOLVED_SECTION_ORDER
  let published: (() => void) | undefined

  return {
    id,
    blastRadius: options.blastRadius ?? 0.2,
    load: options.load,

    bind(scope: Context, state: ArtifactState): () => void {
      // A scoped section shadows the same name globally, which is exactly the
      // isolation a rollout needs: the child sees the candidate, nobody else
      // does, and no name has to be invented per candidate.
      return scope.systemPrompt.section({ name, order, text: textOf(state) })
    },

    async publish(state: ArtifactState): Promise<() => void> {
      // Wait first, then swap. Waiting after registering would already have
      // spent the cache this delay exists to protect.
      await options.atTurnBoundary?.()
      const previous = published
      // Dispose before registering here, unlike the skill adapters: a duplicate
      // section name throws rather than being resolved by rank, so the two
      // cannot briefly coexist. The gap is one synchronous statement, and it
      // sits at a turn boundary where no assembly is running.
      previous?.()
      const dispose = ctx.systemPrompt.section({ name, order, text: textOf(state) })
      published = dispose
      return () => {
        if (published === dispose) published = undefined
        dispose()
      }
    },
  }
}

/**
 * A `atTurnBoundary` that resolves when no agent is running.
 *
 * Resolves immediately when the harness is already idle, so a commit arriving
 * in a quiet moment is not delayed for the sake of symmetry.
 */
export function whenIdle(ctx: Context, isBusy: () => boolean): () => Promise<void> {
  return async () => {
    if (!isBusy()) return
    await new Promise<void>((resolve) => {
      const off = ctx.on('agent/status', (payload: any) => {
        if (payload?.status === 'idle' && !isBusy()) {
          off()
          resolve()
        }
      })
    })
  }
}
