/**
 * The `skill:<name>` artifact: an evolvable skill served straight from the ledger.
 *
 * This is the adapter that makes evolution *visible*. A committed state is
 * registered on `ctx.skills` as a provider, so the next `snapshot()` serves the
 * new body -- the harness changes under the user without a restart, and without
 * waiting for a file watcher to notice a write.
 *
 * @module dsh-plugin-agentdescent/artifacts/skill
 */

import type { Context } from '@deepseek-ai/cordis'
import type {
  SkillCandidate,
  SkillDefinition,
  SkillLookupOptions,
  SkillProvider,
} from '@deepseek-ai/dsh-skill'

import type { ArtifactAdapter, ArtifactId, ArtifactState } from '../evolution.js'

/** The file whose body is the skill. Everything else in the tree is a resource. */
export const ENTRYPOINT = 'SKILL.md'

/**
 * Rank for a published evolved skill.
 *
 * It has to beat the filesystem provider's nearest root (`project-dsh`, rank
 * 100), because the evolved body *came from* that file and the point of
 * publishing is that the harness now serves the improved one. Losing the tie
 * would leave the ledger and the running harness disagreeing with no error
 * anywhere -- the run would report an accepted commit while every later rollout
 * still read the old body.
 *
 * Rank only decides duplicates *within* one layer, which is what this is: both
 * this provider and the filesystem one are host-level registrations. A
 * rollout's candidate does not rely on rank at all -- it is registered against
 * one agent's own scope, and a nearer layer wins a name outright.
 */
export const EVOLVED_SKILL_RANK = 50

const FRONTMATTER = /^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/
const FIELD = (key: string): RegExp => new RegExp(`^${key}:[ \\t]*(.+?)[ \\t]*$`, 'm')

/** What the model is told a skill is for, when the skill does not say. */
const FALLBACK_DESCRIPTION = 'An evolved skill.'

interface Parsed {
  readonly description: string
  readonly whenToUse?: string
  readonly content: string
}

/**
 * Split frontmatter off a skill body and read the routing fields.
 *
 * A regex, not a YAML parser: the fields that matter here are one-line scalars,
 * and a block scalar declines to answer rather than yielding `>` as the
 * description. Nothing downstream is improved by a half-understood value.
 */
export function parseSkill(body: string): Parsed {
  const matched = FRONTMATTER.exec(body)
  if (matched === null) {
    const firstLine = body.split(/\r?\n/).find((line) => line.trim().length > 0)
    return {
      description: firstLine?.replace(/^#+\s*/, '').trim() || FALLBACK_DESCRIPTION,
      content: body,
    }
  }
  const block = matched[1] ?? ''
  const content = body.slice(matched[0].length)
  const scalar = (key: string): string | undefined => {
    const found = FIELD(key).exec(block)
    const value = found?.[1]?.trim()
    if (value === undefined || value.length === 0) return undefined
    if (value.startsWith('|') || value.startsWith('>')) return undefined
    return value.replace(/^['"]|['"]$/g, '') || undefined
  }
  const whenToUse = scalar('when_to_use') ?? scalar('whenToUse')
  return {
    description: scalar('description') ?? FALLBACK_DESCRIPTION,
    ...(whenToUse === undefined ? {} : { whenToUse }),
    content,
  }
}

/** Build the one-skill provider that serves a state. */
function providerFor(name: string, state: ArtifactState, rank: number,
                     providerName: string): SkillProvider {
  const body = state[ENTRYPOINT] ?? ''
  const parsed = parseSkill(body)
  const summary = {
    name,
    description: parsed.description,
    ...(parsed.whenToUse === undefined ? {} : { whenToUse: parsed.whenToUse }),
    invocation: { modelInvocable: true, userInvocable: true },
    source: 'custom' as const,
    provider: providerName,
  }
  const candidate: SkillCandidate = { ...summary, rank, locator: name }
  const definition: SkillDefinition = { ...summary, content: parsed.content }
  return {
    name: providerName,
    list: async (_options: SkillLookupOptions) => [candidate],
    get: async (selected: SkillCandidate) =>
      (selected.locator === name ? definition : undefined),
  }
}

export interface SkillArtifactOptions {
  /** Artifact id, e.g. `skill:sql`. Defaults to `skill:<name>`. */
  readonly id?: ArtifactId
  /** The skill's kebab-case name, as the harness addresses it. */
  readonly name: string
  /** The current state, normally the on-disk tree the ledger started from. */
  load: () => Promise<ArtifactState>
  /** Skills are local: merged on held-out score, no oracle. */
  readonly blastRadius?: number
  readonly rank?: number
}

/**
 * An {@link ArtifactAdapter} for one skill.
 *
 * `bind` and `publish` differ only in *where* they register, which is the whole
 * reason rollouts can share one process:
 *
 * - `bind(agent.ctx, candidate)` registers into that agent's scope layer, and a
 *   nearer layer wins a name outright -- so eight workers can each see their
 *   own candidate concurrently, with no containers and no second harness.
 * - `publish(state)` registers globally at a rank that beats the file on disk,
 *   and replaces any previous publication so the registry emits `skills/change`
 *   and consumers refetch.
 */
export function skillArtifact(ctx: Context, options: SkillArtifactOptions): ArtifactAdapter {
  const { name } = options
  const id = options.id ?? `skill:${name}`
  const rank = options.rank ?? EVOLVED_SKILL_RANK
  let published: (() => void) | undefined

  return {
    id,
    blastRadius: options.blastRadius ?? 0.2,
    load: options.load,

    bind(scope: Context, state: ArtifactState): () => void {
      // Unique per binding: two candidates alive at once would otherwise
      // collide on the provider name, and duplicate names in one layer throw.
      const providerName = `${id}#candidate-${bindingSeq += 1}`
      return scope.skills.registerProvider(() => providerFor(name, state, rank, providerName))
    },

    async publish(state: ArtifactState): Promise<() => void> {
      const previous = published
      // Unique per publication, not a fixed `#head`: the new provider is
      // registered before the old is disposed, so for that window both exist
      // and a shared name would throw on the second publish.
      const dispose = ctx.skills.registerProvider(
        () => providerFor(name, state, rank, `${id}#head-${bindingSeq += 1}`))
      // Dispose *after* registering. Between the two the registry briefly holds
      // both, and the new one wins on registration order, so no lookup in that
      // window can see the skill missing.
      previous?.()
      published = dispose
      return () => {
        if (published === dispose) published = undefined
        dispose()
      }
    },
  }
}

let bindingSeq = 0
