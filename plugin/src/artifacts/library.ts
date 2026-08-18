/**
 * The `skills:<root>` artifact: a whole skill **library** as one evolvable tree.
 *
 * `skill:<name>` refines one skill someone already decided to have. This is the
 * other half, and the more interesting one: the artifact is the root, so a run
 * can *add* a skill nobody wrote, drop one that never helped, and move a lesson
 * from one skill into another -- discovery rather than refinement.
 *
 * The state is the root's whole tree, keyed by path (`sql/SKILL.md`,
 * `sql/references/style.md`, ...). Path-keyed state is what gives the aggregator
 * its semantics for free here: two workers adding different skills are
 * complementary and fuse, two rewriting the same skill contradict and are
 * resolved on held-out score.
 *
 * @module dsh-plugin-agentdescent/artifacts/library
 */

import type { Context } from '@deepseek-ai/cordis'
import type {
  SkillCandidate,
  SkillDefinition,
  SkillLookupOptions,
  SkillProvider,
} from '@deepseek-ai/dsh-skill'

import type { ArtifactAdapter, ArtifactId, ArtifactState } from '../evolution.js'
import { ENTRYPOINT, EVOLVED_SKILL_RANK, parseSkill } from './skill.js'

/** One skill found inside a library state. */
export interface LibrarySkill {
  readonly name: string
  readonly body: string
}

/**
 * Split a library tree into the skills it holds.
 *
 * A skill is a top-level directory with a `SKILL.md`. Everything else in the
 * tree is a resource belonging to one of them, and a stray file at the root is
 * neither -- silently promoting it to a skill would let a README become part of
 * the model's catalogue.
 */
export function parseLibrary(state: ArtifactState): LibrarySkill[] {
  const skills: LibrarySkill[] = []
  for (const [path, body] of Object.entries(state)) {
    const parts = path.split('/')
    if (parts.length !== 2 || parts[1] !== ENTRYPOINT) continue
    const name = parts[0]
    if (name === undefined || name.length === 0 || name.startsWith('.')) continue
    skills.push({ name, body })
  }
  return skills.sort((a, b) => a.name.localeCompare(b.name))
}

/** Build one provider serving every skill in a library state. */
function providerFor(state: ArtifactState, rank: number, providerName: string): SkillProvider {
  const definitions = new Map<string, SkillDefinition>()
  const candidates: SkillCandidate[] = []
  for (const skill of parseLibrary(state)) {
    const parsed = parseSkill(skill.body)
    const summary = {
      name: skill.name,
      description: parsed.description,
      ...(parsed.whenToUse === undefined ? {} : { whenToUse: parsed.whenToUse }),
      invocation: { modelInvocable: true, userInvocable: true },
      source: 'custom' as const,
      provider: providerName,
    }
    candidates.push({ ...summary, rank, locator: skill.name })
    definitions.set(skill.name, { ...summary, content: parsed.content })
  }
  return {
    name: providerName,
    list: async (_options: SkillLookupOptions) => candidates,
    get: async (selected: SkillCandidate) => definitions.get(String(selected.locator)),
  }
}

export interface SkillLibraryOptions {
  /** Defaults to `skills:<name>`. */
  readonly id?: ArtifactId
  /** A label for the library, usually the root directory's name. */
  readonly name: string
  /** The current tree, normally read from the root on disk. */
  load: () => Promise<ArtifactState>
  /** Skills stay local, so the library merges on held-out score. */
  readonly blastRadius?: number
  readonly rank?: number
}

let seq = 0

/**
 * An {@link ArtifactAdapter} for a whole skill library.
 *
 * **One provider serves the entire set, and that is the design.** The obvious
 * alternative -- a provider per skill -- makes additions work and deletions
 * silently not: a skill dropped from the state would keep being served by the
 * registration nobody disposed, so the model would still see a skill the ledger
 * says was removed, and the run's held-out score would not reflect what the
 * harness does. Replacing one provider makes the served set exactly the
 * committed set, additions and deletions alike.
 */
export function skillLibraryArtifact(ctx: Context,
                                     options: SkillLibraryOptions): ArtifactAdapter {
  const id = options.id ?? `skills:${options.name}`
  const rank = options.rank ?? EVOLVED_SKILL_RANK
  let published: (() => void) | undefined

  return {
    id,
    blastRadius: options.blastRadius ?? 0.2,
    load: options.load,

    bind(scope: Context, state: ArtifactState): () => void {
      return scope.skills.registerProvider(
        () => providerFor(state, rank, `${id}#candidate-${(seq += 1)}`))
    },

    async publish(state: ArtifactState): Promise<() => void> {
      const previous = published
      const dispose = ctx.skills.registerProvider(
        () => providerFor(state, rank, `${id}#head-${(seq += 1)}`))
      // Register before disposing, so no lookup in the gap sees an empty
      // library; unique names because both exist for that instant.
      previous?.()
      published = dispose
      return () => {
        if (published === dispose) published = undefined
        dispose()
      }
    },
  }
}
