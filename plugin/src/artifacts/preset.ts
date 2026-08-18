/**
 * The `preset:<name>` artifact: a persona plus the tools an agent may use.
 *
 * A skill changes what an agent knows. A preset changes what it *is* -- how it
 * is told to behave, and which tools it can reach at all. That is why it sits at
 * L1 beside plugin source rather than at L2 beside skills: narrowing a tool set
 * is a capability change, and widening one is a capability change in the
 * direction nobody wants to make by accident.
 *
 * The state is two files, because they are two different kinds of thing and a
 * diff should be able to touch one without the other:
 *
 *     PERSONA.md   the standing instruction, a prompt section
 *     TOOLS.md     one tool name per line; `-name` denies, a bare name allows
 *
 * @module dsh-plugin-agentdescent/artifacts/preset
 */

import type { Context } from '@deepseek-ai/cordis'

import type { ArtifactAdapter, ArtifactId, ArtifactState } from '../evolution.js'
import { EVOLVED_SECTION_ORDER } from './prompt.js'

export const PERSONA_FILE = 'PERSONA.md'
export const TOOLS_FILE = 'TOOLS.md'

/** A parsed tool policy: what to keep, and what to remove. */
export interface ToolPolicy {
  readonly allow: string[]
  readonly deny: string[]
}

/**
 * Read `TOOLS.md`.
 *
 * A line is a tool name; `-name` denies it; `#` comments and blanks are
 * ignored. Deliberately a format a person can read and a model can edit without
 * a schema -- the alternative is YAML nobody proofreads.
 *
 * **An empty file means "no restriction", not "no tools".** The other reading
 * turns a reflector that trimmed the file into an agent that can do nothing,
 * which looks like a broken harness rather than a bad proposal.
 */
export function parseToolPolicy(text: string): ToolPolicy {
  const allow: string[] = []
  const deny: string[] = []
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    if (line.length === 0 || line.startsWith('#')) continue
    if (line.startsWith('-')) {
      const name = line.slice(1).trim()
      if (name.length > 0) deny.push(name)
    } else {
      allow.push(line)
    }
  }
  return { allow, deny }
}

export interface PresetArtifactOptions {
  readonly id?: ArtifactId
  readonly name: string
  load: () => Promise<ArtifactState>
  readonly order?: number
  /** L1 by default: a preset is harness, so a person approves it. */
  readonly blastRadius?: number
}

/**
 * An {@link ArtifactAdapter} for one agent preset.
 *
 * `bind` applies both halves to a single agent, which is what makes a preset
 * measurable at all: the candidate's persona and tool set apply to that rollout
 * and nowhere else.
 */
export function presetArtifact(ctx: Context, options: PresetArtifactOptions): ArtifactAdapter {
  const { name } = options
  const id = options.id ?? `preset:${name}`
  const order = options.order ?? EVOLVED_SECTION_ORDER
  let published: Array<() => void> = []
  let committed: ArtifactState | undefined

  const applyTo = (scope: Context, state: ArtifactState): (() => void) => {
    const disposers: Array<() => void> = []
    const persona = state[PERSONA_FILE]
    if (persona !== undefined && persona.trim().length > 0) {
      disposers.push(scope.systemPrompt.section({
        name: `preset:${name}`, order, text: persona,
      }))
    }
    const tools = state[TOOLS_FILE]
    if (tools !== undefined) {
      const policy = parseToolPolicy(tools)
      // No clauses means no restriction. Registering an empty `allow` would
      // mask every tool, and an agent that can do nothing reads as a broken
      // harness rather than as a bad candidate.
      if (policy.allow.length > 0 || policy.deny.length > 0) {
        try {
          disposers.push(scope.tools.restrict({
            ...(policy.allow.length > 0 ? { allow: policy.allow } : {}),
            ...(policy.deny.length > 0 ? { deny: policy.deny } : {}),
          }))
        } catch (error) {
          // dsh refuses an unscoped restriction, and it is right to: a global
          // mask hits every agent in the process. Say which context was needed,
          // because the raw message does not know it was a preset asking.
          for (const dispose of disposers.reverse()) dispose()
          throw new Error(
            `${id} cannot apply its tool policy here: a preset's tool mask is ` +
            `per-agent, so it needs that agent's own context (agent.ctx). ` +
            `Underlying: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    }
    return () => { for (const dispose of disposers.reverse()) dispose() }
  }

  return {
    id,
    // A preset decides what an agent may do, so it is oracle-gated and waits
    // for a person -- the same layer as a plugin's source.
    blastRadius: options.blastRadius ?? 0.6,
    load: options.load,

    bind(scope: Context, state: ArtifactState): () => void {
      return applyTo(scope, state)
    },

    /**
     * Commit the preset.
     *
     * The persona lands globally, which is what a deployment persona is. The
     * **tool mask does not**, and that is dsh's rule rather than a shortcut of
     * ours: `tools.restrict()` refuses an unscoped context outright, because a
     * global restriction would mask every agent in the process -- including the
     * ones not using this preset at all. So the committed policy is stored and
     * applied per agent through {@link PresetArtifact.applyTo}, which an agent
     * preset's own composition wires under its scope.
     *
     * The honest consequence, stated because it is easy to assume otherwise: a
     * published preset changes the *persona* immediately and changes the tool
     * set only for agents composed with it.
     */
    async publish(state: ArtifactState): Promise<() => void> {
      for (const dispose of published.reverse()) dispose()
      published = []
      committed = state
      const persona = state[PERSONA_FILE]
      if (persona !== undefined && persona.trim().length > 0) {
        published = [ctx.systemPrompt.section({
          name: `preset:${name}`, order, text: persona,
        })]
      }
      const disposers = published.slice()
      return () => {
        for (const dispose of disposers) dispose()
        published = published.filter((entry) => !disposers.includes(entry))
      }
    },

    /** The committed state, for a composition that applies it to its agents. */
    committedState: () => committed,
    /** Apply a preset to one agent's scope. The tool half only works here. */
    applyTo,
  } as ArtifactAdapter & {
    committedState(): ArtifactState | undefined
    applyTo(scope: Context, state: ArtifactState): () => void
  }
}
