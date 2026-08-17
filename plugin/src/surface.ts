/**
 * The Consumer role: `/evolve` for people, `evolve_*` for the model.
 *
 * Both go through `ctx.evolution` and neither knows an optimiser exists, which
 * is the point of the seam -- swapping the engine leaves these untouched.
 *
 * @module dsh-plugin-agentdescent/surface
 */

import type { Context } from '@deepseek-ai/cordis'
import type { CommandResult } from '@deepseek-ai/dsh-commands'
import { defineTool } from '@deepseek-ai/dsh-tools'

import type { EvolutionRegistry, EvolutionSpec, RunStatus } from './evolution.js'

/**
 * What `/evolve` needs before it can start anything.
 *
 * Asking is the whole design of the objective ladder: a user with a dataset or
 * a command should never be silently dropped onto the weakest signal. So a run
 * with no scorer named refuses when the choice is ambiguous, rather than
 * picking one and reporting a number nobody can interpret.
 */
export function describeChoices(evolution: EvolutionRegistry): string {
  const lines = [
    `artifacts   : ${format(evolution.listArtifacts())}`,
    `scorers     : ${format(evolution.listScorers())}`,
    `task sources: ${format(evolution.listTaskSources())}`,
  ]
  return lines.join('\n')
}

function format(names: readonly string[]): string {
  return names.length > 0 ? names.join(', ') : '(none registered)'
}

function renderStatus(status: RunStatus): string {
  const budget = status.tokensSpent > 0 ? `, ${status.tokensSpent} tokens` : ''
  return `${status.runId} ${status.phase} — round ${status.round}, ` +
    `${status.accepted} accepted / ${status.rejected} rejected${budget}` +
    (status.error === undefined ? '' : `\nerror: ${status.error}`)
}

/**
 * Resolve `/evolve <artifact> [scorer] [task-source]` into a spec.
 *
 * Single registrations resolve themselves; anything ambiguous is an error that
 * lists the choices. A wrong objective produces a confidently wrong artifact
 * and there is no later stage that catches it.
 */
export function parseEvolveInput(evolution: EvolutionRegistry, raw: string): EvolutionSpec {
  const words = raw.trim().split(/\s+/).filter((word) => word.length > 0)
  const artifact = words[0]
  if (artifact === undefined) {
    throw new Error(`/evolve needs an artifact.\n${describeChoices(evolution)}`)
  }
  if (evolution.artifact(artifact) === undefined) {
    throw new Error(`unknown artifact "${artifact}".\n${describeChoices(evolution)}`)
  }
  return {
    artifact,
    scorer: pick(evolution, 'scorer', words[1], evolution.listScorers()),
    taskSource: pick(evolution, 'task source', words[2], evolution.listTaskSources()),
  }
}

function pick(evolution: EvolutionRegistry, kind: string, given: string | undefined,
              known: readonly string[]): string {
  if (given !== undefined) return given
  const only = known.length === 1 ? known[0] : undefined
  if (only !== undefined) return only
  throw new Error(
    `/evolve needs a ${kind}: ${known.length} are registered ` +
    `(${format(known)}). Naming the wrong one produces a confidently wrong ` +
    `artifact, so this will not choose for you.`)
}

/** Register `/evolve`. Requires `ctx.commands` and `ctx.evolution`. */
export function registerCommand(ctx: Context): () => void {
  const { evolution } = ctx
  return ctx.commands.register({
    name: 'evolve',
    description: 'Improve one of this harness\'s own artifacts against an objective',
    input: { hint: '<artifact> [scorer] [task-source] | status <run> | cancel <run> | list' },
    handler: async (invocation): Promise<CommandResult> => {
      const raw = invocation.rawInput.trim()
      const [verb, ...rest] = raw.split(/\s+/)
      try {
        if (verb === undefined || verb === '' || verb === 'list') {
          return { kind: 'success', text: describeChoices(evolution) }
        }
        if (verb === 'status') {
          const runId = rest[0]
          if (runId === undefined) return { kind: 'error', text: '/evolve status <run>' }
          const status = evolution.status(runId)
          return status === undefined
            ? { kind: 'error', text: `no run "${runId}"` }
            : { kind: 'success', text: renderStatus(status) }
        }
        if (verb === 'cancel') {
          const runId = rest[0]
          if (runId === undefined) return { kind: 'error', text: '/evolve cancel <run>' }
          await evolution.cancel(runId, 'cancelled from /evolve')
          return { kind: 'success', text: `cancelling ${runId}` }
        }
        const spec = parseEvolveInput(evolution, raw)
        const runId = await evolution.start(spec)
        return {
          kind: 'success',
          text: `evolving ${spec.artifact} — scorer ${spec.scorer}, tasks from ` +
                `${spec.taskSource}. Run ${runId}; /evolve status ${runId}`,
        }
      } catch (error) {
        return { kind: 'error', text: error instanceof Error ? error.message : String(error) }
      }
    },
  })
}

/**
 * Register the model-facing tools.
 *
 * The model starting its own training run is not a gimmick: it is the one
 * participant that knows it keeps getting a class of task wrong, and it knows
 * before any human notices.
 */
export function registerTools(ctx: Context): () => void {
  const { evolution } = ctx
  return ctx.effect(function* () {
    yield ctx.tools.register(defineTool({
      name: 'evolve_start',
      description:
        'Start improving one of this harness\'s own artifacts (a skill, a prompt) ' +
        'against a scorer. Use when a class of task keeps going wrong and the ' +
        'artifact responsible could be improved. Returns a run id.',
      parameters: {
        artifact: { type: 'string', required: true, description: 'e.g. skill:sql' },
        scorer: { type: 'string', description: 'how candidates are judged' },
        task_source: { type: 'string', description: 'where tasks come from' },
      },
      output: {
        schema: { type: 'string' },
        render: (_args, value) => [{ type: 'text', text: value }],
      },
      async execute(args) {
        const spec: EvolutionSpec = {
          artifact: args.artifact,
          scorer: args.scorer ?? pick(evolution, 'scorer', undefined, evolution.listScorers()),
          taskSource: args.task_source
            ?? pick(evolution, 'task source', undefined, evolution.listTaskSources()),
        }
        return await evolution.start(spec)
      },
    }))

    yield ctx.tools.register(defineTool({
      name: 'evolve_status',
      description: 'Check an evolution run started with evolve_start.',
      parameters: { run_id: { type: 'string', required: true, description: 'the run id' } },
      output: {
        schema: { type: 'string' },
        render: (_args, value) => [{ type: 'text', text: value }],
      },
      async execute(args) {
        const status = evolution.status(args.run_id)
        return status === undefined ? `no run "${args.run_id}"` : renderStatus(status)
      },
    }))
  }, 'evolution.tools')
}
