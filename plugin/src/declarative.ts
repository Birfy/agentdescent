/**
 * Saying what to train on, and what counts as better, **in configuration**.
 *
 * Everything else in this package can be assembled by someone writing
 * TypeScript. That is the wrong bar for the two decisions that are genuinely
 * the user's: which tasks, and which objective. So both are declarable in the
 * profile's `cordis.patch.yml`, in the same three rungs the design describes --
 * a dataset with answers, a command that already exists, a rubric in prose.
 *
 * **Everything here validates at mount.** A misspelled objective kind must stop
 * the plugin loading with a message naming the row, not surface forty minutes
 * into a run as a scorer that was never registered -- or worse, as a run that
 * quietly used a different objective than the one intended.
 *
 * @module dsh-plugin-agentdescent/declarative
 */

import { readFileSync } from 'node:fs'

import type { EvolutionTask, Scorer, TaskSource } from './evolution.js'
import {
  all, commandScorer, efficiency, goldScorer, judgeScorer, replayPairwiseScorer,
  type GoldMatch, type Judge,
} from './scorers.js'

/** A dataset the user points at. */
export interface DatasetSpec {
  readonly name: string
  /** JSONL: `{"prompt": ..., "gold": ...}` per line; a bare line is a prompt. */
  readonly file: string
  /** Inline tasks, for a handful of cases not worth a file. */
  readonly tasks?: readonly EvolutionTask[]
}

/** What counts as better. `kind` picks the rung. */
export type ObjectiveSpec =
  | { readonly name?: string; readonly kind: 'gold'; readonly match?: GoldMatch
      readonly goldKey?: string; readonly tolerance?: number
      readonly efficiency?: boolean | { readonly stepBudget?: number } }
  | { readonly name?: string; readonly kind: 'command'; readonly run: readonly string[]
      readonly cwd?: string; readonly timeoutMs?: number; readonly ok?: readonly number[]
      readonly efficiency?: boolean | { readonly stepBudget?: number } }
  | { readonly name?: string; readonly kind: 'judge'; readonly rubric: string
      readonly efficiency?: boolean | { readonly stepBudget?: number } }
  | { readonly name?: string; readonly kind: 'replay'
      readonly efficiency?: boolean | { readonly stepBudget?: number } }
  | { readonly name?: string; readonly kind: 'all'; readonly of: readonly ObjectiveSpec[]
      readonly efficiency?: boolean | { readonly stepBudget?: number } }

const KINDS = ['gold', 'command', 'judge', 'replay', 'all'] as const

export class ConfigError extends Error {}

function fail(where: string, detail: string): never {
  throw new ConfigError(`${where}: ${detail}`)
}

/**
 * Read a JSONL dataset.
 *
 * Forgiving in exactly one way, and it is the way that matters: a line that is
 * not JSON is taken as a bare prompt, so a plain list of questions -- which is
 * what someone actually has when their objective is a command rather than an
 * answer -- is a valid dataset with no ceremony.
 */
export function readJsonl(file: string): EvolutionTask[] {
  let text: string
  try {
    text = readFileSync(file, 'utf8')
  } catch (error) {
    fail(`dataset "${file}"`, `cannot be read (${String(error)})`)
  }
  const tasks: EvolutionTask[] = []
  text.split(/\r?\n/).forEach((line, index) => {
    const trimmed = line.trim()
    if (trimmed.length === 0) return
    let row: any
    try {
      row = JSON.parse(trimmed)
    } catch {
      row = { prompt: trimmed }
    }
    if (typeof row !== 'object' || row === null || typeof row.prompt !== 'string') {
      fail(`${file}:${index + 1}`,
           'has no "prompt". Each line is JSON with a prompt key, or a bare prompt.')
    }
    const { prompt, id, ...meta } = row
    tasks.push({ id: typeof id === 'string' ? id : `t${tasks.length}`, prompt, meta })
  })
  if (tasks.length === 0) fail(`dataset "${file}"`, 'holds no tasks')
  return tasks
}

/** Turn a dataset declaration into a task source. */
export function buildTaskSource(spec: DatasetSpec): TaskSource {
  if (typeof spec.name !== 'string' || spec.name.length === 0) {
    fail('dataset', 'needs a name')
  }
  if (spec.tasks !== undefined && spec.tasks.length > 0) {
    const inline = spec.tasks
    return { name: spec.name, fetch: async (count) => inline.slice(0, count) }
  }
  if (typeof spec.file !== 'string' || spec.file.length === 0) {
    fail(`dataset "${spec.name}"`, 'needs either a file or inline tasks')
  }
  // Read at mount. A dataset that does not exist should stop the plugin
  // loading, not fail the first run someone kicks off a week later.
  const tasks = readJsonl(spec.file)
  return { name: spec.name, fetch: async (count) => tasks.slice(0, count) }
}

export interface ObjectiveDeps {
  /** Runs a completion for the `judge` rung, through the harness's own model. */
  readonly complete?: Judge
}

/**
 * Turn an objective declaration into a scorer.
 *
 * `efficiency: true` wraps whichever rung was chosen -- it multiplies, behind a
 * hard success gate, so it can never buy off a wrong answer. `kind: all` takes
 * the minimum of its parts, because a list of objectives is an "and".
 */
export function buildScorer(spec: ObjectiveSpec, deps: ObjectiveDeps = {}): Scorer {
  const where = `objective "${spec?.name ?? spec?.kind ?? '(unnamed)'}"`
  if (spec === null || typeof spec !== 'object') fail('objective', 'must be an object')
  if (!KINDS.includes(spec.kind as never)) {
    fail(where, `unknown kind "${String(spec.kind)}". One of: ${KINDS.join(', ')}`)
  }
  const inner = buildInner(spec, deps, where)
  const named = spec.name === undefined ? inner : { ...inner, name: spec.name }
  if (spec.efficiency === undefined || spec.efficiency === false) return named
  const options = spec.efficiency === true ? {} : spec.efficiency
  return efficiency(named, { name: named.name, ...options })
}

function buildInner(spec: ObjectiveSpec, deps: ObjectiveDeps, where: string): Scorer {
  switch (spec.kind) {
    case 'gold':
      return goldScorer({
        ...(spec.match === undefined ? {} : { match: spec.match }),
        ...(spec.goldKey === undefined ? {} : { goldKey: spec.goldKey }),
        ...(spec.tolerance === undefined ? {} : { tolerance: spec.tolerance }),
      })
    case 'command': {
      if (!Array.isArray(spec.run) || spec.run.length === 0) {
        fail(where, 'kind "command" needs `run` as a non-empty argv list, e.g. ' +
                    '["psql", "--quiet", "-f", "{output}"]')
      }
      return commandScorer({
        argv: spec.run,
        ...(spec.cwd === undefined ? {} : { cwd: spec.cwd }),
        ...(spec.timeoutMs === undefined ? {} : { timeoutMs: spec.timeoutMs }),
        ...(spec.ok === undefined ? {} : { ok: spec.ok }),
      })
    }
    case 'judge': {
      if (typeof spec.rubric !== 'string' || spec.rubric.trim().length === 0) {
        fail(where, 'kind "judge" needs a `rubric` saying what good looks like')
      }
      if (deps.complete === undefined) {
        // Without a model this rung silently cannot work, and a scorer that
        // always fails looks exactly like a model that cannot do the task.
        fail(where, 'kind "judge" needs a model, and none is available here')
      }
      return judgeScorer({ rubric: spec.rubric, judge: deps.complete })
    }
    case 'replay':
      return replayPairwiseScorer(
        deps.complete === undefined ? {} : { judge: deps.complete })
    case 'all': {
      if (!Array.isArray(spec.of) || spec.of.length === 0) {
        fail(where, 'kind "all" needs `of` as a non-empty list of objectives')
      }
      return all(spec.name ?? 'all', spec.of.map((part) => buildScorer(part, deps)))
    }
    default:
      fail(where, `unknown kind "${String((spec as { kind: string }).kind)}"`)
  }
}
