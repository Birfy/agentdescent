/**
 * The shipped scorers: the objective ladder, on the harness side.
 *
 * Self-evolution does not mean "no labels". Most people have an objective, it
 * just is not a benchmark, so these are ordered by how strong a signal the user
 * actually brought:
 *
 *   A  `gold`             they have expected answers
 *   B  `command`          they have a check that already exists
 *   B  `judge`            they can describe what good looks like
 *   C  `replayPairwise`   they have neither, but the log has what happened before
 *
 * `efficiency` is not a rung. It multiplies whichever rung is in use, behind a
 * hard success gate, so "fewer steps" can never buy off "got it wrong".
 *
 * @module dsh-plugin-agentdescent/scorers
 */

import { spawn } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { EvolutionTask, RolloutResult, Scorer } from './evolution.js'

/** How a gold answer is compared. */
export type GoldMatch = 'exact' | 'contains' | 'lastNumber'

const NUMBER = /-?\d[\d,]*\.?\d*/g

function normalise(text: string): string {
  return text.trim().replace(/^[.!?"'\s]+|[.!?"'\s]+$/g, '').replace(/\s+/g, ' ').toLowerCase()
}

function lastNumber(text: string): number | undefined {
  const found = text.match(NUMBER)
  if (found === null || found.length === 0) return undefined
  const parsed = Number(found[found.length - 1]?.replace(/,/g, ''))
  return Number.isFinite(parsed) ? parsed : undefined
}

export interface GoldOptions {
  readonly name?: string
  readonly match?: GoldMatch
  /** Where in `task.meta` the expected answer lives. */
  readonly goldKey?: string
  readonly tolerance?: number
}

/**
 * Rung A: score against the expected answer on each task.
 *
 * `lastNumber` reads the *gold* the same way it reads the output, which matters
 * more than it sounds: a dataset's answer column is often a whole worked
 * solution ending in the figure, and parsing it as a bare number fails silently
 * -- every item scores 0 and it reads as a hopeless model rather than a scorer
 * pointed at the wrong thing.
 */
export function goldScorer(options: GoldOptions = {}): Scorer {
  const match = options.match ?? 'contains'
  const goldKey = options.goldKey ?? 'gold'
  const tolerance = options.tolerance ?? 0
  return {
    name: options.name ?? 'gold',
    score: async (task: EvolutionTask, run: RolloutResult) => {
      const raw = task.meta?.[goldKey]
      if (raw === undefined || raw === null) {
        throw new Error(
          `task "${task.id}" has no meta.${goldKey}. The gold scorer needs an ` +
          'expected answer; use `command` or `judge` when there is not one.')
      }
      const want = String(raw)
      const got = run.output
      if (match === 'lastNumber') {
        const expected = lastNumber(want)
        if (expected === undefined) {
          throw new Error(
            `task "${task.id}": meta.${goldKey} holds no number, so lastNumber ` +
            'can never match it. Use exact or contains instead.')
        }
        const actual = lastNumber(got)
        if (actual === undefined) return 0
        return Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected))
          ? 1 : 0
      }
      const a = normalise(want)
      const b = normalise(got)
      if (match === 'exact') return a === b ? 1 : 0
      return a.length > 0 && b.includes(a) ? 1 : 0
    },
  }
}

export interface CommandOptions {
  readonly name?: string
  /** Argv. `{output}` in any argument becomes a path to the rollout's output. */
  readonly argv: readonly string[]
  readonly cwd?: string
  readonly timeoutMs?: number
  /** Exit codes that count as success. */
  readonly ok?: readonly number[]
}

/**
 * Rung B: the user's own command is the judge.
 *
 * The rung reachable without writing code, because the check usually exists
 * already: it compiles, it runs against my database, it passes these tests.
 *
 * A timeout scores 0 rather than throwing -- a check that hangs on this output
 * *is* a failing check, and throwing would drop the sample instead of scoring
 * it, biasing the run toward candidates that happen not to hang the checker.
 * A missing executable throws, because that is a configuration mistake and
 * scoring it 0 makes every rollout fail identically, which is indistinguishable
 * from a model that cannot do the task.
 */
export function commandScorer(options: CommandOptions): Scorer {
  if (options.argv.length === 0) throw new Error('commandScorer needs a command to run')
  const ok = options.ok ?? [0]
  const timeoutMs = options.timeoutMs ?? 30_000
  return {
    name: options.name ?? 'command',
    score: async (_task: EvolutionTask, run: RolloutResult) => {
      const scratch = mkdtempSync(join(tmpdir(), 'dsh-evolve-check-'))
      const outputPath = join(scratch, 'output.txt')
      writeFileSync(outputPath, run.output, 'utf8')
      const argv = options.argv.map((arg) => arg.replace('{output}', outputPath))
      try {
        return await new Promise<number>((resolve, reject) => {
          const child = spawn(argv[0] as string, argv.slice(1), {
            ...(options.cwd === undefined ? {} : { cwd: options.cwd }),
            stdio: ['pipe', 'ignore', 'ignore'],
          })
          const timer = setTimeout(() => { child.kill('SIGKILL'); resolve(0) }, timeoutMs)
          timer.unref?.()
          child.on('error', (error: NodeJS.ErrnoException) => {
            clearTimeout(timer)
            reject(error.code === 'ENOENT'
              ? new Error(
                  `commandScorer cannot run "${String(argv[0])}": not found. Every ` +
                  'rollout would score 0, which is indistinguishable from a model ' +
                  'that cannot do the task -- so this throws instead.')
              : error)
          })
          child.on('close', (code) => {
            clearTimeout(timer)
            resolve(ok.includes(code ?? -1) ? 1 : 0)
          })
          child.stdin?.end(run.output)
        })
      } finally {
        rmSync(scratch, { recursive: true, force: true })
      }
    },
  }
}

/** Turns a prompt plus the trajectory into a verdict in [0, 1]. */
export type Judge = (prompt: string) => Promise<string>

export interface JudgeOptions {
  readonly name?: string
  readonly rubric: string
  readonly judge: Judge
}

/**
 * Rung B: a rubric, judged by a model.
 *
 * Absolute scoring is the least reliable thing an LLM judge does, so this asks
 * for a verdict against explicit criteria rather than a number out of ten, and
 * anything it cannot parse scores 0 -- an unreadable verdict is not a pass.
 */
export function judgeScorer(options: JudgeOptions): Scorer {
  return {
    name: options.name ?? 'judge',
    score: async (task: EvolutionTask, run: RolloutResult) => {
      const verdict = await options.judge(
        `Judge one answer against the criteria. Reply with exactly PASS or FAIL ` +
        `on the first line, then one sentence of reason.\n\n` +
        `CRITERIA:\n${options.rubric}\n\nTASK:\n${task.prompt}\n\nANSWER:\n${run.output}\n`)
      return /^\s*PASS\b/i.test(verdict) ? 1 : 0
    },
  }
}

export interface ReplayOptions {
  readonly name?: string
  /** Consulted only when the objective comparison is a tie. */
  readonly judge?: Judge
}

/**
 * Rung C: compare against what actually happened last time.
 *
 * The fallback, and the only rung that needs a reference. dsh guarantees that
 * model-visible means logged, so a past turn is a free reference trajectory --
 * and comparing two trajectories is the one thing an LLM judge is reliable at,
 * unlike scoring one in isolation.
 *
 * Objective signals decide first and cost nothing: a turn that ended in error
 * loses to one that completed, and among turns that both completed, fewer tool
 * errors wins. Only a genuine tie reaches the judge, and without one a tie is
 * 0.5 -- "no evidence either way", which is what it is.
 */
export function replayPairwiseScorer(options: ReplayOptions = {}): Scorer {
  return {
    name: options.name ?? 'replay-pairwise',
    score: async (task: EvolutionTask, run: RolloutResult, reference?: RolloutResult) => {
      if (reference === undefined) {
        // No reference is not a failure of the candidate; it is a missing
        // measurement, and scoring it 0 would punish the artifact for that.
        return run.finishReason === 'error' ? 0 : 0.5
      }
      const candidateBroke = run.finishReason === 'error'
      const referenceBroke = reference.finishReason === 'error'
      if (candidateBroke !== referenceBroke) return candidateBroke ? 0 : 1
      const errors = (run.toolErrors ?? 0) - (reference.toolErrors ?? 0)
      if (errors !== 0) return errors < 0 ? 1 : 0
      if (options.judge === undefined) return 0.5
      const verdict = await options.judge(
        `Two answers to the same task. Reply with exactly A or B on the first ` +
        `line: which is better?\n\nTASK:\n${task.prompt}\n\n` +
        `A (previous):\n${reference.output}\n\nB (candidate):\n${run.output}\n`)
      if (/^\s*B\b/i.test(verdict)) return 1
      if (/^\s*A\b/i.test(verdict)) return 0
      return 0.5
    },
  }
}

export interface EfficiencyOptions {
  readonly name?: string
  /** Steps at or above which the efficiency term is fully spent. */
  readonly stepBudget?: number
  /** Below this the wrapped score counts as a failure and efficiency is not applied. */
  readonly passAt?: number
}

/**
 * Multiply a scorer by how cheaply it succeeded. **Never a weighted sum.**
 *
 * A sum lets "two fewer steps" buy off "got it wrong", which is the one way an
 * efficiency term can quietly ruin a run: the aggregator starts accepting
 * artifacts that are faster and worse, and the held-out score still goes up.
 * So the gate is hard -- fail and the reward is 0 -- and efficiency only moves
 * a passing result within the upper half of the range.
 */
export function efficiency(inner: Scorer, options: EfficiencyOptions = {}): Scorer {
  const budget = options.stepBudget ?? 10
  const passAt = options.passAt ?? 1
  return {
    name: options.name ?? `${inner.name}+efficiency`,
    score: async (task, run, reference) => {
      const base = await inner.score(task, run, reference)
      if (base < passAt) return 0
      const spent = Math.min(1, Math.max(0, (run.steps ?? 0) / budget))
      return 0.5 + 0.5 * (1 - spent)
    },
  }
}

/**
 * Every objective must hold: the score is the **minimum**, not the mean.
 *
 * An average lets a candidate that nails one objective and fails another look
 * like a partial success, when what the user said was "and", not "or".
 */
export function all(name: string, scorers: readonly Scorer[]): Scorer {
  if (scorers.length === 0) throw new Error('all() needs at least one scorer')
  return {
    name,
    score: async (task, run, reference) => {
      let worst = 1
      for (const scorer of scorers) {
        worst = Math.min(worst, await scorer.score(task, run, reference))
        if (worst === 0) return 0                 // nothing can rescue a failed gate
      }
      return worst
    },
  }
}
