/**
 * Declaring a dataset and an objective in configuration.
 *
 * These are the two decisions that are genuinely the user's, so they must not
 * require writing TypeScript. And because they are config, every mistake has to
 * fail at mount: a scorer that was never registered, or -- worse -- one that
 * quietly measured something other than what was meant, is not something a run
 * can tell you about afterwards.
 */

import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  buildScorer, buildTaskSource, ConfigError, readJsonl, type ObjectiveSpec,
} from '../src/declarative.js'
import type { EvolutionTask, RolloutResult } from '../src/evolution.js'

const TASK: EvolutionTask = { id: 't0', prompt: 'total it', meta: { gold: '42' } }
const ran = (output: string, extra: Partial<RolloutResult> = {}): RolloutResult =>
  ({ output, finishReason: 'completed', steps: 1, toolErrors: 0, ...extra })

function jsonl(lines: string[]): string {
  const file = join(mkdtempSync(join(tmpdir(), 'dsh-data-')), 'tasks.jsonl')
  writeFileSync(file, lines.join('\n'), 'utf8')
  return file
}

describe('declaring a dataset', () => {
  it('reads prompts and answers from JSONL', async () => {
    const file = jsonl([
      JSON.stringify({ prompt: 'q1', gold: 'a1' }),
      JSON.stringify({ prompt: 'q2', gold: 'a2' }),
    ])
    const tasks = await buildTaskSource({ name: 'mine', file }).fetch(10)
    expect(tasks.map((t) => t.prompt)).toEqual(['q1', 'q2'])
    expect(tasks[0]?.meta?.gold).toBe('a1')
  })

  it('takes a bare line as a prompt', async () => {
    // What someone actually has when their objective is a command rather than
    // an answer: a list of questions, with no ceremony.
    const tasks = await buildTaskSource({ name: 'mine', file: jsonl(['just ask this', '']) })
      .fetch(10)
    expect(tasks.map((t) => t.prompt)).toEqual(['just ask this'])
  })

  it('accepts a handful of inline tasks without a file', async () => {
    const source = buildTaskSource({
      name: 'inline', file: '',
      tasks: [{ id: 'a', prompt: 'one' }, { id: 'b', prompt: 'two' }],
    })
    expect((await source.fetch(1)).map((t) => t.prompt)).toEqual(['one'])
  })

  it('fails at mount for a file that is not there', () => {
    // Not on the first run someone kicks off a week later.
    expect(() => buildTaskSource({ name: 'mine', file: '/no/such/file.jsonl' }))
      .toThrow(ConfigError)
  })

  it('names the line that is malformed', () => {
    expect(() => readJsonl(jsonl([JSON.stringify({ prompt: 'ok' }), JSON.stringify({ gold: 'x' })])))
      .toThrow(/:2\b/)
  })

  it('refuses an empty dataset rather than a run that measures nothing', () => {
    expect(() => buildTaskSource({ name: 'mine', file: jsonl([]) })).toThrow(/no tasks/)
  })

  it('needs a name, and needs some tasks from somewhere', () => {
    expect(() => buildTaskSource({ name: '', file: 'x' })).toThrow(/needs a name/)
    expect(() => buildTaskSource({ name: 'mine', file: '' })).toThrow(/file or inline/)
  })
})

describe('declaring an objective', () => {
  it('rung A: an expected answer', async () => {
    const scorer = buildScorer({ kind: 'gold' })
    expect(await scorer.score(TASK, ran('the answer is 42'))).toBe(1)
  })

  it('rung A: the comparison is selectable', async () => {
    const scorer = buildScorer({ kind: 'gold', match: 'exact' })
    expect(await scorer.score(TASK, ran('the answer is 42'))).toBe(0)
  })

  it('rung B: a command that already exists', async () => {
    const scorer = buildScorer({ kind: 'command', run: ['node', '-e', 'process.exit(0)'] })
    expect(await scorer.score(TASK, ran('anything'))).toBe(1)
  })

  it('rung B: a rubric, judged by the harness model', async () => {
    const scorer = buildScorer({ kind: 'judge', rubric: 'snake_case only' },
                               { complete: async () => 'PASS' })
    expect(await scorer.score(TASK, ran('x'))).toBe(1)
  })

  it('rung C: comparing against what happened before', async () => {
    const scorer = buildScorer({ kind: 'replay' })
    expect(await scorer.score(TASK, ran('ok'), ran('x', { finishReason: 'error' }))).toBe(1)
  })

  it('stacks objectives with "and", not with an average', async () => {
    const spec: ObjectiveSpec = {
      name: 'correct-and-clean', kind: 'all',
      of: [{ kind: 'gold' }, { kind: 'command', run: ['node', '-e', 'process.exit(1)'] }],
    }
    expect(await buildScorer(spec).score(TASK, ran('42'))).toBe(0)
  })

  it('wraps any rung in efficiency, which multiplies rather than sums', async () => {
    const scorer = buildScorer({ kind: 'gold', efficiency: { stepBudget: 10 } })
    expect(await scorer.score(TASK, ran('41', { steps: 1 }))).toBe(0)      // wrong: zero
    const cheap = await scorer.score(TASK, ran('42', { steps: 1 }))
    const dear = await scorer.score(TASK, ran('42', { steps: 9 }))
    expect(cheap).toBeGreaterThan(dear)
  })

  it('keeps the name the user gave it', () => {
    expect(buildScorer({ name: 'runs-on-my-db', kind: 'command', run: ['true'] }).name)
      .toBe('runs-on-my-db')
  })

  describe('mistakes fail at mount, naming the row', () => {
    it('an unknown kind lists the ones that exist', () => {
      expect(() => buildScorer({ kind: 'vibes' } as never))
        .toThrow(/unknown kind "vibes"\. One of: gold, command, judge, replay, all/)
    })

    it('a command with nothing to run', () => {
      expect(() => buildScorer({ kind: 'command', run: [] })).toThrow(/non-empty argv/)
    })

    it('a judge with no rubric', () => {
      expect(() => buildScorer({ kind: 'judge', rubric: '  ' }, { complete: async () => 'PASS' }))
        .toThrow(/needs a `rubric`/)
    })

    it('a judge with no model to ask', () => {
      // A scorer that always fails looks exactly like a model that cannot do
      // the task, and sends the user hunting for the wrong bug.
      expect(() => buildScorer({ kind: 'judge', rubric: 'be good' }))
        .toThrow(/needs a model/)
    })

    it('an "all" with nothing in it', () => {
      expect(() => buildScorer({ kind: 'all', of: [] })).toThrow(/non-empty list/)
    })

    it('a nested mistake, reported where it is', () => {
      expect(() => buildScorer({
        kind: 'all', of: [{ kind: 'gold' }, { name: 'inner', kind: 'command', run: [] }],
      })).toThrow(/objective "inner"/)
    })
  })
})
