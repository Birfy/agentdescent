/**
 * The objective ladder.
 *
 * The rungs are ordinary; the composition rules are where a mistake is
 * invisible. Efficiency multiplies a hard gate rather than summing with it, and
 * stacked objectives take the minimum rather than the mean -- both because the
 * alternative produces a run that improves its own score while getting worse.
 */

import { describe, expect, it } from 'vitest'

import type { EvolutionTask, RolloutResult, Scorer } from '../src/evolution.js'
import {
  all, commandScorer, efficiency, goldScorer, judgeScorer, replayPairwiseScorer,
} from '../src/scorers.js'

const TASK: EvolutionTask = { id: 't0', prompt: 'what is the total?', meta: { gold: '42' } }

function ran(output: string, extra: Partial<RolloutResult> = {}): RolloutResult {
  return { output, finishReason: 'completed', steps: 1, toolErrors: 0, tokens: 0, ...extra }
}

describe('rung A: gold answers', () => {
  it('matches loosely by default, so a trailing period is not a failure', async () => {
    const scorer = goldScorer()
    expect(await scorer.score(TASK, ran('The answer is 42.'))).toBe(1)
    expect(await scorer.score(TASK, ran('The answer is 41.'))).toBe(0)
  })

  it('can require an exact answer', async () => {
    const scorer = goldScorer({ match: 'exact' })
    expect(await scorer.score(TASK, ran('42'))).toBe(1)
    expect(await scorer.score(TASK, ran('the answer is 42'))).toBe(0)
  })

  it('reads the gold the same way it reads the output', async () => {
    // A dataset's answer column is often the whole worked solution ending in
    // the figure. Parsing it as a bare number fails silently: every item scores
    // 0, and it reads as a hopeless model rather than a misaimed scorer.
    const scorer = goldScorer({ match: 'lastNumber' })
    const worked = { ...TASK, meta: { gold: 'he had 5, then 37 more #### 42' } }
    expect(await scorer.score(worked, ran('so the total is 42'))).toBe(1)
  })

  it('tolerates rounding when asked', async () => {
    const scorer = goldScorer({ match: 'lastNumber', tolerance: 0.01 })
    expect(await scorer.score(TASK, ran('41.7'))).toBe(1)
    expect(await scorer.score(TASK, ran('30'))).toBe(0)
  })

  it('says which task lacks a gold rather than scoring it zero', async () => {
    await expect(goldScorer().score({ id: 'bare', prompt: 'q' }, ran('x')))
      .rejects.toThrow(/task "bare" has no meta.gold/)
  })

  it('refuses a gold it can never match numerically', async () => {
    const scorer = goldScorer({ match: 'lastNumber' })
    await expect(scorer.score({ id: 't', prompt: 'q', meta: { gold: 'blue' } }, ran('7')))
      .rejects.toThrow(/holds no number/)
  })
})

describe('rung B: a command', () => {
  it('scores on the exit code', async () => {
    const pass = commandScorer({ argv: ['node', '-e', 'process.exit(0)'] })
    const fail = commandScorer({ argv: ['node', '-e', 'process.exit(3)'] })
    expect(await pass.score(TASK, ran('anything'))).toBe(1)
    expect(await fail.score(TASK, ran('anything'))).toBe(0)
  })

  it('hands the output over as a file', async () => {
    const scorer = commandScorer({
      argv: ['node', '-e',
             'const fs=require("fs");process.exit(fs.readFileSync(process.argv[1],"utf8")==="ok"?0:1)',
             '{output}'],
    })
    expect(await scorer.score(TASK, ran('ok'))).toBe(1)
    expect(await scorer.score(TASK, ran('no'))).toBe(0)
  })

  it('scores a hanging check 0 rather than throwing', async () => {
    // Throwing would drop the sample instead of scoring it, biasing the run
    // toward whichever candidates happen not to hang the checker.
    const scorer = commandScorer({
      argv: ['node', '-e', 'setTimeout(()=>{}, 30000)'], timeoutMs: 300,
    })
    expect(await scorer.score(TASK, ran('x'))).toBe(0)
  })

  it('throws when the executable is missing, because that is a config mistake', async () => {
    const scorer = commandScorer({ argv: ['dsh-no-such-binary-anywhere'] })
    await expect(scorer.score(TASK, ran('x'))).rejects.toThrow(/not found/)
  })

  it('rejects an empty command where the mistake is', () => {
    expect(() => commandScorer({ argv: [] })).toThrow()
  })
})

describe('rung B: a rubric', () => {
  it('passes on an explicit PASS', async () => {
    const scorer = judgeScorer({ rubric: 'snake_case only', judge: async () => 'PASS — fine' })
    expect(await scorer.score(TASK, ran('x'))).toBe(1)
  })

  it('treats an unreadable verdict as a failure, not a pass', async () => {
    const scorer = judgeScorer({ rubric: 'r', judge: async () => 'well, it depends' })
    expect(await scorer.score(TASK, ran('x'))).toBe(0)
  })

  it('puts the criteria and the answer in front of the judge', async () => {
    let seen = ''
    const scorer = judgeScorer({ rubric: 'no SELECT *', judge: async (p) => { seen = p; return 'PASS' } })
    await scorer.score(TASK, ran('select id from t'))
    expect(seen).toContain('no SELECT *')
    expect(seen).toContain('select id from t')
  })
})

describe('rung C: comparing against what happened before', () => {
  const scorer = replayPairwiseScorer()

  it('prefers the candidate when the previous turn broke', async () => {
    expect(await scorer.score(TASK, ran('ok'), ran('x', { finishReason: 'error' }))).toBe(1)
  })

  it('fails the candidate when it is the one that broke', async () => {
    expect(await scorer.score(TASK, ran('x', { finishReason: 'error' }), ran('ok'))).toBe(0)
  })

  it('prefers fewer tool errors when both completed', async () => {
    expect(await scorer.score(TASK, ran('a', { toolErrors: 0 }), ran('b', { toolErrors: 2 }))).toBe(1)
    expect(await scorer.score(TASK, ran('a', { toolErrors: 3 }), ran('b', { toolErrors: 1 }))).toBe(0)
  })

  it('says "no evidence" rather than inventing a winner', async () => {
    expect(await scorer.score(TASK, ran('a'), ran('b'))).toBe(0.5)
  })

  it('asks a judge only when the objective signals tie', async () => {
    let asked = 0
    const judged = replayPairwiseScorer({ judge: async () => { asked += 1; return 'B is better' } })
    await judged.score(TASK, ran('a', { toolErrors: 0 }), ran('b', { toolErrors: 5 }))
    expect(asked).toBe(0)                        // decided without spending a call
    expect(await judged.score(TASK, ran('a'), ran('b'))).toBe(1)
    expect(asked).toBe(1)
  })

  it('does not punish the artifact for a missing reference', async () => {
    expect(await scorer.score(TASK, ran('fine'))).toBe(0.5)
    expect(await scorer.score(TASK, ran('x', { finishReason: 'error' }))).toBe(0)
  })
})

describe('efficiency multiplies, it never sums', () => {
  const gold = goldScorer()

  it('is zero when the answer is wrong, however cheap it was', async () => {
    // The failure this prevents: a sum lets "two fewer steps" buy off "got it
    // wrong", the aggregator starts accepting artifacts that are faster and
    // worse, and the held-out score still goes up.
    const scorer = efficiency(gold)
    expect(await scorer.score(TASK, ran('41', { steps: 0 }))).toBe(0)
  })

  it('rewards a correct answer more the fewer steps it took', async () => {
    const scorer = efficiency(gold, { stepBudget: 10 })
    const cheap = await scorer.score(TASK, ran('42', { steps: 1 }))
    const dear = await scorer.score(TASK, ran('42', { steps: 9 }))
    expect(cheap).toBeGreaterThan(dear)
    expect(dear).toBeGreaterThanOrEqual(0.5)     // still a pass
    expect(cheap).toBeLessThanOrEqual(1)
  })

  it('never drops a correct answer below half', async () => {
    const scorer = efficiency(gold, { stepBudget: 2 })
    expect(await scorer.score(TASK, ran('42', { steps: 999 }))).toBe(0.5)
  })
})

describe('stacked objectives take the minimum', () => {
  const yes: Scorer = { name: 'yes', score: async () => 1 }
  const no: Scorer = { name: 'no', score: async () => 0 }
  const half: Scorer = { name: 'half', score: async () => 0.5 }

  it('fails when any single objective fails', async () => {
    // The user said "and". A mean would call this a partial success.
    expect(await all('both', [yes, no]).score(TASK, ran('x'))).toBe(0)
  })

  it('takes the weakest when all pass', async () => {
    expect(await all('both', [yes, half]).score(TASK, ran('x'))).toBe(0.5)
  })

  it('stops at the first failure instead of paying for the rest', async () => {
    let called = 0
    const counted: Scorer = { name: 'counted', score: async () => { called += 1; return 1 } }
    await all('short', [no, counted]).score(TASK, ran('x'))
    expect(called).toBe(0)
  })

  it('rejects an empty list where the mistake is', () => {
    expect(() => all('none', [])).toThrow()
  })
})
