# Choosing a domain to evolve a search rule on

Five experiments in this line produced one result and four nulls, and every null
was a property of the *setup* rather than of the method. This page is the
resulting checklist — what a domain must supply before a `selection` experiment
on it can say anything — and what each domain in this repository actually
supplies when measured.

The short version: **measure the domain before evolving on it.** Each of the
three checks below costs one or two inner runs; each null below cost between an
hour and four.

## The four properties, and the failure that named each

| property | what happens without it | where it was learned |
|---|---|---|
| **the inner run is a function of `(rule, seed)`** | paired gains measure the endpoint, not the rule | AlgoTune: seed rule against itself, −0.0055; on another task sd 0.054 |
| **the reward is fine-grained** | archived candidates tie, so nothing can rank them | GSM-Hard at 8 held-out tasks: reward steps of 0.125, archive permanently 0.625/0.625 |
| **candidates are genuinely incomparable** | the seed is the only reachable answer | `evolve()`'s population archive: `argmax(score) == head` in 63/63 calls |
| **the search actually improves at the affordable budget** | the best-so-far curve is flat, so every rule scores identically *by arithmetic* | hyp2f1 and LLM-SRBench: 30 expansions, spread exactly 0.0000 |

The fourth is the one that is easiest to miss, because a flat curve looks like a
measurement rather than an absence of one. A rule that decides *what to expand
next* can only be judged by how fast the curve rises. If nothing ever beats the
root, there is no curve, and a deliberately-bad rule scores exactly what the
best one does.

## What each domain supplies, measured

| domain | function of the rule? | rule spread | note |
|---|---|---|---|
| synthetic landscape | **yes**, exactly 0.000 | 0.203 at budget 60 | the one that produced a result: +0.0227 / +0.0120 |
| GSM-Hard instructions | **yes**, exactly 0.000 | n/a — slot has no leverage | archive is a monotone chain |
| AlgoTune | **no** — structural | 0.0066 vs noise 0.0055 | the reward *is* a wall-clock speedup |
| LLM-SRBench | **yes**, 0.0000 | 0.0000 at 30 expansions | see *Two ways to get this wrong* |
| hyp2f1 | **yes**, 0.0000 | 0.0000 at 30 expansions | `scipy.special.hyp2f1` is near-optimal; nothing beat it |

### AlgoTune's failure is the only structural one

Every other row can in principle be fixed by configuration. AlgoTune cannot: an
expansion is scored by timing it, the measured milliseconds are written into the
next prompt (`_eval_block`, `_timing_report`, `_profile_block`), so jitter
changes the prompt, the completion cache misses, and a different program is
sampled. The timing *is* the feedback the search runs on — remove it and the
search is blind, keep it and the search is stochastic. Details in
[`metasearch-algotune.md`](metasearch-algotune.md).

### A wall-clock budget *guarding* an exact reward is almost as bad

LLM-SRBench scores `min(12, -log10(NMSE))` — accuracy, not speed — and still
came back irreproducible at first: three runs of the same rule gave 0.0731,
0.0731 and 0.4625. The cause was `PROBLEM_SECONDS = 10.0`, a per-problem
wall-clock budget guarding the metric, straddled whenever the container was
busy. Raising it to 120 took the noise floor to exactly **0.0000**.

So the property to check is not "is the reward a timing?" but **"does any
wall-clock budget stand between the program and its score?"** — and where one
does, whether it is a settable argument. AlgoTune's is not; SRBench's is.

## Two ways to get this wrong, both paid for here

**Reconstructing a configuration by hand.** The recorded run
[`era-srbench-deepseek-transform.json`](era-srbench-deepseek-transform.json)
reaches `mean_digits 0.438 → 6.85` on the *same model*, with
`per_problem=True, seed_program='linear', answer_format='program'`. The probes
above used the whole-category ERA protocol from the `library` (SINDy) seed in
`expression` format — the mode whose own docstring says *"a single program
cannot express `(-A + x1*y1 - x2*y2)/x3`"*. The flat curve was the configuration
behaving as documented. **The result files carry the exact config that worked;
read one before building a variant.**

**Picking a problem arbitrarily.** That recorded `0.438` is a mean over 111
problems, so most individual problems score zero. Choosing index 0 for a
per-problem run landed on one whose baseline is 0.00 digits and where nothing
committed in 24 expansions.

## Check it before running it

```python
# 1. is the inner run a function of the rule?  Run the SEED rule twice.
#    Anything but an identical curve means paired gains measure noise.
# 2. do rules separate?  seed vs greedy vs `return -rank`.
#    If the deliberately bad rule does not come last, the measurement is broken;
#    if all three tie, the curve is flat and there is nothing to measure.
# 3. does the search commit anything?  Read `outcomes` from the inner result.
#    `{'tree-updated': N}` with no 'committed' means no candidate beat the root.
```

`bench/metasearch_algotune.py --determinism-check N` does the first for that
port; `bench/metasearch_slots.py --leverage-check` does the third for the
`selection` slot on the instruction-evolution domains.

Check 3 also catches an environment failure that looks exactly like a scientific
one: a six-problem scan here reported zero commits everywhere, which was the API
returning `HTTP 429 AccountQuotaExceeded` for every call. The tell was in the
usage line — **137 calls carrying 4,175 tokens**, about 30 tokens per call,
where a program-writing prompt is ~10,000. A run whose model calls all failed
produces a clean, plausible, entirely meaningless null.

## Status

Verified: determinism on LLM-SRBench and hyp2f1 (both 0.0000); zero rule spread
on both under the whole-category protocol at 30 expansions.

Not yet verified: whether LLM-SRBench's **per-problem** protocol — the one the
recorded run used — gives a rising curve and separable rules. That run is the
remaining step, and it is blocked on API quota rather than on anything measured.
