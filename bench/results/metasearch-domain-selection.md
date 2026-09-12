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
| LLM-SRBench | evaluator yes; **whole run no** until a duration was taken out of the prompt | 0.0842 on `lsr_synth`, 0.0000 under the category protocol | see below, and *Two ways to get this wrong* |
| hyp2f1 | **yes**, 0.0000 | 0.0000 at 30 expansions | `scipy.special.hyp2f1` is near-optimal; nothing beat it |
| SWE-bench-Science | **yes**, byte-identical | n/a — nothing beat the root | best on properties 1–3 of any real domain here; see below |

### SWE-bench-Science: three properties out of four, and the fourth is not the domain's

The strongest real-data domain on the first three and the clearest failure on the
fourth. Verification is a container running a fixed test suite, so it is
**byte-identical across repeats** — the first real-data evaluator here that is a
function of its input by construction rather than by tuning a wall-clock budget
out of the way. Reward granularity is 3 to 31 levels once the grader's nested
partial credit is read (the headline `reward` is binary), 10 of 15 baselines are
partially correct, and a verification costs 5.9 s at the median.

And nothing beat the root: **0 of 14 attempts on a weak model, 0 of 14 on a
strong one.** The stronger model produces visibly healthier candidates — 14 of 14
reach the verifier rather than 13 of 14 — and the curve is still flat. So this
is not the domain and not the model: it is the agent phase the benchmark is built
around, which this repository deliberately does not reimplement
(`harbor run --agent`). Until that exists the curve stays flat for reasons that
say nothing about selection rules.

Two things worth knowing before anyone runs it, both in `_harbor.py`'s comments:
the verifier is a **separate image** whose `test.sh` begins `git reset --hard`
and applies the patch from `/logs/artifacts/model.patch` (apply it yourself and
it is silently discarded, and every candidate scores the baseline); and **6 of
15 published verifier images ship a grader that looks for a private test file by
a name it does not have**, so `reward` is 0 for any patch whatsoever. Mounting
the dataset's own `tests/grader.py` over the image's repairs them — baselines
with no usable signal go from 6 of 15 to 0 of 15.

### One defect, three disguises: anything that varies must not reach the model

Three domains failed the *"is the run a function of the rule?"* property, in
what looked like three different ways and was one:

| where it hid | what it did |
|---|---|
| **the reward is a timing** (AlgoTune) | an expansion is scored by timing it, and the milliseconds go into the next prompt (`_eval_block`, `_timing_report`, `_profile_block`). Jitter changes the prompt, the cache misses, a different program is sampled |
| **a wall-clock budget guards the reward** (LLM-SRBench, first) | `PROBLEM_SECONDS = 10.0` straddled whenever the container was busy: three runs of one rule gave 0.0731, 0.0731, 0.4625. Raising it to 120 took the *evaluator's* floor to 0.0000 |
| **a duration sits beside the reward in the prompt** (LLM-SRBench, again) | `which scored 2.998 in 0.84s.` — the score identical, the duration not. Prompts of the same length (8162) and different hashes, diverging at model call 1 |

**Only AlgoTune's is irreparable.** Remove the timing and the search is blind;
keep it and the search is stochastic — the timing *is* the feedback. The port
itself works (a 5.6x speedup over the task's own reference on
`psd_cone_projection`), and `bench/metasearch_algotune.py --determinism-check N`
prints the noise floor before the outer loop starts, which is the number that
decides whether anything the run reports is evidence.

The third one is the expensive one, because **this page had asserted that
LLM-SRBench was free of it** — and that assertion cost the `lsr_synth` run a
**0.024 paired noise floor** with a deterministic evaluator and a completion
cache both in place. So the rule is wider than "is the reward a timing":

> **Nothing that varies between two runs of the same candidate may appear in the
> prompt** — not the metric, and not a fluent aside beside it. Durations, memory
> figures, timestamps, paths with a pid in them, and any set iterated without
> `sorted()` all qualify.

The earlier determinism check missed it for the reason it looked convincing: it
compared *scores*, and the score was never wrong. Two cheap checks that would
have caught it — grep the rendered prompt for `\d+\.\d+s` and for the run's own
temp paths, and run the seed rule twice and diff the **prompts**, not the
rewards. Pinned by
`tests/test_era_srbench.py::test_no_wall_clock_duration_reaches_the_srbench_prompt`.

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

### 4. count the failed model calls, and abort on them

An environment failure looks exactly like a scientific one. A six-problem scan
here reported zero commits everywhere; the API was returning `HTTP 429
AccountQuotaExceeded` for every call. A run whose model calls fail produces a
clean, plausible, entirely meaningless null, because **a call that fails is a
candidate that was never written, which the engine sees as a rollout that found
nothing** — biased in exactly the direction that looks like a real negative
result.

The first version of this check compared **tokens per call**: that scan showed
137 calls carrying 4,175 tokens, about 30 per call, where a program-writing
prompt is ~10,000. **That threshold is not sufficient, and it let a later run
through.** `with_retries` retries a refused call and often succeeds, so the
average stays plausible while most attempts fail — measured on an evolution run
here: **483 calls, 296 failures (61%), and still 1,817 tokens per call**, which
sails past any per-call threshold. Its validation table read
`train +0.0013 / held-out -0.0378`, exactly the shape of an honest "fits its
training problems, does not transfer" finding, and it meant nothing.

So check `Usage.failures` directly and abort:

```python
fail_rate = usage.failures / usage.calls if usage.calls else 0.0
if fail_rate > 0.02:
    raise SystemExit(f"{usage.failures}/{usage.calls} model calls failed "
                     f"({fail_rate:.0%}) -- this run is void")
```

## The domain that passes: LLM-SRBench `lsr_synth`, per-problem

Scanning problems for a **rising curve** — the fourth property — separates the
two LLM-SRBench categories completely:

| category | problems with a rising curve |
|---|---|
| `lsr_transform` | **0 of 6** — 3 solved straight to the 12.00 cap, 3 never moved |
| `lsr_synth` | **4 of 8** — one from each of bio_pop_growth, chem_react, matsci, phys_osc |

`lsr_transform`'s answers are transformations of known equations: get the
structure right and NMSE collapses to the cap, get it wrong and score nothing.
A step function has no slope for a selection rule to accelerate. `lsr_synth` is
genuine discovery on noisy scientific data, so a candidate can be *partly*
right — baselines of 2.50 and 2.76 climbing to 4.05 and 4.18, through two
commits, in intermediate steps.

Seed / greedy / worst-first on the four rising problems, 8 expansions, with the
seed rule run again as the noise floor:

| problem | seed | greedy | worst-first | noise | spread |
|---|---:|---:|---:|---:|---:|
| `bpg1` | 0.2272 | 0.2089 | 0.2728 | 0.0042 | 0.0638 |
| `crk31` | 0.3419 | 0.3419 | 0.3276 | 0.0000 | 0.0143 |
| `matsci13` | 0.5993 | 0.5993 | 0.5995 | 0.0000 | 0.0003 |
| `po22` | 0.1823 | 0.1084 | 0.3667 | 0.0000 | 0.2583 |
| **mean** | **0.3377** | **0.3146** | **0.3917** | **0.0011** | **0.0842** |

**Measurable**: noise 0.0011 against spread 0.0842, a 75x margin. This is the
first real-data domain here where rules separate well above the noise floor.

### The deliberately-bad rule wins, and that is not a broken measurement

`worst-first` (`return -rank`) has the highest mean, and on `po22` it doubles
the seed. The check stated elsewhere on this page — *"if worst-first does not
come last, the measurement is broken"* — **does not hold here, and the reason is
worth keeping.** That check was calibrated on the synthetic landscape, where
rank carries real information. In an 8-node tree of mostly-failed candidates it
does not: expanding the *lowest*-ranked node mostly means staying near the root
and trying independent variations rather than deepening a bad line. Breadth
beats depth at a tiny budget, and pure exploitation (`greedy`) is worst of the
three.

So the direction here is the **opposite** of the landscape's, where "explore
less" was worth +0.0227. That makes it a good target rather than a spoiled one:
a rule evolved on `lsr_synth` should discover "explore more", and the transfer
question becomes whether that survives a larger budget.

Hold the caveats: four problems, one seed each, 8 expansions; the spread is
carried by `po22` (0.2583) while `matsci13` contributes 0.0003; and `bpg1`'s
noise is 0.0042 rather than 0.

## The prediction held

This page predicted, before any evolution was run, that *"a rule evolved on
`lsr_synth` should discover 'explore more'"* — from `worst-first` beating
`greedy` at a small budget. It did:
[`metasearch-selection-srbench.md`](metasearch-selection-srbench.md) reports a
rule that **adds** an exploration term and a depth penalty, 5 wins / 0 losses /
4 ties over nine held-out paired comparisons (sign test p = 0.031).

One property this page did **not** state, and which cost half a sample size:
**a seed is only a replicate if it moves the run.** On this domain ERA's seed
moves nothing — seeds 7, 8, 9, 10 and 123 give byte-identical curves with zero
model calls, precisely *because* the inner run is a function of the rule. The
determinism that makes a domain measurable is the same thing that makes its seed
axis inert. Replicate across problems, or across the data split, and check that
the numbers differ before counting them as independent.

## Status

Verified: determinism on LLM-SRBench (0.0000 once `problem_seconds` is raised)
and hyp2f1 (0.0000); zero rule spread on both under the whole-category protocol
at 30 expansions; and on `lsr_synth` per-problem, a measurable spread of 0.0842
against a 0.0011 noise floor.

Run, and positive: `meta_evolve` over `priority()` on the rising `lsr_synth`
problems, validated on problems the outer loop never saw —
[`metasearch-selection-srbench.md`](metasearch-selection-srbench.md).
