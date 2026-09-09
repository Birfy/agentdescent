# Evolving a search rule on real scientific data

The artifact is the source of `priority(rank, visits, total, prior, depth,
n_nodes)` — the rule ERA's tree uses to pick which node to expand next. It was
already evolved once on a **synthetic landscape**, where it gained +0.0227 and
landed on the Pareto front of the hand-tuned PUCT family
([`metasearch-tree.md`](metasearch-tree.md)). This page is the same experiment on
a real benchmark: **LLM-SRBench `lsr_synth`**, symbolic regression on noisy
scientific data, per-problem protocol.

It is the first positive real-data result in this line, and it took two
corrections to get to — one to the harness and one to my own reading of the
evidence. Both are below, because they are the transferable part.

## The result

Nine independent paired comparisons — three held-out problems the outer loop
never saw, each on three data splits:

| problem | split 0 | split 1 | split 2 |
|---|---:|---:|---:|
| `bpg23` | **+0.0741** | +0.0000 | **+0.0714** |
| `bpg7` | **+0.0375** | +0.0000 | +0.0000 |
| `matsci8` | **+0.0736** | +0.0000 | **+0.2339** |

| | |
|---|---:|
| wins / losses / ties | **5 / 0 / 4** |
| mean gain | **+0.0545** |
| median gain | +0.0375 |
| sign test (one-sided, 5 non-tied pairs) | **p = 0.031** |
| model calls / failures | 20 / **0** |

**The evolved rule never loses.** It either wins substantially or changes
nothing, and it replicates across splits of the same problem — `bpg23` gains
+0.0741 and +0.0714 on two different splits, against a paired noise floor
measured at exactly **0.0000**.

Hold the caveats: 4 of 9 comparisons are exact ties, the mean is carried by
`matsci8` split 2 (+0.2339) so the median is the more robust summary, and this
is 3 problems, not 30.

## The rule, and why its *direction* is the strongest evidence

```python
def priority(rank, visits, total, prior, depth, n_nodes):
    c, d = 1.0, 0.1
    exploration = c * math.sqrt(math.log1p(total) + 1.0) / (1.0 + visits)
    prior_term = prior * math.sqrt(total + 1.0) * (1.0 - rank)
    return rank + prior_term + exploration - d * depth
```

It **adds** an exploration term and a depth penalty: *explore more, go
shallower*. That is the **opposite** of what evolved on the synthetic landscape,
where the whole gain came from exploring **less**.

And it is the direction
[`metasearch-domain-selection.md`](metasearch-domain-selection.md) predicted in
advance, from a measurement made before any evolution was run: on `lsr_synth`,
`worst-first` (`return -rank`) *beat* `greedy`, because in an 8-node tree of
mostly-failed candidates, expanding the lowest-ranked node means staying near the
root and trying independent variations instead of deepening a bad line.

A predicted direction, found independently by the search, is worth more than the
size of the gain.

## Correction 1: the domain was never a null — a duration was in the prompt

This domain was previously recorded as a **null**: 10 rollouts, 0 commits. Its
free control — the seed rule scored against *itself* — reported a **0.024 paired
noise floor**, which did not fit a domain whose evaluator had been measured
deterministic to 0.0000 with completions cached.

Tracing every model call of three identical runs put the divergence at call 1,
where both prompts were 8162 bytes and hashed differently. The entire diff:

```
-which scored 2.998 in 0.84s.
+which scored 2.998 in 0.89s.
```

Both prompt builders printed the candidate's wall-clock to two decimals. Same
score, different duration, different prompt text — and the prompt *is* the
completion-cache key, so the cache missed and the model sampled a different
program. **That is AlgoTune's mechanism exactly, in the domain the checklist had
explicitly cleared of it.** Removed, and the seed rule against itself is now
0.0000 on 3 of 3 problems retested.

The general rule that replaces the old one: **nothing that varies between two
runs of the same candidate may appear in the prompt** — not the metric, and not
a fluent aside beside it.

## Correction 2: the seed axis is inert, so half the sample was imaginary

The first validation reported "**2 wins, 0 losses**" per problem. That was **one
comparison counted twice.**

ERA's `seed` randomises nothing in this configuration. Seeds 7, 8, 9, 10 and 123
produce byte-identical curves on `bpg23` — `auc 0.050018` every time — with
**zero model calls**, because at `mode='serial'`, `workers=1`, with a
deterministic evaluator and a prompt-keyed cache, there is nothing left for a
seed to vary. A "fresh seeds 9/10" run queued as the independent check
reproduced 7/8 to four decimals and made 0 API calls; it was not independent at
all.

This is the flip side of the property the domain was chosen for. **The noise
floor is 0.0000 because the inner run is a function of the rule — and that same
determinism makes the seed axis inert.** Replication has to come from problems,
or from the data-split seed handed to `prepare_problem_suite`, which every run
so far had pinned at 0. That is the axis the table above uses, and it moves:
`bpg23` goes +0.0741 / +0.0000 / +0.0714 across three splits.

**If a validation reports identical numbers for different seeds, the seed is not
a replicate. Check that it moves before counting it.**

### The same defect shrinks the outer gate, which is the weaker half of this run

`meta_evolve` was given 4 problems x 3 seeds = **12 outer tasks**, and
`held_out_frac=0.4` cut 5 of them for the gate. With the seed inert, those 12
tasks are 4 distinct problems repeated three times each, so the gate was really
deciding on about **2 distinct problems**, not 5.

It shows in the run: 6 rollouts, 2 commits, **0 rejections**. The gate never had
to turn anything down, so it did no filtering — it only confirmed that each
commit raised held-out reward (0.2838 -> 0.2868). That is the honest weak point
of the evolution phase, and it is why the nine post-hoc comparisons above, on
problems and splits the outer loop never touched, are what the result rests on
rather than the commit count.

A run repeating this should pass `seeds=[0]` and put the budget into **more
problems** instead.

## What it cost, and the guard it tripped

The evolution itself: 6 rollouts, 2 commits, 40 model calls, 197k tokens, 2.7
hours wall clock. It tripped its own abort guard — **1 of 40 calls failed
(2.5%), against a 2% threshold** set after a 61%-failure run produced a
plausible null.

The guard was not waived. Its rationale is that a failed call is a candidate
never written, which the engine reads as a rollout that found nothing — so the
bias runs toward a **null**, and this result is positive. That is an argument for
re-running the claim, not for accepting it. The replication above is that re-run:
**20 calls, 0 failures, clean.**

## Reproducing

```bash
# 1. the domain must clear all four properties first -- see metasearch-domain-selection.md
# 2. the held-out problems are those whose curve rises; scan and checkpoint them
# 3. evolve, then replicate across DATA SPLITS, not across ERA seeds
```

Numbers, rows, the evolved source and the full config are in
[`metasearch-selection-srbench.json`](metasearch-selection-srbench.json).

The pre-fix run is **not** kept. It was a null produced by the prompt bug above,
its own annotation said every row compared the seed rule with itself, and the
story it documented is the two corrections on this page. A void run is a bug
report, not a result, and prose is the right place for it.
