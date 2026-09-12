# Evolving a decision slot: a benchmark x slot matrix

Seven cells. Each evolves one `Policies` field against four 20-task windows of one
benchmark (2 inner seeds, 3 outer sweeps = **6 outer rollouts**), then scores the
seed rule and the evolved rule paired on the same windows, two windows it never
saw, and one window of another benchmark. `deepseek-v4-flash`, temperature 0,
thinking disabled, inner runs deterministic (see *Determinism* below).
Produced by [`bench/metasearch_slots.py`](../metasearch_slots.py).

## The corrected matrix: seven cells at an adequate inner budget

Every cell re-run at **12 rollouts per inner run** instead of 4, everything else
held fixed. The last column is what the same cell reported at the old budget.

| slot | evolved on | train gain | **unseen gain** | other benchmark | committed | unseen @ 4 rollouts |
|---|---|---:|---:|---:|---:|---:|
| `task_sampler` | gsmhard | +0.029 (2/0) | **+0.042 (1/0)** | -0.052 (0/1) (aime) | 1/3 | −0.141 (0/2) |
| `task_sampler` | hotpotqa | +0.029 (3/0) | **+0.031 (1/1)** | +0.062 (1/0) (gsmhard) | 1/3 | −0.031 (1/1) |
| `task_sampler` | bbh | +0.023 (2/0) | **+0.021 (1/0)** | +0.000 (0/0) (triviaqa) | 1/3 | +0.000 (0/0) |
| `task_sampler` | gpqa | +0.026 (1/0) | **+0.000 (0/0)** | +0.000 (0/0) (bbh) | 1/3 | +0.000 (0/0) |
| `task_sampler` | aime | +0.000 (0/0) | **+0.000 (0/0)** | +0.000 (0/0) (gsmhard) | 0/3 | +0.000 (0/0) |
| `task_sampler` | mgsm_zh | +0.000 (0/0) | **+0.000 (0/0)** | +0.000 (0/0) (gsmhard) | 0/3 | −0.094 (0/2) |
| `acceptance` | gsmhard | +0.000 (0/0) | **+0.000 (0/0)** | +0.000 (0/0) (aime) | 0/3 | +0.000 (0/0) |

**No cell transfers negatively any more.** At 4 rollouts the three cells that
committed all lost on their unseen windows (−0.141, −0.094, −0.031); at 12,
four cells commit and their unseen gains are +0.042, +0.031, +0.021 and +0.000 —
none negative, and every train row is 1-3 wins with no losses. Transfer ratios
where a train gain exists: 1.45, 1.09, 0.89, 0.00.

**Three cells still commit nothing, for three different reasons.** AIME finds no
rule that beats round-robin at either budget, so its null is a property of that
cell. MGSM-zh's unseen windows sit at **0.958** on the meta-reward before
anything is evolved — no headroom to move into. And `acceptance` does not beat
the engine's own gate: a Beta posterior against an annealed threshold, which no
LLM-written rule displaced in six rollouts, at either budget.

**What is still small.** The gains are +0.02 to +0.04 of AUC, against a
hand-written "retry the tasks that failed" sampler that reaches +0.065 on the
GSM-Hard windows. One validation seed per problem, so each cell is a handful of
paired comparisons. And on GSM-Hard the cross-benchmark column (−0.052)
disagrees with the unseen column (+0.042) — within a benchmark the rule holds
up, across benchmarks this evidence does not say so.

**Three cells commit nothing, for three different reasons, and none of them is
"the reflector cannot write a rule".** AIME finds nothing that beats round-robin
at either budget. MGSM-zh's unseen windows already sit at 0.958 before anything
is evolved — no headroom. And `acceptance` does not displace the engine's own
Beta-posterior gate. For `acceptance` the proposal record separates the two
possible readings: its first six proposals were 3 refused and 3 no-diff (two
`ZeroDivisionError` from dividing `(successes, failures)` by hand instead of
calling `MergeContext.rate`), and saying so in the slot's notes took it to six
valid proposals with nothing refused — which still committed nothing. The rules
it writes do not win; it is not that it cannot write them.

Running the GSM-Hard cell with `--reflective-merge` gives **identical numbers**.
Both runs had six distinct proposals, so there was something to fuse, and every
rule this search finds is the same idea worded differently — *do not spend a
rollout on a task the artifact already solves*. On this domain the slot has
about one discoverable degree of freedom.

## What is small about it, stated plainly

* The gains are +0.02 to +0.04 of AUC. A hand-written "retry the tasks that
  failed" sampler reaches **+0.065** on the GSM-Hard windows, so the outer loop
  found roughly three quarters of what was available.
* One validation seed per problem: each cell is a handful of paired comparisons.
* On GSM-Hard the cross-benchmark column (−0.052) disagrees with the unseen
  column (+0.042). Within a benchmark the rule holds up; across benchmarks this
  evidence does not say so.

**A benchmark the seed already solves cannot be a transfer target.** Plain GSM8K
scores **1.000** for the seed instruction, so the only available move is down,
and an early version of this matrix read a −0.100 there as negative transfer
when it was a saturated window regressing. AIME (0.708) and HotpotQA (0.750)
replaced it, and HotpotQA is a different modality on purpose — a sampler that
only works on arithmetic is not a sampler.

## Run it

```bash
python -m bench.metasearch_slots --dry-run
python -m bench.metasearch_slots --model deepseek-v4-flash \
    --rounds 6 --workers 2 --seeds 3 --validate-seeds 2 \
    --train-windows 4 --unseen-windows 2 --other-windows 2 \
    --eval-cache .cache/metasearch --yes
```

`--eval-cache` also switches on the completion cache underneath it, which is
what makes an inner run a function of the sampler. Without it a paired gain
measures noise; the run plan says so.

## The result files

One per cell, all at the corrected 12-rollout inner budget:
[gsmhard](metasearch-task_sampler-gsmhard-big.json),
[hotpotqa](metasearch-task_sampler-hotpotqa-big.json),
[bbh](metasearch-task_sampler-bbh-big.json),
[gpqa](metasearch-task_sampler-gpqa-big.json),
[aime](metasearch-task_sampler-aime-big.json),
[mgsm_zh](metasearch-task_sampler-mgsm_zh-big.json),
[acceptance](metasearch-acceptance-gsmhard-big.json), plus the
[`--reflective-merge` arm](metasearch-task_sampler-gsmhard-reflective.json) of the
GSM-Hard cell and [the instruction-evolution run](metasearch-gsm.json) the slot
ports sit on top of.

The original 4-rollout arm is **not** kept. Its numbers are the last column of
the table above and its lesson -- a budget too small let a *deliberately
degenerate* sampler tie for best -- is a row in
[`metasearch-domain-selection.md`](metasearch-domain-selection.md). A
configuration that has been corrected away is a lesson, not a result.

[`metasearch-selection-gsmhard.json`](metasearch-selection-gsmhard.json) is the
`selection` slot on this domain: a rankable inner reward, and still no transfer,
because `argmax(score) == head` in 63 of 63 calls. That one is kept because it
is the measurement behind the *candidates must be genuinely incomparable* row.
