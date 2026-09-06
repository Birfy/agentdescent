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

### Reflective merge changed nothing here

The outer loop's two workers produce two contradicting candidate rules per
round, and by default one is ranked out and discarded. Running the corrected
GSM-Hard cell with `--reflective-merge` — a model synthesising one rule from
both, gated by the slot's own validator — gives **identical numbers**: +0.029
train, +0.042 unseen, −0.052 other. Both runs had six distinct proposals, so
there was something to fuse; the two evolved rules differ in text and score
identically on all seven validation problems.

The reason is visible in the rules themselves: every one this search finds is
the same idea worded differently — *do not spend a rollout on a task the
artifact already solves*. On this domain the slot has about one discoverable
degree of freedom, which is also why a deliberately degenerate sampler captured
it by accident at the small budget. Merging two expressions of one idea has
nothing to add.

## Why the matrix is flat: the inner budget was too small to measure a sampler

The seven cells above ran at **4 rollouts per inner run over 12 train tasks**,
and that is not enough for the slot to express anything. Instrumented on one
window, round-robin spent **three of its four rollouts on tasks the artifact had
already solved** — and the engine asks no proposal from a rollout that passed, so
the run bought one proposal out of four. A sampler that learns from `record`
cannot do better, because it has to spend a rollout to discover a failure and the
budget is gone before it can act on what it learned.

The consequence is worse than noise. Four samplers on two windows, the same
inner problem at two budgets:

| window | rollouts | round-robin | repeat-failures | skip-solved | **always-first (degenerate)** | spread |
|---|---:|---:|---:|---:|---:|---:|
| gsmhard-0 | 4 | 0.625 | 0.719 | 0.688 | **0.719 — tied best** | 0.094 |
| gsmhard-2 | 4 | 0.500 | 0.500 | 0.625 | 0.500 | 0.125 |
| gsmhard-0 | 12 | 0.625 | 0.740 | 0.729 | 0.740 | 0.115 |
| gsmhard-2 | 12 | 0.625 | 0.573 | **0.708** | **0.500 — last** | 0.208 |

`always-first` picks `keys[0]` every time and is there to be bad. **At four
rollouts it ties for best; at twelve it is last.** A budget that inverts the
ranking of a deliberately broken rule is not measuring the rule, and the flat
matrix above is substantially that artifact rather than a property of the
method.

So the table above should be read as: *at a budget too small for the slot to
matter, nothing was learned and nothing transferred* — which is a fact about the
configuration. The default is now 12 rollouts, and the rule of thumb the
measurement supports is **at least one rollout per train task**.

## What the matrix did establish

- The machinery runs end to end on four datasets and two slots, with the inner
  run deterministic, so every paired number above is exact rather than noisy.
- The proposal record makes each null legible: `acceptance` first produced 3
  refused and 3 no-diff proposals out of 6 — two `ZeroDivisionError` from
  dividing `(successes, failures)` by hand instead of calling
  `MergeContext.rate`, and one `TypeError` from unpacking the cheap-layer float
  as a pair. Saying so in the slot's notes took it to 6 valid proposals with
  nothing refused, and the cell still committed nothing — which separates "the
  reflector cannot write this" from "the rules it writes do not win".
- Two benchmarks were added because the first validation target was useless:
  plain GSM8K scores **1.000** for the seed instruction, so it can only move
  down. AIME 1983-2024 (0.708) and HotpotQA (0.750) both leave real headroom,
  and HotpotQA is a different modality — a sampler that only works on arithmetic
  is not a sampler.

## Operational note: background runs do not survive an idle session

A four-cell matrix launched with `nohup` advanced for eight minutes and then
stopped: the parent shell and the child were both gone, with no failure line
between them, while disk, memory and the endpoint were all fine. In this remote
container a background job only progresses while the session is active. The
completion cache turns that from a loss into a pause — a re-run repeats the same
deterministic calls and hits cache — so the cells above were run one at a time
in the foreground.

---
## The deep dive: one cell at twelve rollouts

Everything above is six outer rollouts per cell. This section is the same
`task_sampler` x GSM-Hard cell run at **twelve**, with two validation seeds, and
it is where the mechanism was worked out. Its numbers are more favourable than
the matrix row and that difference is the point: read the two together, not
either alone.

The outer artifact is the `task_sampler` policy — which task each rollout of an
**inner** `evolve()` spends. One outer rollout is one whole inner run: evolving
an instruction against a 20-task slice of GSM-Hard, five sweeps, one worker,
`deepseek-v4-flash` at temperature 0 with thinking disabled. The meta-reward is
the inner run's AUC (mean best-so-far held-out reward). Governance is L1, so
every merge also passes the oracle. Produced by
[`bench/metasearch_slots.py`](../metasearch_slots.py); raw record in
[`metasearch-gsm.json`](metasearch-gsm.json).

Four GSM-Hard windows to evolve on (3 seeds each = 12 outer tasks, 6 held out),
six outer sweeps, 12 rollouts. Validation is paired on 2 fresh seeds per
problem: the same windows, two GSM-Hard windows the run never saw, and two
GSM8K windows.

| group | problems | seed rule | evolved rule | gain | wins/losses |
|---|---|---:|---:|---:|---:|
| **train** | 4 GSM-Hard windows | 0.719 | **0.769** | **+0.050** | 4/1 |
| **unseen** | 2 GSM-Hard windows | 0.713 | 0.725 | +0.013 | 1/1 |
| **other** | 2 GSM8K windows | 1.000 | 0.900 | −0.100 | 0/1 |

Transfer ratio (gain over the train gain): **0.25** within the benchmark,
**−2.00** across it. The outer loop committed twice and was oracle-rejected four
times; 12 proposals, none refused by the gate, none invalid. 1,573 model calls,
855 s wall, with 555 of 1,025 completions served from cache.

The rule it found:

```python
class Policy:
    # UCB-style exploration that balances trying unknown tasks with exploiting
    # known low scores, and avoids re-picking solved tasks when alternatives exist.
    def pick(self, keys, round_index):
        unsolved = [k for k in keys if self.scores.get(k, 0) < 1.0] or keys
        ...
```

That is the mechanism the slot is about: the engine asks for **no proposal from
a rollout that passed**, so a pick landing on an already-solved task buys
nothing, and the sampler that avoids them converts more of a fixed rollout
budget into proposals.

## Read the three rows apart, not down

**The train row is the claim that holds *in this configuration*.** +0.050 across four windows, four
wins and one loss on paired seeds, and the inner runs are deterministic (see
below), so the pairing is exact rather than noisy.

**The unseen row is weak and honest: +0.013, one win and one loss.** On this
evidence the evolved rule is closer to a fit to the windows it was evolved on
than to a better sampler in general. A transfer ratio of 0.25 is what that
looks like; it is not a null result and it is not a win.

**The other row cannot be read as transfer at all, and that is a flaw in the
validation set.** The seed rule already scores **1.000** on both GSM8K windows —
a current model solves them, which is what `docs/results.md` says about GSM8K —
so there is no headroom and the only available move is down. The −0.100 is one
of four paired runs regressing a saturated window. **A benchmark the seed
already solves cannot serve as a transfer target**, and the next run of this
should replace GSM8K with something that leaves room.

**A hand-written reference says the search left value on the table.** A
"retry the tasks that failed" sampler written by hand scores **0.784** on the
same four windows against the seed's 0.719 and the evolved rule's 0.769. The
outer loop found roughly three quarters of the available gain in 12 rollouts.

## What had to be fixed before any of this was measurable

Eight live runs. The first seven committed nothing, and each one was a
different defect rather than the same one:

| symptom | cause | fix |
|---|---|---|
| crash on the last line, whole run lost | the payload built its own usage dict with the Anthropic SDK's field names | use `examples._measure.usage_dict`; a test that drives `main()` |
| `{'oracle-rejected': 3}`, nothing commits | **the inner run was not reproducible** — the gate was cached, the rollouts were live calls, and temperature 0 was necessary but not sufficient. Validating the seed rule against *itself* reported a gain of −0.0625 | `cached_completion` memoises `prompt -> text`; every paired gain is now exactly 0.000 when the two rules are identical, and the run got 6x faster |
| a null result that could not say what it tried | proposals were not recorded | record each with whether the gate would take it; `SourceSlot.accepts()` so the reporter asks the gate instead of imitating it |
| all 12 rollouts on one problem | `meta_evolve` built its task list problem-major and `evolve()` splits by position — the search never rolled out on the problem it was judged on | interleave seed-major |
| every proposal killed the inner run with `KeyError` | proposals kept per-task state and answered from memory, not from the shard they were handed; the smoke test used one fixed key list, where that is invisible | the sampler smoke test now walks a changing shard, an all-seen shard, and unsorted lists |
| valid proposals, still nothing commits | the outer gate held out **2 windows**, and the good proposals tied the seed on exactly those two while gaining on the two it never gated on | evolve on four windows, so the gate averages over more of the distribution |

The last row is the one worth carrying to another slot: **an effect that only
appears across problems cannot be committed by a gate that sees one or two of
them**, and under L1 a tie is a veto, so a narrow gate reads as "nothing works".

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
