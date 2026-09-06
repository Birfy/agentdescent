# Evolving a decision slot: a benchmark x slot matrix

Seven cells. Each evolves one `Policies` field against four 20-task windows of one
benchmark (2 inner seeds, 3 outer sweeps = **6 outer rollouts**), then scores the
seed rule and the evolved rule paired on the same windows, two windows it never
saw, and one window of another benchmark. `deepseek-v4-flash`, temperature 0,
thinking disabled, inner runs deterministic (see *Determinism* below).
Produced by [`bench/metasearch_slots.py`](../metasearch_slots.py).

| slot | evolved on | item baseline | train gain | unseen gain | other benchmark | committed | proposals |
|---|---|---:|---:|---:|---:|---:|---|
| `task_sampler` | gsmhard | 0.500 | +0.047 (1/2) | -0.141 (0/2) | -0.031 (0/1) (aime) | 2/3 | 6, 0 refused |
| `task_sampler` | aime | 0.708 | +0.000 (0/0) | +0.000 (0/0) | +0.000 (0/0) (gsmhard) | 0/3 | 6, 0 refused |
| `task_sampler` | hotpotqa | 0.750 | +0.016 (2/0) | -0.031 (1/1) | +0.062 (1/0) (gsmhard) | 1/3 | 6, 0 refused |
| `task_sampler` | mgsm_zh | 0.625 | +0.000 (0/0) | -0.094 (0/2) | +0.062 (1/0) (gsmhard) | 1/3 | 6, 1 refused |
| `task_sampler` | gpqa | 0.375 | +0.000 (0/0) | +0.000 (0/0) | +0.000 (0/0) (bbh) | 0/3 | 6, 0 refused |
| `task_sampler` | bbh | 0.542 | +0.000 (0/0) | +0.000 (0/0) | +0.000 (0/0) (triviaqa) | 0/3 | 6, 0 refused |
| `acceptance` | gsmhard | 0.500 | +0.000 (0/0) | +0.000 (0/0) | +0.000 (0/0) (aime) | 0/3 | 6, 0 refused |

`(w/l)` counts paired wins and losses over the windows in the group. **One
validation seed per problem**, so every `sd` in the raw files is 0.000 by
construction and no cell carries a variance estimate — that is a budget choice,
not a measurement.

## What this says, sober

**Four of seven cells committed nothing at all.** On AIME, GPQA, BBH and the
`acceptance` slot every proposal was valid and every merge was oracle-rejected:
under L1 a candidate must strictly beat the base on ground truth, and none did.
For `acceptance` that is a result about the engine's own default too — a Beta
posterior against an annealed threshold was not beaten by an LLM-written rule in
six rollouts.

**Not one cell transferred.** Every cell that committed something lost on the
unseen windows of its own benchmark: −0.141 on GSM-Hard, −0.094 on MGSM-zh,
−0.031 on HotpotQA. Three for three, in the same direction. The two positive
`other` figures (+0.062, both on a single GSM-Hard window at one seed) are one
paired comparison each and carry no weight against that.

**Where a train gain exists, one window carries it.** GSM-Hard's +0.047 is
gsmhard-2 moving +0.250 while the other three windows do not move or move down
(1 win, 2 losses). MGSM-zh's train row is +0.000 across all four windows even
though the run committed — the rule it found changed nothing where it was
evolved and hurt where it was not.

**And a single configuration proves nothing.** The GSM-Hard cell at *twelve*
outer rollouts with two validation seeds (the deep dive below) reads +0.050 with
4 wins on train and +0.013 on unseen. The same cell at six rollouts with one
seed reads +0.047 with 1 win and −0.141 on unseen. Same code, same model, same
windows.

The honest summary over seven cells, six benchmarks and two slots: **at this
budget, evolving a decision slot yields occasional small gains on the problems
it was evolved on, and those gains do not generalise — not to unseen windows of
the same benchmark, and not across benchmarks.** Every rule it found belongs to
one family (prefer the tasks the artifact has not solved), which is the
mechanism the slot is about; the search finds it and then fails to show it is
worth anything off the training windows.

**A caveat that limits several cells.** The meta-reward is the inner run's AUC,
and some windows sit at 1.000 on it before anything is evolved — a whole BBH
group, two MGSM-zh windows, GPQA's transfer window. An inner run that starts at
the ceiling cannot show a sampler doing anything, so those rows are not evidence
either way. Choosing windows by their *inner AUC* rather than by their item
baseline is the fix, and it is not done here.

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
