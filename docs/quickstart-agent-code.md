# Quickstart — evolve an agent's code

A complete, measured case: one `evolve()` call that takes a **deliberately
minimal running agent** — a bare model call and a one-line instruction file —
and evolves it on GSM-Hard against a hosted thinking model
(`deepseek-v4-flash` behind an Anthropic-shaped endpoint). One run, 83
minutes: **test 0.692 → 0.757** on a 107-problem split the engine never saw.

The case makes one point above the others: the winning configuration was not
a cleverer architecture. It was a *lower starting point and a larger budget*,
with the engine free to decide what the agent should become.

## What evolves

The artifact is a two-file tree, executed for real on every rollout:

- `agent.py` — initially ~40 lines: read `skills/`, send one prompt, print
  the reply. No tools, no retries, no output parsing.
- `skills/strategy.md` — initially the single line `Solve the problem.`

Both are editable (`FileTree`, at most 2 files per diff). A rollout
materialises the candidate into a workspace and runs
`python3 agent.py "<question>"` through `code_runner` — process isolation,
trimmed environment, hard timeout. Reward is exact-match on the last number
of stdout, normalised the way GSM-Hard's float targets require
(`-9867630.0` ≡ `-9867630`).

Data: 320 GSM-Hard problems (the offline sample shipped in
`examples/_gsmhard_sample.json`), split 106 train / 107 held-out / 107 test
by a seeded shuffle. The engine trains and gates on the first two; test is
measured once before and once after.

## The call

```python
result = evolve(
    train + held_out, reward, run=run, propose=propose,
    strategy=FileTree(initial_files=INITIAL, max_files_per_diff=2),
    artifact_id="dgm-agent",
    blast_radius=0.6,                     # agent code is a harness: L1, oracle-gated
    n_workers=6, asynchronous=True, async_ratio=3,
    eval_concurrency=16,
    rounds=30, max_rollouts=180, patience=8, target_reward=0.95,
    self_verify=False, cheap_eval_tasks=6,
    agg_config=AggregatorConfig(bounded_gate=True, base_delta=0.8,
                                anneal_half_life=256, batch_trigger=4),
    policies=Policies(**reflective_merge(fusion_model, max_proposals=4)),
    staleness_policy=ReflectiveStaleness(),
    held_out_frac=0.5, shuffle=False, seed=0,
)
```

Knobs that are the lessons of this series, not defaults:

- **`max_rollouts=180`** — budget is what lets the engine climb in steps.
  Runs of 60–72 rollouts on the same task plateaued after one commit; this
  run committed three times, each on top of the last.
- **`base_delta=0.8, anneal_half_life=256`** — a relaxed, flat acceptance
  threshold. Real fixes on this workload are worth 1–4 held-out tasks each;
  under the default schedule the Beta posterior wants more lift than that
  and correct small fixes die at the gate.
- **`max_proposals=4` / `batch_trigger=4`** — reflective merge synthesises a
  few competing proposals per model call rather than summarising many.
- **No `selection=`** — single head. With an archive
  (`Archive("sigmoid_novelty")`) commits land on divergent lineages that
  never recombine; single-head stacks every accepted fix on one lineage.
- **Reflector and fusion completions run `thinking={"type": "disabled"}`.**
  On the edit-protocol prompt this model's reasoning runs away (measured:
  32,768 output tokens, zero visible text). Thinking-off returns a valid
  `<EDITS>` block in ~12s. The agent's own solve calls keep thinking on.

The reflector template asks for a **diagnosis before the edit** — classify
the failure as output formatting / arithmetic / comprehension, then make the
smallest generalising fix — and it states the grader's comparison rule
outright, because a reflector that has to guess the grader fixes the wrong
layer.

## What one run did

83 minutes wall clock: 185 rollouts, 988 reflector/fusion calls, 1 transient
call failure. Eight merger sweeps; every accepted commit was a
`ReflectiveFusion` synthesis of 7–9 concurrent proposals, taken through the
statistical gate and the L1 oracle re-check:

| sweep | held-out | event |
|---|---|---|
| 0 | 0.766 | — |
| 1 | 0.766 | 1 oracle-rejected |
| 2 | **0.794** | **commit** — synth of 7 proposals |
| 3 | 0.794 | 1 oracle-rejected |
| 4 | **0.804** | **commit** — synth of 8 proposals |
| 5 | 0.804 | 1 oracle-rejected |
| 6 | **0.822** | **commit** — synth of 9 proposals |
| 7 | 0.822 | budget exhausted |

```
merge synth(w0:6a7fd1d6 + w1:6c55d989 + w2:8d9c6a01 + w3:0c549fbd + …) -> dgm-agent   (sweep 2)
merge synth(w0:d2944aa8 + w1:88cac018 + w1:d78106da + w2:41380c5e + …) -> dgm-agent   (sweep 4)
merge synth(w0:61ca8769 + w1:53943c35 + w2:3c1c1e47 + w2:ef0695da + …) -> dgm-agent   (sweep 6)
```

## What it evolved into

The one-line `skills/strategy.md` grew into a **comprehension rulebook** —
the engine's own diagnosis of where this model actually loses points on
GSM-Hard. Excerpts, verbatim:

> For phrases like … 'three times more than X', interpret it as 'three times
> as many as X' (i.e., 3 × X), not 'X + 3×X'.

> The new value after applying a percentage rate that repeats every T units
> of time is computed by compounding: new_value = base_value *
> (1 + (percentage/100))^(number_of_periods).

> Do not multiply a price by a count unless the phrase explicitly says
> 'each' or 'per item'.

> When a problem asks 'How many will not be used' … compute the remainder as
> initial_total - (used_per_recipient * number_of_recipients). This
> remainder **may be negative** … and that negative value is the correct
> answer.

> Output the number as a plain decimal with full precision (do not round, do
> not add trailing zeros, do not add commas…).

…plus a restate-then-verify protocol (state the interpretation before
computing, re-read the question after). `agent.py` gained a retry loop
around the model call and — a caveat worth naming — one narrow hard-coded
rate-problem heuristic that slipped through the relaxed gate because its
held-out footprint happened to be positive. A looser gate admits more real
fixes *and* the occasional stowaway; the test number below includes both.

## What it was worth

| | held-out (gates on it) | test (never seen) |
|---|---|---|
| before | 0.766 | 0.692 |
| after | **0.822** (+6 tasks) | **0.757** (+7 tasks, +6.5pp) |

For calibration on this same split: evolving *only* a prompt-policy function
(the Gödel-agent port) reached 0.720, and a hand-designed
write-a-program-and-exec agent also reached 0.720. The evolved rulebook
beats both — and a per-item audit puts the honest ceiling near **0.879**,
because 13 of the 107 test items are corrupted upstream (the question's
substituted numbers disagree with the numbers the gold-computing reference
code used, so no faithful solver can match the gold).

Two conclusions to carry out of this case:

1. **Give evolution room instead of architecture.** Every designed variant —
   prescribed tool contracts, dual-path solvers with arbitration — measured
   at or below what the engine reached on its own from a bare agent. The
   headroom was in *comprehension rules*, and the engine found that without
   being told.
2. **Budget converts to staircase commits.** Three synthesis commits, each
   over the last, each clearing the gate and the L1 oracle — the shape that
   short runs on this workload never reached.

## Reproducing

All public surface: `evolve()` with `FileTree` + `code_runner` +
`tree_reflector` (custom diagnose-first template), `reflective_merge`,
`ReflectiveStaleness`, and the offline GSM-Hard sample
(`AGENTDESCENT_GSMHARD_SAMPLE=1`). Two environment notes that cost this
series real hours: construct the Anthropic client with `max_retries=0` (the
SDK's internal retries multiply with `with_retries` into ~45-minute stalls
on a slow endpoint), and give thinking models a generous `max_tokens` — at
1024 a thinking model returns **empty text** for every reflection and the
run silently proposes nothing.
