# Quickstart — evolve an agent's code

A complete, measured case: one `evolve()` call that improves a **running
program** — an agent whose `agent.py` is executed per task — on GSM-Hard,
against a hosted thinking model (`deepseek-v4-flash` behind an
Anthropic-shaped endpoint). Everything below is one real run: its
configuration, the diff the engine committed, and both numbers it produced —
including the one that did not move, and why.

## What evolves

The artifact is a two-file tree, executed for real on every rollout:

- `agent.py` — the agent. It asks the model to *write a Python program* that
  solves the problem, `exec`s that program, and falls back to plain prompting
  on any error. This file is the product of an earlier evolution run; here it
  is the **starting point**.
- `skills/strategy.md` — the instruction the agent prepends to every prompt.

Both are editable (`FileTree` strategy, at most 2 files per diff). A rollout
materialises the candidate into a workspace and runs
`python3 agent.py "<question>"` through `code_runner` — process isolation,
trimmed environment, hard timeout. The reward is exact-match on the last
number of stdout, normalised the way GSM-Hard's targets need
(`-9867630.0` ≡ `-9867630`).

## The call

```python
result = evolve(
    train + held_out, reward, run=run, propose=propose,
    strategy=FileTree(initial_files=INITIAL, max_files_per_diff=2),
    artifact_id="dgm-agent",
    blast_radius=0.6,                     # agent code is a harness: L1, oracle-gated
    n_workers=6, asynchronous=True, async_ratio=3,
    eval_concurrency=16,
    rounds=12, max_rollouts=72, patience=5, target_reward=0.95,
    self_verify=False, cheap_eval_tasks=6,
    agg_config=AggregatorConfig(bounded_gate=True, base_delta=0.8,
                                anneal_half_life=256, batch_trigger=4),
    policies=Policies(**reflective_merge(fusion_model, max_proposals=4)),
    staleness_policy=ReflectiveStaleness(),
    held_out_frac=0.5, shuffle=False, seed=0,
)
```

Four of these knobs are the lessons of this case, not defaults:

- **`base_delta=0.8, anneal_half_life=256`** — a relaxed, flat acceptance
  threshold. The improvements available here are small (a formatting bug
  costs 1–2 of 107 held-out tasks); under the default schedule the Beta
  posterior needs ~4–5 tasks of lift and correct fixes die at the gate.
- **`max_proposals=4` / `batch_trigger=4`** — reflective merge synthesises at
  most four competing proposals per model call; beyond that the merge prompt
  degrades into summarisation.
- **No `selection=`** — single head, deliberately. With an
  archive (`Archive("sigmoid_novelty")`), commits land on *divergent
  lineages* that never recombine, and the final winner-take-all pick dropped
  a correct formatting fix in the run before this one. Single-head stacks
  every accepted fix on one lineage.
- **The reflector and fusion completions run with
  `thinking={"type": "disabled"}`.** On the edit-protocol prompt this model's
  reasoning runs away — measured: 32,768 output tokens of thinking and zero
  visible text. Thinking-off returns a valid `<EDITS>` block in ~12s. The
  *agent's own* solve calls keep thinking on; the split is per-completion.

The reflector's template asks for a **diagnosis before the edit**: classify
the failure as (a) output formatting, (b) arithmetic, or (c) comprehension,
then make the smallest edit that fixes that class. It also states the grader's
actual comparison rule (`4561195.20` does **not** match `4561195.2`), because
a reflector that has to guess the grader proposes fixes for the wrong layer.

## What one run did

Data: 320 GSM-Hard problems, split 106 train / 107 held-out / 107 test
(seeded shuffle; the engine never sees test). Wall clock **83 minutes**:
72 rollouts, 441 reflector/fusion calls, 0 call failures.

| sweep | held-out | committed | rejected |
|---|---|---|---|
| 0 | 0.785 | 0 | 0 |
| 1 | **0.804** | **1** | 0 |
| 2 | 0.804 | 0 | 1 (oracle) |
| 3 | 0.804 | 0 | 0 |

The ledger records what the one commit was:

```
merge synth(w0:ecb0561e + w1:24a27b37 + w1:f15d32cf + w2:253ef29e + w2:7e3d6ef3 + w3:791b4fee) -> dgm-agent
```

— six workers independently diagnosed output-formatting failures, conflict
resolution handed the contradicting rewrites to `ReflectiveFusion`, and one
synthesised candidate carried them through the gate and the L1 oracle
re-check. One later candidate was oracle-rejected; the run then stopped on
budget.

## The committed diff

The synthesis added two functions to `agent.py` and wired them into the
exec path — nothing else changed:

```python
def finalize_output(out):
    """Find final Answer line and ensure it contains a numeric literal, not a fraction expression."""
    m = re.findall(r"Answer:\s*(.*)", out)
    if not m:
        return ""
    raw = m[-1].strip().split()[0]
    # If it's a fraction like a/b, evaluate it as a float
    if "/" in raw and not re.match(r"^[-]?\d+$", raw):
        try:
            num, den = raw.split("/")
            val = float(num) / float(den)
            return "Answer: " + repr(val)
        except Exception:
            return ""
    return "Answer: " + raw


def extract_answer(out):
    """Extract the last number from any output, removing extra 'Answer:' prefixes."""
    m = re.findall(r"Answer:\s*(-?[\d,.]+)", out)
    if m:
        return "Answer: " + m[-1].replace(",", "")
    # fallback: last numeric token
    nums = re.findall(r"-?[\d,]+\.?[\d]*", out)
    if nums:
        return "Answer: " + nums[-1].replace(",", "")
    return ""
```

This is exactly the fix the reflector's diagnoses called for: the agent's
generated programs sometimes print a raw fraction (`1422094/3`) or a doubled
`Answer: Answer:` prefix, and the grader — which reads the last number —
scores those zero even though the computed value is right.

## What it was worth — both numbers

| | held-out (gates on it) | test (never seen) |
|---|---|---|
| before | 0.785 | 0.701 |
| after | **0.804** (+2 tasks) | 0.701 (±0) |

Held-out improved; test did not. That is not the fix failing to generalise —
it is **footprint**. The fraction-output pathology this diff repairs occurs in
this dataset a handful of times, and those occurrences happen to sit in the
held-out split; test's two formatting losses are a *different* sub-bug
(trailing zeros from `%.2f`-style printing, `6532906.40` vs `6532906.4`) that
was proposed in the same run but fell past the budget. A fix whose true value
is 1–2 tasks per 107 shows up only in the split that contains its cases; the
measurement floor of a 107-task split is about ±2 tasks of backend
nondeterminism even at `temperature=0`.

Two honest conclusions to carry out of this case:

1. **The machinery works end-to-end at this scale**: diagnose → targeted
   edit → conflict resolution → reflective synthesis → statistical gate →
   L1 oracle → one lineage. Every accepted change is a real repair with a
   named cause, and every rejection has a reason you can read in the ledger.
2. **Effect size has to clear measurement granularity before test moves.**
   Small correct fixes accumulate across rounds; judging them one run at a
   time on a 107-task split under-counts them. Widen the split, average
   seeds, or let the run continue — the trailing-zero fix was already in the
   proposal stream when the budget ended.

## Reproducing

The pieces are all public surface: `evolve()` with `FileTree` +
`code_runner` + `tree_reflector` (custom template), `reflective_merge`,
`ReflectiveStaleness`, and the offline GSM-Hard sample at
`examples/_gsmhard_sample.json` (`AGENTDESCENT_GSMHARD_SAMPLE=1`). Two
environment notes that cost this series real hours: construct the Anthropic
client with `max_retries=0` (the SDK's internal retries multiply with
`with_retries` into ~45-minute stalls on a slow endpoint), and give thinking
models a generous `max_tokens` — at the runner default of 1024 a thinking
model returns **empty text** for every reflection and the run silently
proposes nothing.
