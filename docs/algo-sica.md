# SICA — Self-improving coding agent (real source edits)

> **Self-edit.** An archive of agent iterations; the best performer edits its own
> Python source and is re-benchmarked. Runs through the shared
> [`MethodPolicy`](policies.md) runner, behind an AST gate. Example:
> [`examples/sica/sica_self_edit.py`](https://github.com/Birfy/agentdescent/blob/main/examples/sica/sica_self_edit.py).

| | |
|---|---|
| **Paper** | *A Self-Improving Coding Agent* — Robeyns et al., 2025 ([arXiv:2504.15228](https://arxiv.org/abs/2504.15228)) |
| **Upstream code** | [MaximeRobeyns/self_improving_coding_agent@ed8275dc](https://github.com/MaximeRobeyns/self_improving_coding_agent/tree/ed8275dca4d3c5dbf77229964351fe9b424797dc) |
| **Example** | [`examples/sica/sica_self_edit.py`](https://github.com/Birfy/agentdescent/blob/main/examples/sica/sica_self_edit.py) |
| **Domain** | **GSM-Hard** ([`reasoning-machines/gsm-hard`](https://huggingface.co/datasets/reasoning-machines/gsm-hard)), 64/64/64 shuffled splits; one AST-gated policy function |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `self_edit_analogue` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

## The mechanism

SICA keeps an **archive of agent iterations**; each meta-iteration selects the
best performer to act as improver and base — at the pinned revision, by best
mean score with a confidence-interval recency tiebreak (the paper's
0.5/0.25/0.25 score/cost/time composite utility is not implemented upstream
either). The selected agent then edits its own source and is re-benchmarked.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Real self-edits | proposals are complete Python sources through an AST gate (function surface, arity, node whitelist, no builtins) |
| Utility gate | the framework's held-out acceptance gate |
| Archive base selection | `Archive('best')` — a deterministic argmax over the archive, as upstream's `idxmax()` is — driven by the population aggregator; the run finalises on the archive's best scorer |

## Boundaries

- The editable surface is one AST-gated function rather than SWE-bench Docker.
- Utility is the score alone, faithful to the pinned code rather than the paper's composite.

## Measured results — GSM-Hard

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge on (this method's own declaration), `deepseek-v4-flash` at
temperature 0.7, 64/64/64 shuffled splits. Recorded in
[`bench/results/sica-self-edit.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/sica-self-edit.json).

| seed | test quality | validation | accepted | invalid | calls |
|---|---|---|---|---|---|
| 0 | 0.703 → **0.750** | 0.562 → 0.625 | 3/80 | 3 | 942 |
| 1 | 0.625 → **0.625** | 0.578 → 0.625 | 0/80 | 1 | 880 |
| 2 | 0.594 → **0.625** | 0.641 → 0.719 | 2/80 | 1 | 876 |

Mean gain **+0.026** on test and +0.063 on validation, no seed regressing.

This row used to run `reflective=False`, and that was the right call while it
held: `ReflectiveFusion` reached the ledger **without** passing `to_diff`, so a
model-synthesised merge of two Python sources would have bypassed the AST gate
that makes executing them safe, and contested edits to the single policy slot
had to be resolved by **ranking** — which costs evaluations. `ReflectiveFusion`
now takes the strategy's validator and a synthesis that fails it returns `None`,
the existing fall-back-to-ranking signal, so the gate holds either way and the
flag is on.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

!!! note "Why the splits are shuffled, and why selection is `best`"
    **The splits are one deterministic shuffle**, not positional. GSM-Hard is
    derived from GSM8K in order and its tail is harder: taking train and held-out
    from the head and test from the tail put an **0.18 gap** between them on every
    seed before any candidate was proposed, so a method was tuned on one
    distribution and reported on another. `gsmhard_splits` draws all three from
    one shuffle; the residual gaps are −0.156 / +0.016 / +0.062, mixed sign.

    **`Archive('best')`, not `'performance'`.** `get_best_agent_iteration` takes
    `idxmax()` of the mean benchmark score and `runner.py` runs that agent's code
    — there is no sampling in it. A softmax over scores in `[0, 1]` leaves only
    `exp(1)/exp(0) = 2.7` between the best and worst entry, which over a
    four-candidate archive scoring 0.2 / 0.9 / 0.5 / 0.9 starts from the *worst*
    agent 8 times in 40. Ties go to the earlier entry, as `idxmax` does.

    **The editable surface is checked before a run, not after.**
    `test_the_gate_admits_a_prompt_that_can_clear_the_domain` compiles a policy
    that solves the domain and asserts the AST gate lets it through, so a 0.000
    here would be the algorithm's result and not the harness's —
    [Voyager](algo-voyager.md#measured-results-crafting-world) is what that
    failure looks like when nobody checks.

## Run it

```bash
python -m examples.sica.sica_self_edit --dry-run

# one seed of the three above
python -m examples.sica.sica_self_edit --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_sica_upstream.py`, `tests/test_candidate_methods.py`.
