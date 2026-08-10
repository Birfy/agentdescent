# SICA — Self-improving coding agent (real source edits)

**Fidelity class: `self_edit_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "A Self-Improving Coding Agent", Robeyns et al., 2025 ([arXiv:2504.15228](https://arxiv.org/abs/2504.15228)) |
| Upstream code (pinned) | [MaximeRobeyns/self_improving_coding_agent@ed8275dc](https://github.com/MaximeRobeyns/self_improving_coding_agent/tree/ed8275dca4d3c5dbf77229964351fe9b424797dc) |
| Definition | [`examples/sica/sica_self_edit.py`](https://github.com/Birfy/agentdescent/blob/main/examples/sica/sica_self_edit.py) |
| Domain | **GSM-Hard** ([`reasoning-machines/gsm-hard`](https://huggingface.co/datasets/reasoning-machines/gsm-hard)); one AST-gated policy function |

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
| Archive base selection | `Archive('performance')`, driven by the population aggregator; the run finalises on the archive's best scorer |

## Boundaries

- The editable surface is one AST-gated function rather than SWE-bench Docker.
- Utility is the score alone, faithful to the pinned code rather than the paper's composite.

## Measured results — GSM-Hard

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge on, `deepseek-v4-flash` at temperature 0.7, 64/64/64 shuffled
splits. Recorded in
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

!!! danger "This row read as overfitting, and was mostly the split"
    On the first GSM-Hard run validation rose **+0.094 on all three seeds** —
    four standard deviations against the 0.021 sd of re-scoring the same split —
    and test did not move at all. The obvious reading is a gate that accepts what
    only works on the set it judges.

    It was not. The splits were **positional**: train and held-out from the head
    of the file, test from the tail. GSM-Hard is derived from GSM8K in order and
    its tail is harder, so the seed instruction scored 0.73 on held-out and 0.55
    on test — an 0.18 gap, the same sign on every seed, present before any
    candidate was proposed. A method was being tuned on one distribution and
    reported on another.

    `gsmhard_splits` now draws all three from one deterministic shuffle. The
    gaps are −0.156 / +0.016 / +0.062 — mixed sign, mean ≈ 0 — and the table
    above is that run. The earlier hypothesis, that the gate was accepting noise,
    was tested first and **refuted**: re-scoring one split five times moved it by
    0.02, not the 0.09 that would have been needed.

!!! danger "The next base was sampled where upstream takes the maximum"
    `get_best_agent_iteration` takes `idxmax()` of the mean benchmark score, and
    `runner.py` then runs `archive.agent_{best_iter}.agent_code`. There is no
    sampling in it.

    This port used `Archive(sampling="performance")` — a softmax over score,
    which at temperature 1 over scores in `[0, 1]` leaves only
    `exp(1)/exp(0) = 2.7` between the best and the worst entry. Measured over a
    four-candidate archive scoring 0.2 / 0.9 / 0.5 / 0.9, that mode starts from
    the **worst** agent 8 times in 40.

    This page had the rule right in prose — "selects by best mean score" — and
    named a mode that does something else. `Archive` gains a `best` mode, in the
    shared selection seam rather than inside this example, because a
    deterministic argmax over the archive is a published rule and belongs beside
    `performance`, `novelty` and the `uniform` ablation. Ties go to the earlier
    entry, as `idxmax` does, so a later candidate has to beat the incumbent
    rather than merely equal it.

!!! note "The gate was checked before the run, not after"
    A self-edit analogue whose editable surface cannot express a solution is
    [Voyager](algo-voyager.md#measured-results)'s failure in another shape: three
    seeds of 0.000 with no invalid proposals, against a target the world does not
    accept. `test_the_gate_admits_a_prompt_that_can_clear_the_domain` compiles a
    policy that teaches integer cents and working-out and asserts the gate lets
    it through, so a zero here would be the algorithm's and not the harness's.

## Run it

```bash
python -m examples.sica.sica_self_edit --dry-run

# the table above, one seed of the three (0, 1, 2)
python -m examples.sica.sica_self_edit --yes --seed 0 \
    --budget-rollouts 80 --workers 8 \
    --async --async-ratio 1 --max-seconds 3600 \
    --staleness full --temperature 0.7 --no-thinking \
    --provider claude --model deepseek-v4-flash
```

**`--async-ratio 1` is what this row ran at.** The flag was declared by the
shared parser and never passed to `run_port`, so the run took the runner's own
default of 1 while
[`bench/results/sica-self-edit.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/sica-self-edit.json)
recorded the 2 its command line had asked for. The flag is threaded now and that
file records 1 — see
[the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line)
for why the default here is 1 rather than the shared 3.

No `--reflective-merge`: the method's own `reflective` declaration set the merge,
and that declaration is a fidelity statement rather than a knob. `--max-seconds`
is the one setting the results file does not record; any value comfortably above
the row's `engine_s` leaves `--budget-rollouts` as the binding stop.

Offline tests: `tests/test_sica_upstream.py`, `tests/test_candidate_methods.py`.
