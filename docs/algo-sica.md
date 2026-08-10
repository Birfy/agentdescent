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
| Domain | deterministic integer-cents arithmetic; one AST-gated policy function |

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

## Measured results

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/sica-self-edit.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/sica-self-edit.json).

| seed | test quality | validation | accepted | calls |
|---|---|---|---|---|
| 0 | 0.000 → **0.562** | 0.000 → 0.812 | 3/80 | 599 |
| 1 | 0.000 → **0.188** | 0.000 → 0.625 | 1/80 | 1351 |
| 2 | 0.000 → **0.938** | 0.000 → 1.000 | 3/80 | 807 |

Mean 0.563, all three seeds moving. The call counts vary by more than a factor
of two because `reflective=False` is a *fidelity* choice here — a model-synthesised
merge of Python source would bypass the AST gate that makes executing it safe —
so contested edits to the single policy slot are resolved by **ranking**, and
ranking costs evaluations. The seed that accepted least spent most.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

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
