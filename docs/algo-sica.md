# SICA — Self-improving coding agent (real source edits)

**Fidelity class: `self_edit_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This is a candidate-method port: the mechanism is
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

*Pending: this section is populated from the live matrix
(`bench/results/candidate-methods-framework-final.json`) after the
post-restructuring rerun. See the
[candidate-method overview](candidate-results-overview.md) for the matrix-wide
tables (quality, [parallel speedup](candidate-parallel-speedup.md), and
[async behaviour](candidate-async.md)).*

| Mode | Quality (test, before → after) | E2E seconds | Engine seconds | TTQ |
|---|---|---|---|---|
| serial | *TBD* | *TBD* | *TBD* | *TBD* |
| sync_parallel | *TBD* | *TBD* | *TBD* | *TBD* |
| async_pipeline | *TBD* | *TBD* | *TBD* | *TBD* |
