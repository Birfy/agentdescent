# Absolute Zero — Zero-data self-play (single model)

**Fidelity class: `inference_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Absolute Zero: Reinforced Self-play Reasoning with Zero Data", Zhao et al., 2025 ([arXiv:2505.03335](https://arxiv.org/abs/2505.03335)) |
| Upstream code (pinned) | [LeapLabTHU/Absolute-Zero-Reasoner@484afa48](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner/tree/484afa480c8f6fd77faa3d35451f24f287f58ee1) |
| Definition | [`examples/absolute_zero/absolute_zero_selfplay.py`](https://github.com/Birfy/agentdescent/blob/main/examples/absolute_zero/absolute_zero_selfplay.py) |
| Domain | self-generated cart arithmetic; frozen evaluation carts (deduction + abduction) |

## The mechanism

One model plays both **proposer and solver** over Python-verifiable task
triplets in three modes (deduction, abduction, induction), with a code executor
as the grounded verifier. The proposer's reward is **learnability**: `1−r̄` when
the solver's success rate r̄ is positive, zero at both extremes — monotone in
difficulty, not peaked at 50%. Both roles update the same weights via TRR++
(task-relative baselines per task-type × role).

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Proposer/solver self-play | training rollouts generate and solve their own carts, verified by the local renderer |
| Grounded verifier | the deterministic cart renderer; reward uses the domain's tolerant integer parse |
| Learnability reward `1−r̄` | surfaced verbatim in the proposer update prompt (deliberately **not** mapped to `DifficultyWeighted`, whose 4p(1−p) peaks at 0.5) |
| Frozen evaluation | held-out/test carts generated from the seed at build time — the evolved memory cannot shape its own test set |

## Boundaries

- Verbal policy memory replaces TRR++ weight updates.
- Deduction and abduction stand in for the paper's three task types; induction is omitted.

## Measured results

*Pending: this section is populated from the live matrix
(`bench/results/candidate-methods-framework-final.json`) after the
post-restructuring rerun. See the
[matrix overview](matrix-overview.md) for the matrix-wide
tables (quality, [parallel speedup](matrix-parallel-speedup.md), and
[async behaviour](matrix-async.md)).*

| Mode | Quality (test, before → after) | E2E seconds | Engine seconds | TTQ |
|---|---|---|---|---|
| serial | *TBD* | *TBD* | *TBD* | *TBD* |
| sync_parallel | *TBD* | *TBD* | *TBD* | *TBD* |
| async_pipeline | *TBD* | *TBD* | *TBD* | *TBD* |
