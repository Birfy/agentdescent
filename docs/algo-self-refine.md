# Self-Refine — Iterative feedback refinement

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Self-Refine: Iterative Refinement with Self-Feedback", Madaan et al., 2023 ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)) |
| Upstream code (pinned) | [madaan/self-refine@9a206d41](https://github.com/madaan/self-refine/tree/9a206d41e5d2d0c241bb441f41eeadb945afaa55) |
| Definition | [`examples/self_refine/self_refine_feedback_loop.py`](https://github.com/Birfy/agentdescent/blob/main/examples/self_refine/self_refine_feedback_loop.py) |
| Domain | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |

## The mechanism

One model plays generator, feedback provider, and refiner: GENERATE an answer,
FEEDBACK on it (a separate call), REFINE from the critique — iterated on the
same instance up to four times, stopping early when the feedback contains the
stop signal (the pinned gsm runner checks for the literal "it is correct").
No training of any kind.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| FEEDBACK and REFINE as separate calls | a two-call proposal (`proposal_calls_per_candidate=2`) |
| The stop signal | feedback containing "it is correct" ends the refinement — the reserved call budget becomes an upper bound |
| Same-instance iteration | rollout → proposal → held-out rerun |

## Boundaries

- Fundamental analogue: upstream refines the *answer* to one instance; this port refines the *instruction artifact*.

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
