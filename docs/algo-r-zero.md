# R-Zero — Challenger/Solver co-evolution

**Fidelity class: `inference_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "R-Zero: Self-Evolving Reasoning LLM from Zero Data", Huang et al., 2025 ([arXiv:2508.05004](https://arxiv.org/abs/2508.05004)) |
| Upstream code (pinned) | [Chengsong-Huang/R-Zero@5699329d](https://github.com/Chengsong-Huang/R-Zero/tree/5699329d018d79535b7910abdedf5a6eebf355fd) |
| Definition | [`examples/r_zero/r_zero_challenger_solver.py`](https://github.com/Birfy/agentdescent/blob/main/examples/r_zero/r_zero_challenger_solver.py) |
| Domain | self-generated cart arithmetic; frozen evaluation carts (deduction + abduction) |

## The mechanism

Two copies of one base model co-evolve in alternating phases: the
**Challenger** is rewarded for questions at the Solver's frontier —
`min(p̂, 1−p̂)`, maximal when the Solver agrees with itself half the time —
minus a BLEU-cluster repetition penalty; the **Solver** trains on majority-vote
pseudo-labels filtered to an informative difficulty band (30–80% at the pinned
revision; the paper says 25–75%). Both are trained with GRPO.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Separate role updates | two plain-text `FieldSlots` keys with separate update calls |
| Uncertainty reward min(p̂,1−p̂) | two solver samples per generated task give an agreement rate, surfaced in the Challenger update |
| GRPO's group-relative shape | `AdvantageAcceptance` shifts the acceptance prior by group advantage |
| Frontier targeting | `DifficultyWeighted` — its 4p(1−p) weight shares peak and zeros with min(p̂,1−p̂) exactly |

## Boundaries

- Verbal role memories replace two GRPO-trained checkpoints.
- No BLEU repetition penalty; evaluation carts are frozen per seed.

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
