# Reflexion — Verbal reinforcement / episodic memory

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Reflexion: Language Agents with Verbal Reinforcement Learning", Shinn et al., 2023 ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) |
| Upstream code (pinned) | [noahshinn/reflexion@218cf0ef](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0) |
| Definition | [`examples/reflexion/reflexion_episodic_memory.py`](https://github.com/Birfy/agentdescent/blob/main/examples/reflexion/reflexion_episodic_memory.py) |
| Domain | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |

## The mechanism

After a failed attempt, Reflexion converts the trajectory and the external
evaluator's signal into a **verbal reflection**, appends it to an episodic
memory, and retries the same task with that memory in context. The memory is
append-only and bounded to the last Ω entries (Ω=1–3 in the paper;
`memory[-3:]` in the pinned alfworld runs).

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Bounded append-only memory | `WindowedMemory`: commit-ordered content-addressed keys, rendered as the last 3 entries |
| Reflect on external feedback | the proposal call, fed the strict evaluator's feedback |
| Retry with memory | the engine's held-out rerun |
| Parallel reflections | appends never contradict, so they union-merge with no ranking evaluation |

## Boundaries

- Upstream retries the same failed instance; the held-out rerun is the analogue.
- The equal-budget design requests a reflection after every rollout, not only on failure.

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
