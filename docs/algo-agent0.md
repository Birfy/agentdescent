# Agent0 — Tool-integrated curriculum co-evolution

**Fidelity class: `inference_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This is a candidate-method port: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning", 2025 ([arXiv:2511.16043](https://arxiv.org/abs/2511.16043)) |
| Upstream code (pinned) | [aiming-lab/Agent0@f775b510](https://github.com/aiming-lab/Agent0/tree/f775b5101e62fe92976831adf4a21a38fcc0a767) |
| Definition | [`examples/agent0/agent0_tool_curriculum.py`](https://github.com/Birfy/agentdescent/blob/main/examples/agent0/agent0_tool_curriculum.py) |
| Domain | self-generated cart arithmetic with a sandboxed calculator; frozen evaluation carts |

## The mechanism

A Curriculum agent and an Executor agent, both from the same base model,
co-evolve in alternating iterations. The curriculum reward combines
**uncertainty** `1−2|p̂−0.5|` (executor self-consistency near 50%), a
**tool-use bonus** `min(N_tool, C)`, a BLEU repetition penalty, and a format
gate. The executor trains with **ADPO** (ambiguity-scaled advantages) on
majority-vote pseudo-labels, rolling out multi-turn with a sandboxed Python
interpreter in stop-and-go fashion.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Stop-and-go tool rollouts | request → AST-gated calculator → continue, on both training and frozen evaluation paths |
| Uncertainty + tool-use reward | both components surfaced in the curriculum update prompt |
| Frontier curriculum | `DifficultyWeighted` — the same curve as 1−2\|p̂−0.5\| |

## Boundaries

- Verbal policy memory replaces ADPO post-training.
- One calculator tool replaces the Python interpreter; no repetition penalty.

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
