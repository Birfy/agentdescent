# AFlow — Agentic workflow search

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This is a candidate-method port: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "AFlow: Automating Agentic Workflow Generation", Zhang et al., ICLR 2025 ([arXiv:2410.10762](https://arxiv.org/abs/2410.10762)) |
| Upstream code (pinned) | [FoundationAgents/AFlow@3f457218](https://github.com/FoundationAgents/AFlow/tree/3f457218fc716093fe53f6df8a5d5e6379d66346) |
| Definition | [`examples/aflow/aflow_workflow_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/aflow/aflow_workflow_search.py) |
| Domain | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |

## The mechanism

AFlow searches the space of code-expressed agentic workflows. Its selection is
**not UCT**: at the pinned revision it draws from the top-k scored workflows
plus the always-included seed with mixed probability
`λ·uniform + (1−λ)·softmax(α·(s−s_max))`, and keeps no visit counts
(`scripts/optimizer_utils/data_utils.py`). Expansion asks an optimizer LLM to
rewrite the selected workflow, with the parent's **experience** — its prior
modifications and whether each helped — injected into the prompt.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Soft mixed selection over top-k + seed | `SoftMixed(SelectionPolicy)`, driven by the population aggregator over the archive of committed workflows |
| Per-father experience | a per-parent modification log injected into the expansion prompt |
| Workflow as code | two fixed model nodes (Solve → ReviewAndRevise) as `FieldSlots` keys |
| Convergence over 20 rounds | the candidate budget |

## Boundaries

- Fixed two-node topology instead of code-level graph rewrites.
- Paper hyper-parameters α=0.4, λ=0.2 (the pinned code itself ships 0.2/0.3).

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
