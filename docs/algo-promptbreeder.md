# PromptBreeder — Prompt self-evolution (genetic)

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This is a candidate-method port: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution", Fernando et al., 2023 ([arXiv:2309.16797](https://arxiv.org/abs/2309.16797)) |
| Upstream code (pinned) | paper only — no official released code |
| Definition | [`examples/promptbreeder/promptbreeder_genetic_prompts.py`](https://github.com/Birfy/agentdescent/blob/main/examples/promptbreeder/promptbreeder_genetic_prompts.py) |
| Domain | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |

## The mechanism

PromptBreeder evolves a population of units — each a set of task-prompts plus a
**mutation-prompt** — with a binary tournament genetic algorithm: sample two
units, mutate the winner with one of nine operators (uniformly drawn), and
overwrite the loser. The self-referential move is **hyper-mutation**: the
mutation-prompt is itself rewritten by applying it to itself. Fitness is
measured on a 100-item training batch; the paper runs a population of 50 for
20–30 generations.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Binary tournament replication | `BinaryTournament(SelectionPolicy)` — declared at the selection seam; degenerate (single live head) until the engine grows multi-head support |
| Task/mutation-prompt unit | `FieldSlots` genome: two plain-text ledger keys that union-merge on disjoint edits and model-merge when contested |
| Nine mutation operators | three implemented (zero-order, first-order, hyper-mutation), rotated per replication event |
| Fitness on a training batch | the engine's held-out gate |

## Boundaries

- Population 50 and batch 100 are budget-sized down.
- Three of nine mutation operators; no few-shot workings-out in the unit.

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
