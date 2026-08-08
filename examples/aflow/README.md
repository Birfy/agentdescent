# AFlow — Agentic workflow search

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Agentic workflow search |
| Fidelity class | `mechanism_microport` |
| Paper | "AFlow: Automating Agentic Workflow Generation", Zhang et al., ICLR 2025 ([arXiv:2410.10762](https://arxiv.org/abs/2410.10762)) |
| Upstream code | [FoundationAgents/AFlow@3f457218](https://github.com/FoundationAgents/AFlow/tree/3f457218fc716093fe53f6df8a5d5e6379d66346) |
| Domain (compact) | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |
| Definition plug-ins | `FieldSlots` workflow (solve/review/modification keys); `Policies(selection=SoftMixed)`; reflective merge |

## Run

```bash
python -m examples.aflow.aflow_workflow_search --dry-run
python -m examples.aflow.aflow_workflow_search --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`aflow_workflow_search.py`](
  aflow_workflow_search.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-aflow.md`](../../docs/algo-aflow.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All MethodPolicy ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
