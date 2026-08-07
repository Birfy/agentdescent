# Agent0 — Tool-integrated curriculum co-evolution

Candidate-method port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Tool-integrated curriculum co-evolution |
| Fidelity class | `inference_analogue` |
| Paper | "Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning", 2025 ([arXiv:2511.16043](https://arxiv.org/abs/2511.16043)) |
| Upstream code | [aiming-lab/Agent0@f775b510](https://github.com/aiming-lab/Agent0/tree/f775b5101e62fe92976831adf4a21a38fcc0a767) |
| Domain (compact) | self-generated cart arithmetic with a sandboxed calculator; frozen evaluation carts |
| Definition plug-ins | `ValidatedSlot` policy memory; `Policies(task_sampler=DifficultyWeighted)`; calculator stop-and-go rollouts; reflective merge |

## Run

```bash
python -m examples.agent0.agent0_tool_curriculum --dry-run
python -m examples.agent0.agent0_tool_curriculum --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`agent0_tool_curriculum.py`](
  agent0_tool_curriculum.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-agent0.md`](../../docs/algo-agent0.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All candidate ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
