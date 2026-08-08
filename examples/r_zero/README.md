# R-Zero — Challenger/Solver co-evolution

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Challenger/Solver co-evolution |
| Fidelity class | `inference_analogue` |
| Paper | "R-Zero: Self-Evolving Reasoning LLM from Zero Data", Huang et al., 2025 ([arXiv:2508.05004](https://arxiv.org/abs/2508.05004)) |
| Upstream code | [Chengsong-Huang/R-Zero@5699329d](https://github.com/Chengsong-Huang/R-Zero/tree/5699329d018d79535b7910abdedf5a6eebf355fd) |
| Domain (compact) | self-generated cart arithmetic; frozen evaluation carts (deduction + abduction) |
| Definition plug-ins | `FieldSlots` dual-role memory; `Policies(acceptance=AdvantageAcceptance, task_sampler=DifficultyWeighted)`; reflective merge |

## Run

```bash
python -m examples.r_zero.r_zero_challenger_solver --dry-run
python -m examples.r_zero.r_zero_challenger_solver --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`r_zero_challenger_solver.py`](
  r_zero_challenger_solver.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-r-zero.md`](../../docs/algo-r-zero.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All MethodPolicy ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
