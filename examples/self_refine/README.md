# Self-Refine — Iterative feedback refinement

Candidate-method port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Iterative feedback refinement |
| Fidelity class | `mechanism_microport` |
| Paper | "Self-Refine: Iterative Refinement with Self-Feedback", Madaan et al., 2023 ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)) |
| Upstream code | [madaan/self-refine@9a206d41](https://github.com/madaan/self-refine/tree/9a206d41e5d2d0c241bb441f41eeadb945afaa55) |
| Domain (compact) | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |
| Definition plug-ins | `ValidatedSlot` instruction; two-call FEEDBACK→REFINE proposal; reflective merge |

## Run

```bash
python -m examples.self_refine.self_refine_feedback_loop --dry-run
python -m examples.self_refine.self_refine_feedback_loop --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`self_refine_feedback_loop.py`](
  self_refine_feedback_loop.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-self-refine.md`](../../docs/algo-self-refine.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All candidate ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
