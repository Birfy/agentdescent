# Absolute Zero — Zero-data self-play (single model)

Candidate-method port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Zero-data self-play (single model) |
| Fidelity class | `inference_analogue` |
| Paper | "Absolute Zero: Reinforced Self-play Reasoning with Zero Data", Zhao et al., 2025 ([arXiv:2505.03335](https://arxiv.org/abs/2505.03335)) |
| Upstream code | [LeapLabTHU/Absolute-Zero-Reasoner@484afa48](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner/tree/484afa480c8f6fd77faa3d35451f24f287f58ee1) |
| Domain (compact) | self-generated cart arithmetic; frozen evaluation carts (deduction + abduction) |
| Definition plug-ins | `ValidatedSlot` policy memory; frozen per-seed evaluation splits; reflective merge |

## Run

```bash
python -m examples.absolute_zero.absolute_zero_selfplay --dry-run
python -m examples.absolute_zero.absolute_zero_selfplay --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`absolute_zero_selfplay.py`](
  absolute_zero_selfplay.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-absolute-zero.md`](../../docs/algo-absolute-zero.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All candidate ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
