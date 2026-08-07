# SICA — Self-improving coding agent (real source edits)

Candidate-method port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Self-improving coding agent (real source edits) |
| Fidelity class | `self_edit_analogue` |
| Paper | "A Self-Improving Coding Agent", Robeyns et al., 2025 ([arXiv:2504.15228](https://arxiv.org/abs/2504.15228)) |
| Upstream code | [MaximeRobeyns/self_improving_coding_agent@ed8275dc](https://github.com/MaximeRobeyns/self_improving_coding_agent/tree/ed8275dca4d3c5dbf77229964351fe9b424797dc) |
| Domain (compact) | deterministic integer-cents arithmetic; one AST-gated policy function |
| Definition plug-ins | `ValidatedSlot` with an AST-gate validator; `Policies(selection=Archive)` |

## Run

```bash
python -m examples.sica.sica_self_edit --dry-run
python -m examples.sica.sica_self_edit --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`sica_self_edit.py`](
  sica_self_edit.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-sica.md`](../../docs/algo-sica.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All candidate ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
