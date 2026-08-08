# Gödel Agent — Recursive runtime self-modification

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Recursive runtime self-modification |
| Fidelity class | `self_edit_analogue` |
| Paper | "Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement", Yin et al., 2024 ([arXiv:2410.04444](https://arxiv.org/abs/2410.04444)) |
| Upstream code | [Arvid-pku/Godel_Agent@bbb50879](https://github.com/Arvid-pku/Godel_Agent/tree/bbb508796be31c7140cdfc7106efd830a1324242) |
| Domain (compact) | deterministic integer-cents arithmetic; two AST-gated policy functions |
| Definition plug-ins | `ValidatedSlot` with an AST-gate validator; optional `--gateless` `AcceptancePolicy` |

## Run

```bash
python -m examples.godel_agent.godel_agent_self_modify --dry-run
python -m examples.godel_agent.godel_agent_self_modify --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`godel_agent_self_modify.py`](
  godel_agent_self_modify.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-godel-agent.md`](../../docs/algo-godel-agent.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All MethodPolicy ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
