# GEPA — Reflective Prompt Evolution

Faithful port of the released implementation onto the AgentDescent engine.

| | |
|---|---|
| Kind | Skill / prompt self-evolution |
| Governance layer | L2 (`blast_radius=0.2`) |
| Paper | "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", Lakshya A. Agrawal et al., 2025 ([arXiv:2507.19457](https://arxiv.org/abs/2507.19457)) |
| Upstream code | https://github.com/gepa-ai/gepa (also `dspy.GEPA`) |
| Dataset (faithful) | HotpotQA — multi-hop QA, distractor setting, exact match |
| `evolve()` plug-ins | `aggregator_factory=` Pareto optimizer |

## Run

```bash
python -m examples.gepa.gepa_prompt_evolution --dry-run
python -m examples.gepa.gepa_prompt_evolution --model claude-haiku-4-5
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`gepa_prompt_evolution.py`](gepa_prompt_evolution.py) — the runnable port
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-gepa.md`](../../docs/algo-gepa.md)
- Offline tests: [`tests/test_gepa_example.py`](../../tests/test_gepa_example.py)

All seven ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
