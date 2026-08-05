# SkillOpt — the ReflACT loop

Faithful port of the released implementation onto the AgentDescent engine.

| | |
|---|---|
| Kind | Skill document self-evolution |
| Governance layer | L2 (`blast_radius=0.2`) |
| Paper | "SkillOpt: Executive Strategy for Self-Evolving Agent Skills", Yifan Yang et al., 2025 ([arXiv:2605.23904](https://arxiv.org/abs/2605.23904)) |
| Upstream code | https://github.com/microsoft/SkillOpt (PyPI: `skillopt`) |
| Dataset (faithful) | SearchQA (`lucadiliello/searchqa`) — single-turn text QA, EM/F1 |
| `evolve()` plug-ins | `strategy` (bounded edits) + `aggregator_factory=` strict held-out gate |

## Run

```bash
python -m examples.skillopt.skillopt_skill_training --dry-run
python -m examples.skillopt.skillopt_skill_training --model claude-haiku-4-5
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`skillopt_skill_training.py`](skillopt_skill_training.py) — the runnable port
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-skillopt.md`](../../docs/algo-skillopt.md)
- Offline tests: [`tests/test_skillopt_example.py`](../../tests/test_skillopt_example.py)

All seven ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
