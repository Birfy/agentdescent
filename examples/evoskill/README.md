# EvoSkill — Automated Skill Discovery

Faithful port of the released implementation onto the AgentDescent engine.

| | |
|---|---|
| Kind | Skill library self-evolution |
| Governance layer | L2 (`blast_radius=0.2`) |
| Paper | "EvoSkill: Automated Skill Discovery for Coding Agents / Multi-Agent Systems", Salaheddin Alzubi et al., 2026 ([arXiv:2603.02766](https://arxiv.org/abs/2603.02766)) |
| Upstream code | https://github.com/sentient-agi/EvoSkill |
| Dataset (faithful) | OfficeQA — grounded reasoning over U.S. Treasury Bulletins |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` top-K frontier (sync) / SGD descent (async) |

## Run

```bash
python -m examples.evoskill.evoskill_skill_discovery --dry-run
python -m examples.evoskill.evoskill_skill_discovery --model claude-haiku-4-5
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`evoskill_skill_discovery.py`](evoskill_skill_discovery.py) — the runnable port
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-evoskill.md`](../../docs/algo-evoskill.md)
- Offline tests: [`tests/test_evoskill_example.py`](../../tests/test_evoskill_example.py)

All seven ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
