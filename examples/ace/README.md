# ACE — Agentic Context Engineering

Faithful port of the released implementation onto the AgentDescent engine.

| | |
|---|---|
| Kind | Skill / context self-evolution |
| Governance layer | L2 (`blast_radius=0.2`) |
| Paper | "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models", Qizheng Zhang et al., 2025 ([arXiv:2510.04618](https://arxiv.org/abs/2510.04618)) |
| Upstream code | https://github.com/ace-agent/ace |
| Dataset (faithful) | FiNER-139 — financial XBRL tagging (HF `nlpaueb/finer-139`) |
| `evolve()` plug-ins | `strategy=ACEPlaybook`; Curator = the default aggregator |

## Run

```bash
python -m examples.ace.ace_context_evolution --dry-run
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5 --async
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`ace_context_evolution.py`](ace_context_evolution.py) — the runnable port
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-ace.md`](../../docs/algo-ace.md)
- Offline tests: [`tests/test_ace_example.py`](../../tests/test_ace_example.py)

All seven ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
