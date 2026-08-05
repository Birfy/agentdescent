# ADAS — Meta Agent Search

Faithful port of the released implementation onto the AgentDescent engine.

| | |
|---|---|
| Kind | Harness self-evolution |
| Governance layer | L1 (`blast_radius=0.6`, oracle-gated) |
| Paper | "Automated Design of Agentic Systems", Shengran Hu, Cong Lu, Jeff Clune, 2024 ([arXiv:2408.08435](https://arxiv.org/abs/2408.08435); ICLR 2025) |
| Upstream code | https://github.com/ShengranHu/ADAS |
| Dataset (faithful) | MGSM — Multilingual Grade-School Math |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` keep-all archive with bootstrap-CI fitness |

## Run

```bash
python -m examples.adas.adas_meta_agent_search --dry-run
python -m examples.adas.adas_meta_agent_search --model claude-haiku-4-5 --generations 6
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`adas_meta_agent_search.py`](adas_meta_agent_search.py) — the runnable port
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-adas.md`](../../docs/algo-adas.md)
- Offline tests: [`tests/test_adas_example.py`](../../tests/test_adas_example.py)

All seven ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
