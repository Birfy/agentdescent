# DGM — Darwin Gödel Machine

Faithful port of the released implementation onto the AgentDescent engine.

| | |
|---|---|
| Kind | Harness self-evolution |
| Governance layer | L1 (`blast_radius=0.6`, oracle-gated) |
| Paper | "Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents", Jenny Zhang et al., 2025 ([arXiv:2505.22954](https://arxiv.org/abs/2505.22954)) |
| Upstream code | https://github.com/jennyzzt/dgm |
| Dataset (faithful) | SWE-bench Verified (HF `princeton-nlp/SWE-bench_Verified`) |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` archive + parent selection |

## Run

```bash
python -m examples.dgm.dgm_self_improve
python -m examples.dgm.dgm_self_improve --generations 12 --archive keep_all
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`dgm_self_improve.py`](dgm_self_improve.py) — the runnable port
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-dgm.md`](../../docs/algo-dgm.md)
- Offline tests: [`tests/test_dgm_example.py`](../../tests/test_dgm_example.py)

All seven ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
