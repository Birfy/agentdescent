# OpenEvolve — program evolution

Faithful port of the released implementation onto the AgentDescent engine.

| | |
|---|---|
| Kind | Program evolution |
| Governance layer | L1 (`blast_radius=0.6`, sandbox-evaluated) |
| Paper | — (upstream is a released implementation, not a paper) |
| Upstream code | https://github.com/algorithmicsuperintelligence/openevolve (commit `411fb59`) |
| Dataset (faithful) | Function minimization — the upstream `examples/function_minimization` task |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` MAP-Elites islands |

## Run

```bash
python -m examples.openevolve.openevolve_program_evolution --dry-run
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`openevolve_program_evolution.py`](openevolve_program_evolution.py) — the runnable port
- [`_openevolve_support.py`](_openevolve_support.py) — sandbox, evaluator, and mutation helpers
- [`_openevolve_runner.py`](_openevolve_runner.py) — the stdlib-only script executed inside Bubblewrap
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-openevolve.md`](../../docs/algo-openevolve.md)
- Offline tests: [`tests/test_openevolve_example.py`](../../tests/test_openevolve_example.py)

All seven ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
