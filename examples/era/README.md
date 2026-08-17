# ERA — empirical-software search (Flat UCB tree search)

Faithful port of Google Research's released ERA implementation onto the
AgentDescent engine.

| | |
|---|---|
| Kind | Program search over a scientific-computing task |
| Governance layer | L1 (`blast_radius=0.6`, sandbox-evaluated) |
| Paper | *An AI system to help scientists write expert-level empirical software* ([arXiv:2509.06503](https://arxiv.org/abs/2509.06503), Nature 2026) |
| Upstream code | https://github.com/google-research/era (commit `b836730`) |
| Dataset (faithful) | Kaggle Playground Series S3E1 — the upstream `implementation/playground_s3e1.py` task |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` FUTS tree, `selection.FlatPuct` |

## Run

```bash
python -m examples.era.era_empirical_software --dry-run
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`era_empirical_software.py`](era_empirical_software.py) — the runnable port
- [`_era_support.py`](_era_support.py) — dataset, AST gate, sandbox, evaluator, prompt
- [`_era_runner.py`](_era_runner.py) — the stdlib-only script executed inside the sandbox
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-era.md`](../../docs/algo-era.md)
- Offline tests: [`tests/test_era_example.py`](../../tests/test_era_example.py)

## The one thing upstream does not ship

`implementation/sandbox.py` is an abstract class whose `run` raises
`NotImplementedError("Must provide a sandbox for executing untrusted code.")`.
This port provides it — Bubblewrap on Linux, Seatbelt on macOS — and refuses to
run on a host with neither. Because the benchmark requires `pandas`, `numpy`
and `scikit-learn`, the AST gate here is far weaker than the OpenEvolve port's,
so the sandbox rather than the gate is the boundary;
`test_the_sandbox_blocks_the_writes_and_network_it_claims_to_block` checks that
against the kernel rather than by reading the profile back.

All ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
