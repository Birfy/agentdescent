# ERA — empirical-software search (Flat UCB tree search)

Faithful port of Google Research's released ERA implementation onto the
AgentDescent engine, running **two** of the paper's tasks on one search.

| | |
|---|---|
| Kind | Program search over a scientific-computing task |
| Governance layer | L1 (`blast_radius=0.6`, sandbox-evaluated) |
| Paper | *An AI system to help scientists write expert-level empirical software* ([arXiv:2509.06503](https://arxiv.org/abs/2509.06503), Nature 2026) |
| Upstream code | https://github.com/google-research/era (commit `b836730`) |
| Dataset (faithful) | Kaggle Playground Series S3E1 — the upstream `implementation/playground_s3e1.py` task |
| Second task | *Numerical solution of integrals* — named in the paper's abstract; upstream released no implementation, so the nine-family suite is constructed here |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` FUTS tree, `selection.FlatPuct` |

## Run

```bash
python -m examples.era.era_empirical_software --dry-run   # Kaggle S3E1, RMSE
python -m examples.era.era_hard_integrals --dry-run       # hard integrals, correct digits
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## The two tasks

Both entry points run the *same* flat-PUCT tree search, the same aggregator, the
same sandbox and the same governance layer. What differs is a
[`Domain`](_era_domain.py): the seed program, the sandboxed evaluator, the
mutation prompt, and the name of the metric.

**`era_empirical_software.py` — Kaggle Playground S3E1.** Upstream's own bundled
task, verbatim: the 80/20 head/tail split, the `train_and_predict(train_path,
test_path)` contract, RMSE, and the mutation prompt with its four speed
constraints. The candidate assembles a regression pipeline out of scikit-learn.

**`era_hard_integrals.py` — numerical solution of integrals.** The paper's sixth
demonstration, and the only one whose ground truth is arithmetic rather than a
leaderboard. A candidate writes `integrate(f, a, b)` and is handed a **black-box
scalar integrand** — no formula, no parameters, no family name — over `[0, 1]`,
`[0, inf)` or `(-inf, inf)`. Each problem set holds nine integrals, one from each
of nine difficulty classes, and every one of them has a **closed form**, so the
score is *correct significant digits against an exact value* (capped at 12) and
not agreement with a rival integrator. `scipy.integrate.quad` on defaults is the
root node: it solves several of the nine to machine precision and returns
confident nonsense on the rest.

Each problem also has a hard cap on calls to the integrand. That pairing is the
task — with no cap, the best program is whichever one is allowed to spend the
most, which is not a question about method.

## What is in here

- [`era_empirical_software.py`](era_empirical_software.py) — the runnable port, and the search every task shares
- [`_era_support.py`](_era_support.py) — dataset, AST gate, sandbox, evaluator, prompt (S3E1)
- [`_era_runner.py`](_era_runner.py) — the stdlib-only script executed inside the sandbox (S3E1)
- [`_era_domain.py`](_era_domain.py) — what a task has to supply for the search to run on it
- [`era_hard_integrals.py`](era_hard_integrals.py) — the integrals entry point
- [`_era_integrals.py`](_era_integrals.py) — the nine integrand families, their closed forms, and the draw
- [`_era_integration.py`](_era_integration.py) — suite, sandboxed evaluator, prompt (integrals)
- [`_era_integration_runner.py`](_era_integration_runner.py) — the sandbox-side runner (integrals)
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-era.md`](../../docs/algo-era.md)
- Offline tests: [`tests/test_era_example.py`](../../tests/test_era_example.py),
  [`tests/test_era_integrals.py`](../../tests/test_era_integrals.py)

## The one thing upstream does not ship

`implementation/sandbox.py` is an abstract class whose `run` raises
`NotImplementedError("Must provide a sandbox for executing untrusted code.")`.
This port provides it — Bubblewrap on Linux, Seatbelt on macOS — and refuses to
run on a host with neither. Because the benchmark requires `pandas`, `numpy`
and `scikit-learn`, the AST gate here is far weaker than the OpenEvolve port's,
so the sandbox rather than the gate is the boundary;
`test_the_sandbox_blocks_the_writes_and_network_it_claims_to_block` checks that
against the kernel rather than by reading the profile back. Both tasks run under
the same profile — `sandbox_wrapper` — rather than under a copy of it.

All ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
