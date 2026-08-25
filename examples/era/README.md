# ERA — empirical-software search (Flat UCB tree search)

Faithful port of Google Research's released ERA implementation onto the
AgentDescent engine, running **four** tasks on one search.

| | |
|---|---|
| Kind | Program search over a scientific-computing task |
| Governance layer | L1 (`blast_radius=0.6`, sandbox-evaluated) |
| Paper | *An AI system to help scientists write expert-level empirical software* ([arXiv:2509.06503](https://arxiv.org/abs/2509.06503), Nature 2026) |
| Upstream code | https://github.com/google-research/era (commit `b836730`) |
| Dataset (faithful) | Kaggle Playground Series S3E1 — the upstream `implementation/playground_s3e1.py` task |
| Second task | *Numerical solution of integrals* — named in the paper's abstract; upstream released no implementation, so the nine-family suite is constructed here |
| Third task | Gauss hypergeometric `2F1` in double precision — not in the paper at all; 3000 points against a 25-digit mpmath reference, baseline `scipy.special.hyp2f1` |
| Fourth task | `scipy.special.pbdv` and `hyperu` — chosen by a 47-function precision sweep rather than from the literature; one tree per function |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` FUTS tree, `selection.FlatPuct` |

## Run

```bash
python -m examples.era.era_empirical_software --dry-run   # Kaggle S3E1, RMSE
python -m examples.era.era_hard_integrals --dry-run       # hard integrals, correct digits
python -m examples.era.era_hypergeometric --dry-run       # 2F1 vs a 25-digit reference
python -m examples.era.era_special_precision --function pbdv --dry-run    # one tree
python -m examples.era.era_special_precision --function hyperu --dry-run  # another
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## The four tasks

All four entry points run the *same* flat-PUCT tree search, the same aggregator, the
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

**`era_hypergeometric.py` — the Gauss hypergeometric function.** A candidate
writes `hyp2f1(a, b, c, z)` for real parameters over a wide declared range. No
leaderboard exists for this, so three other things carry the result: the problem
is hard on the standard survey's authority (Pearson, Olver & Porter, *Numerical
Algorithms* 74:821–866, 2017 — no single method covers the parameter space); the
baseline is `scipy.special.hyp2f1`, which every scientist already calls and
which loses more than six digits on about a third of the points; and the
reference is mpmath at 30 **and** 60 digits, kept only where the two agree to
25, committed as a file that `python -m tools.gen_hyp2f1_stress --check`
re-derives byte for byte. The suite is **3000 points** — a 1000-point acceptance
gate and 1000 held back — because per-point correct digits have a standard
deviation of 3.20, so an 80-point gate could not separate a half-digit gain from
noise. `mpmath`, `decimal` and `fractions` are off this
task's allowlist — the deliverable is a float64 routine.

**`era_special_precision.py` — the functions a sweep picked out.** The task
above chose `2F1` from a survey paper. This one chooses by measurement:
[`tools/scan_numeric_precision.py`](../../tools/scan_numeric_precision.py)
scores **47 NumPy and SciPy float64 entry points** against mpmath at 30 and 60
digits, over ranges declared before anything was run, and keeps only the points
where the two precisions agree.

Its main finding is that the libraries are mostly excellent — every NumPy
elementary function tested returns the full 16 digits, `sin` and `tan` included
at arguments up to 1e18, and around thirty SciPy entry points sit above 15. Two
do not, and they are this task's targets:

| target | mean digits | < 8 digits | < 1 digit |
|---|---|---|---|
| `scipy.special.pbdv` | 11.67 | 17.8% | 12.2% |
| `scipy.special.hyperu` | 14.36 | 3.0% | 2.8% |

Not rounding complaints. `pbdv` returns `4.81e100` at `v=19.83, x=-29.28` where
the value is `2.46e80`, and `-2.44e24` at `v=17.02, x=-14.61` where it is
`+6.01e15` — wrong sign, wrong magnitude, on a coherent region (large positive
order, negative argument). `hyperu` does not return a wrong number at all: on
3% of its range it returns **`nan`**, at points like `a=-15.82, b=-1.30,
x=23.10` where the function equals `2.45e17` and is perfectly well-conditioned.

`--function` picks the target and **each call is its own tree** — its own
stress set, root node, search and result file, sharing code and nothing else. A
single pooled score across both would let a gain on one hide a regression on
the other, and the claim being tested is per-function.

The stress sets use the sweep's *declared* ranges verbatim rather than
narrowing onto the failures the sweep found — a suite drawn around known
failures would measure the drawing. Sizes come from the measured variance:
per-point digits have an SD of 4.78 on `pbdv` and 2.83 on `hyperu`, so the
2000-point gate carries a standard error of 0.11 digits.

## What is in here

- [`era_empirical_software.py`](era_empirical_software.py) — the runnable port, and the search every task shares
- [`_era_support.py`](_era_support.py) — dataset, AST gate, sandbox, evaluator, prompt (S3E1)
- [`_era_runner.py`](_era_runner.py) — the stdlib-only script executed inside the sandbox (S3E1)
- [`_era_domain.py`](_era_domain.py) — what a task has to supply for the search to run on it
- [`era_hard_integrals.py`](era_hard_integrals.py) — the integrals entry point
- [`_era_integrals.py`](_era_integrals.py) — the nine integrand families, their closed forms, and the draw
- [`_era_integration.py`](_era_integration.py) — suite, sandboxed evaluator, prompt (integrals)
- [`_era_integration_runner.py`](_era_integration_runner.py) — the sandbox-side runner (integrals)
- [`era_hypergeometric.py`](era_hypergeometric.py) — the 2F1 entry point
- [`_era_hyp2f1.py`](_era_hyp2f1.py) — suite, sandboxed evaluator, prompt (2F1)
- [`_era_hyp2f1_runner.py`](_era_hyp2f1_runner.py) — the sandbox-side runner (2F1)
- [`data/hyp2f1_stress.json`](data/hyp2f1_stress.json) — the committed stress set and its references,
  produced by [`tools/gen_hyp2f1_stress.py`](../../tools/gen_hyp2f1_stress.py)
- [`era_special_precision.py`](era_special_precision.py) — the sweep-chosen targets, one tree per `--function`
- [`_era_special.py`](_era_special.py) — target registry, suite, sandboxed evaluator, prompt
- [`_era_special_runner.py`](_era_special_runner.py) — the sandbox-side runner (shared by every target)
- [`data/pbdv_stress.json`](data/pbdv_stress.json), [`data/hyperu_stress.json`](data/hyperu_stress.json) —
  committed stress sets, produced by [`tools/gen_special_stress.py`](../../tools/gen_special_stress.py)
- The sweep itself: [`tools/scan_numeric_precision.py`](../../tools/scan_numeric_precision.py),
  results in [`docs/data/numeric_precision_scan.json`](../../docs/data/numeric_precision_scan.json)
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-era.md`](../../docs/algo-era.md)
- Offline tests: [`tests/test_era_example.py`](../../tests/test_era_example.py),
  [`tests/test_era_integrals.py`](../../tests/test_era_integrals.py),
  [`tests/test_era_hyp2f1.py`](../../tests/test_era_hyp2f1.py)

## When the channel damages a reply

Measured on a hosted GLM-5.2 endpoint: about **one reply in five** of a few
thousand characters came back with bytes spliced into the middle of tokens
(`return val9.3192`). Identical through the Anthropic SDK, its streaming API and
a hand-rolled `urllib` request, while 25 fetches of a similar-sized file over
the same proxy hashed identically — so it is the endpoint, not any client here.

`--reply-attempts` (default 4) redraws a reply that **is not Python at all** —
it does not parse, or holds a character Python source cannot hold. A program
that is merely wrong, slow, fatal or gate-banned is never redrawn: it becomes a
node scoring `-inf`, exactly as upstream requires. Every run records
`reply_damage` beside its result.

## The one thing upstream does not ship

`implementation/sandbox.py` is an abstract class whose `run` raises
`NotImplementedError("Must provide a sandbox for executing untrusted code.")`.
This port provides it — Bubblewrap on Linux, Seatbelt on macOS — and refuses to
run on a host with neither. Because the benchmark requires `pandas`, `numpy`
and `scikit-learn`, the AST gate here is far weaker than the OpenEvolve port's,
so the sandbox rather than the gate is the boundary;
`test_the_sandbox_blocks_the_writes_and_network_it_claims_to_block` checks that
against the kernel rather than by reading the profile back. All three tasks run
under the same profile — `sandbox_wrapper` — rather than under a copy of it.

All ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
