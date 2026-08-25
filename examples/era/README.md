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
| Fourth task | **LLM-SRBench** (arXiv:2504.10415, ICML 2025) — 240 scientific equation-discovery problems, someone else's benchmark and someone else's metrics |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` FUTS tree, `selection.FlatPuct` |

## Run

```bash
python -m examples.era.era_empirical_software --dry-run   # Kaggle S3E1, RMSE
python -m examples.era.era_hard_integrals --dry-run       # hard integrals, correct digits
python -m examples.era.era_hypergeometric --dry-run       # 2F1 vs a 25-digit reference
python -m examples.era.era_llm_srbench --dry-run          # LLM-SRBench equation discovery
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

**`era_llm_srbench.py` — LLM-SRBench.** The only one of the four scored on a
benchmark somebody else built, published and reported numbers for:
[LLM-SRBench](https://arxiv.org/abs/2504.10415) (ICML 2025 Oral), 240 scientific
equation-discovery problems in five subsets — 111 Feynman equations rearranged
into unfamiliar forms, and 129 synthetic problems in chemistry, biology, physics
and materials science, each with an out-of-distribution split. A candidate writes
`discover(x, y, spec)` and returns a **closed-form equation as a string**, which
is parsed and never executed: numeric constants, the problem's own variables,
`pi`/`e`, the four operators, powers and a fixed list of elementary functions.
That is what keeps it equation discovery — a nearest-neighbour table cannot be
written down as an equation — and it is also what keeps the held-out samples out
of reach, since the candidate hands back a formula rather than code that runs.
Scoring is the benchmark's own NMSE and Acc(0.1); the tree ranks on
`min(12, -log10(NMSE))` averaged over the problem set, because Acc(0.1) is an
indicator that is flat almost everywhere. The root node is sequentially
thresholded least squares over a fixed nonlinear library — SINDy's fitting step
without its domain-chosen library.

**The protocol is ERA's, not the benchmark's**, and that is the one thing to keep
straight when putting a number from here beside the paper's tables. LLM-SRBench
evaluates searchers that see one problem at a time with the data in context; here
the model never sees a sample, it writes one program, and that program is run
sandboxed over every problem. Same benchmark, same splits, same metrics,
different experiment — said in the result file, in the module docstring, and in
[`docs/algo-era.md`](../../docs/algo-era.md).

The samples come from an **ungated mirror** (`pkuHaowei/llm-srbench`), because the
benchmark's own release is gated and returns 401 without a HuggingFace token. The
mirror is checked rather than trusted: `tests/test_era_srbench.py` re-evaluates
every ground-truth expression that is evaluable as published against the samples
shipped beside it.

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
- [`era_llm_srbench.py`](era_llm_srbench.py) — the LLM-SRBench entry point
- [`_era_srbench.py`](_era_srbench.py) — download, shards, sandboxed evaluator, prompt (LLM-SRBench)
- [`_era_srbench_expr.py`](_era_srbench_expr.py) — the answer grammar and the benchmark's metrics
- [`_era_srbench_runner.py`](_era_srbench_runner.py) — the sandbox-side runner (LLM-SRBench)
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-era.md`](../../docs/algo-era.md)
- Offline tests: [`tests/test_era_example.py`](../../tests/test_era_example.py),
  [`tests/test_era_integrals.py`](../../tests/test_era_integrals.py),
  [`tests/test_era_hyp2f1.py`](../../tests/test_era_hyp2f1.py),
  [`tests/test_era_srbench.py`](../../tests/test_era_srbench.py)

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
against the kernel rather than by reading the profile back. All four tasks run
under the same profile — `sandbox_wrapper` — rather than under a copy of it.

All ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
