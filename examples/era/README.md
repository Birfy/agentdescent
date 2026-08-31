# ERA — empirical-software search (Flat UCB tree search)

Faithful port of Google Research's released ERA implementation onto the
AgentDescent engine, running **six** tasks on one search.

| | |
|---|---|
| Kind | Program search over a scientific-computing task |
| Governance layer | L1 (`blast_radius=0.6`, sandbox-evaluated) |
| Paper | *An AI system to help scientists write expert-level empirical software* ([arXiv:2509.06503](https://arxiv.org/abs/2509.06503), Nature 2026) |
| Upstream code | https://github.com/google-research/era (commit `b836730`) |
| Dataset (faithful) | Kaggle Playground Series S3E1 — the upstream `implementation/playground_s3e1.py` task |
| Second task | *Numerical solution of integrals* — named in the paper's abstract; upstream released no implementation, so the nine-family suite is constructed here |
| Third task | Gauss hypergeometric `2F1` in double precision — not in the paper at all; 3000 points against a 25-digit mpmath reference, baseline `scipy.special.hyp2f1` |
| Fourth task | **LLM-SRBench** (arXiv:2504.10415, ICML 2025) — the 240 scientific equation-discovery problems the benchmark released (its paper says 239), someone else's benchmark and someone else's metrics |
| Fifth task | [AlgoTune](https://github.com/oripress/AlgoTune) ([arXiv:2507.15887](https://arxiv.org/abs/2507.15887)) — 147 of its 154 tasks, scored in **speedup** over the task's own reference implementation, **one tree per task** |
| Sixth task | [SWE-bench Science](https://huggingface.co/datasets/OpenMOSS-Team/SWE-bench-Science) — 119 repository-scale tasks from scientific-computing projects, where a node is a **patch** and an expansion is a **coding-agent session**, **one tree per task** |
| `evolve()` plug-ins | `strategy` + `aggregator_factory=` FUTS tree, `selection.FlatPuct` |

## Run

```bash
python -m examples.era.era_empirical_software --dry-run   # Kaggle S3E1, RMSE
python -m examples.era.era_hard_integrals --dry-run       # hard integrals, correct digits
python -m examples.era.era_hypergeometric --dry-run       # 2F1 vs a 25-digit reference
python -m examples.era.era_llm_srbench --dry-run          # LLM-SRBench equation discovery
python -m examples.era.era_llm_srbench --dry-run --per-problem   # ... its own per-problem protocol
python -m examples.era.era_algotune --dry-run            # AlgoTune, speedup over the reference
python -m examples.era.era_swe_science --dry-run         # SWE-bench Science, patches by agent
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## The six tasks

All six entry points run the *same* flat-PUCT tree search, the same aggregator, the
same governance layer, and — for the five that evaluate a single file — the same
sandbox. What differs is a [`Domain`](_era_domain.py): the seed program, the
evaluator, the mutation prompt, and the name of the metric.

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

**`era_llm_srbench.py` — LLM-SRBench.** The only one of the five scored on a
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
evaluates searchers that evolve **per problem** — LLM-SR's own config gives each
problem 1 000 samples — while here the model never sees a sample, writes **one**
program, and that program is run sandboxed over every problem in the benchmark:
14 model calls for a 129-problem category, about 0.11 per problem. Same
benchmark, same splits, same metrics, different experiment — said in the result
file, in the module docstring, and in
[`docs/algo-era.md`](../../docs/algo-era.md), which also puts the two side by
side with the budget column attached.

The samples come from an **ungated mirror** (`pkuHaowei/llm-srbench`), because the
benchmark's own release is gated and returns 401 without a HuggingFace token. The
mirror is checked rather than trusted: `tests/test_era_srbench.py` re-evaluates
every ground-truth expression that is evaluable as published against the samples
shipped beside it.
**`era_algotune.py` — AlgoTune, and the other axis.** The four tasks above all
optimise *accuracy*. This one holds accuracy fixed and optimises **speed**: a
candidate writes `solve(problem)` and is scored by how much faster it is than the
task's own reference implementation, on problems the task's own `is_solution`
accepts. A solution the checker rejects scores nothing at all however fast it
was — AlgoTune's rule, and what stops a search from discovering that the fastest
SVD is the one that is not computed.

The baseline is not a strawman written for a benchmark: it is
`scipy.linalg.eig`, `scipy.integrate.solve_ivp` on a stiff problem,
`scipy.signal.upfirdn`, `scipy.spatial.Delaunay` — the call a working scientist
already makes. The root node of each tree *is* that reference, lifted out of its
`Task` class into a runnable program by
[`derive_seed_program`](_algotune_tasks.py), so a tree starts at exactly 1.0x and
every gain is measured against the library.

**One tree per task**, because they are separate searches over separate program
spaces: a factorisation trick found for `qr_factorization` is not a node in
`ode_stiff_vanderpol`'s tree and could not be selected there. Across tasks the
run reports the **geometric** mean, which is the mean a set of ratios has.

147 of AlgoTune's 154 tasks are runnable here, and the seven that are not are
named rather than silently skipped: `aes_gcm_encryption` and `chacha_encryption`
want `os.urandom` (and `os` is out, because the gate's forbidden-name check
cannot tell `os.urandom` from `os.system`), two references do not lift out of
their class, `lqr`'s own `is_solution` calls `float()` on a 1×1 array which
NumPy has refused since 1.25, and two need a package this repository does not
carry.

This used to be 72, and the gap was never the benchmark — it was this port's
dependency set. AlgoTune pins its own 156 packages; installing the ones its
*references* need and allowing what its own 2595 published solvers *import*
takes it to 147. Carried here: numpy, scipy, cvxpy (+ECOS/OSQP/HiGHS), jax, numba, cython, scikit-learn, networkx, OR-Tools, SymPy, mpmath, POT, PySAT, faiss, hdbscan, cryptography.
Each was ranked by what it alone unlocks, measured rather than guessed —
OR-Tools 13 tasks, networkx 9, scikit-learn 8, PySAT 2, SymPy 2, POT 2,
faiss 2, mpmath 1, hdbscan 1 — and each verified to import *and solve* inside
the sandbox. They have to land in `dist-packages` rather than the user site or
the bind cannot see them; OSQP (via jinja2), OR-Tools (via dateutil) and
jinja2 itself each hit that, and each failed with a message about the solver
rather than about the bind.

`--list-tasks` prints the runnable set; `--tasks all` runs it.

Problem sizes are **upstream's published ones**, read from AlgoTune's own
`reports/generation.json` (the `n` at which the reference took ~100 ms on the
machine that generated the dataset), so two runs of this port are comparable
without either of them calibrating against whatever host it landed on.

Measured, 20 tasks against GLM-5.2, 9 expansions per tree: **geometric-mean
held-back speedup 0.995x → 1.348x**, 16 of 20 improved, 9 at 1.10x or better.
The four largest are algorithm changes rather than flag-twiddling — 8.36x on
`wasserstein_dist` by replacing a general two-sample routine with the 1-D
closed form, 3.95x on `psd_cone_projection`, 2.21x on `ode_lorenz96_nonchaotic`
from the right-hand side alone at the reference's own tolerances, 2.05x on
`eigenvalues_real`. Full table, the caveats, and an independent repeat of the
same configuration: [`docs/algo-era.md`](../../docs/algo-era.md#measured-results--algotune).

**`--prior-exponent` — the model's own rating in `P(s,a)`.** Upstream ERA has no
prior: `futs.py` gives every node `1 / len(nodes)`, so the exploration budget is
spread evenly over a tree in which most nodes are dead ends. Above zero, the
mutation prompt asks for a `PROMISE: <n>` line rating the *approach after
tuning* 1–10, and the priors become `p^k / Σp^k`. Asking about the approach
rather than the draft is the point — asked the other way the rating collapses
into the score the evaluator already produces. The rating is read out of a reply
the port was already paying for, so it costs no extra call, and the default of
`0.0` reproduces upstream's uniform prior exactly.

Over 250 rated nodes it separates cleanly: 60 of the 137 rated ≥8 reached 2x,
against 3 of the 93 rated ≤6 (Fisher p=2.2e-13). It predicts speed and not
correctness, so on a task where `is_solution` rather than the clock is the
binding constraint it aims the budget at the wall — measurably worse there than
a uniform prior. On AlgoTune's eight most-published tasks it takes the harmonic
mean from 1.440x to 2.195x:
[`bench/results/era-algotune-model-prior.md`](../../bench/results/era-algotune-model-prior.md).

**`era_swe_science.py` — SWE-bench Science, and a mutation that is an agent.**
The five tasks above evolve *one file*, and a single model call can rewrite one.
These cannot: they are repository-scale — "repair an inconsistent
transition-state rotor workflow" across a 3 MB reaction-chemistry package. So a
node here is a **patch to a repository** (the release's own `model.patch`, `git
diff --cached --binary <baseline>`), the root of every tree is the *empty*
patch, and an expansion is a **coding-agent session**: Claude Code, Codex or any
command-line agent, run inside a checkout with the selected node's patch already
applied, able to read the whole tree and to *run* it — `run-in-env` executes
inside the task's own offline container, where its dependencies are installed.

The benchmark's reward is binary (the public reproduction exits 0 **and** every
private test passes), which is the number this port reports and the number the
search may not have: it is flat almost everywhere, and a tree that ranks by rank
cannot tell a patch that fixed four checks of five from one that deleted the
module. The tree ranks on a **pass rate** instead, over a private suite that is
split — `--held-back-frac` of each task's tests are drawn once from a seeded
shuffle and never shown to the search, in any shard or any prompt. The reported
reward comes from the release's own `/tests/grader.py`, unmodified, over the
whole suite.

**The protocol is ERA's, not the benchmark's**, as with LLM-SRBench and for the
same reason: on the leaderboard an agent gets one attempt per task and no
verifier feedback, while here a tree of agent sessions is scored between
attempts and the best node is kept. Same tasks, same pinned images, same grader,
different experiment — said in the result file, in the module docstring, and in
[`docs/algo-era.md`](../../docs/algo-era.md).

Measured, the six `--tasks default` tasks against the Claude Code CLI at 4
expansions per tree: **3 of 6 resolved by the release's own grader, from 0 of
6**, at `--feedback public` — the information the benchmark itself gives an
agent. The same configuration at `--feedback tests` resolves **6 of 6**, and the
difference is the point: pytest's traceback prints the body of the failing test,
so quoting the visible split's output hands the agent the hidden suite's
assertions. Half the resolutions were bought with the specification, and the
held-back split cannot detect that — it shows the fix generalised, not that the
spec was withheld.

A middle level, `--feedback counts` — how many visible checks failed, never
which — resolves **5 of 6**, and it is the one that says what the tree is worth:
**one tree in eighteen** ever improved on its own first round. `counts` mostly
buys a better *first* session rather than a useful second one, and a third to a
half of the later expansions never ran at all (endpoint failures), so this
under-tests the search rather than settling it. So the tables measure the agent
under this harness, not the search.
Full tables, the two defects in the published release this had to route around,
and the caveats:
[`docs/algo-era.md`](../../docs/algo-era.md#measured-results--swe-bench-science).

This is the one task that needs **Docker** rather than the shared sandbox: the
benchmark is distributed as 238 pinned `linux/amd64` images (about 1.6 GB per
task) and there is no offline substitute — the dependencies, the public fixtures
and the held-out tests all live in them. `--list-tasks` prints the release's
table; `--tasks unrestricted` is its own 96-task default selection, and the 23
tasks it gates on GPL-family, non-commercial or restricted-material licences
need `--allow-restricted-licenses`.

## What is in here

- [`era_empirical_software.py`](era_empirical_software.py) — the runnable port, and the search every task shares
- [`_era_support.py`](_era_support.py) — dataset, AST gate, sandbox, evaluator, prompt (S3E1)
- [`_era_runner.py`](_era_runner.py) — the stdlib-only script executed inside the sandbox (S3E1)
- [`_era_domain.py`](_era_domain.py) — what a task has to supply for the search to run on it
- [`era_hard_integrals.py`](era_hard_integrals.py) — the integrals entry point
- [`_era_integrals.py`](_era_integrals.py) — the nine integrand families, their closed forms, and the draw
- [`_era_integration.py`](_era_integration.py) — suite, sandboxed evaluator, prompt (integrals)
- [`_era_integration_runner.py`](_era_integration_runner.py) — the sandbox-side runner (integrals)
- [`era_algotune.py`](era_algotune.py) — the AlgoTune entry point, one tree per task
- [`_era_algotune.py`](_era_algotune.py) — task catalogue, sandboxed evaluator, prompt (AlgoTune)
- [`_era_algotune_runner.py`](_era_algotune_runner.py) — the sandbox-side runner (AlgoTune)
- [`_algotune_tasks.py`](_algotune_tasks.py) — the import shim and the reference-to-program transform
- [`era_hypergeometric.py`](era_hypergeometric.py) — the 2F1 entry point
- [`_era_hyp2f1.py`](_era_hyp2f1.py) — suite, sandboxed evaluator, prompt (2F1)
- [`_era_hyp2f1_runner.py`](_era_hyp2f1_runner.py) — the sandbox-side runner (2F1)
- [`data/hyp2f1_stress.json`](data/hyp2f1_stress.json) — the committed stress set and its references,
  produced by [`tools/gen_hyp2f1_stress.py`](../../tools/gen_hyp2f1_stress.py)
- [`era_swe_science.py`](era_swe_science.py) — the SWE-bench Science entry point
- [`_era_swe_science.py`](_era_swe_science.py) — release catalogue, Docker workspace, verifier, prompt (SWE-bench Science)
- [`_era_swe_science_runner.py`](_era_swe_science_runner.py) — the release's own grader plus a selection, run inside the verifier image
- [`era_llm_srbench.py`](era_llm_srbench.py) — the LLM-SRBench entry point
- [`_era_srbench.py`](_era_srbench.py) — download, shards, sandboxed evaluator, prompt (LLM-SRBench)
- [`_era_srbench_expr.py`](_era_srbench_expr.py) — the answer grammar and the benchmark's metrics
- [`_era_srbench_runner.py`](_era_srbench_runner.py) — the sandbox-side runner (LLM-SRBench)
- Port notes, upstream trace, and every recorded deviation: [`docs/algo-era.md`](../../docs/algo-era.md)
- Offline tests: [`tests/test_era_example.py`](../../tests/test_era_example.py),
  [`tests/test_era_integrals.py`](../../tests/test_era_integrals.py),
  [`tests/test_era_hyp2f1.py`](../../tests/test_era_hyp2f1.py),
  [`tests/test_era_srbench.py`](../../tests/test_era_srbench.py),
  [`tests/test_era_algotune.py`](../../tests/test_era_algotune.py),
  [`tests/test_era_swe_science.py`](../../tests/test_era_swe_science.py)

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
against the kernel rather than by reading the profile back. All five
single-file tasks run under the same profile — `sandbox_wrapper` — rather than
under a copy of it. SWE-bench Science is the exception and not a weakening of
the rule: its candidate is a patch to someone else's repository and its
evaluator is that release's own container, so the isolation there is the
benchmark's own pinned image with `--network none`, and the port refuses to run
where no Docker daemon answers.

All ports share one command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes`),
defined in [`examples/_common.py`](../_common.py) and enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
