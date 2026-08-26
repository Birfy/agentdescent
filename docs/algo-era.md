# ERA — Empirical-software search (Flat UCB tree search)

> **Program search, tree-shaped.** A Python solution to a scientific-computing
> task is the artifact: a model rewrites a selected node, a sandboxed evaluator
> supplies RMSE, and a **flat PUCT tree** — every node selectable, exploitation
> by rank rather than by score — decides what to expand next. Runs through
> [`evolve()`](evolution.md) / `async_evolve()` with a custom `Strategy` +
> `aggregator_factory` at **L1** governance. Example:
> [`examples/era/era_empirical_software.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_empirical_software.py).

| | |
|---|---|
| **Paper** | *An AI system to help scientists write expert-level empirical software*, [arXiv:2509.06503](https://arxiv.org/abs/2509.06503) (Nature, 2026) |
| **Upstream code** | [google-research/era@b836730](https://github.com/google-research/era/tree/b836730b5c000526af95116b1d0e2c60c8cf0a10), `implementation/futs.py` + `implementation/playground_s3e1.py` |
| **Example** | [`examples/era/era_empirical_software.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_empirical_software.py) |
| **Domain** | Kaggle Playground Series S3E1 (synthetic California housing), RMSE — upstream's own bundled task |
| **Second task** | [`examples/era/era_hard_integrals.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_hard_integrals.py) — the paper's *numerical solution of integrals*, scored in correct significant digits |
| **Third task** | [`examples/era/era_hypergeometric.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_hypergeometric.py) — `2F1` in double precision, against a 25-digit mpmath reference |
| **Fourth task** | [`examples/era/era_llm_srbench.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_llm_srbench.py) — [LLM-SRBench](https://arxiv.org/abs/2504.10415) equation discovery, on the benchmark's own metrics |
| **Layer** | L1 program (`blast_radius=0.6`, AST-gated and sandbox-isolated) |
| **Fidelity** | `benchmark_faithful` — [what the classes mean](port-fidelity.md) |

Port author: `chendanyang`.

## The algorithm

FUTS is 155 lines upstream and the whole of it is one loop:

1. **Rank.** Sort every node by score; `rank_score = rank / (N − 1)`, so the
   worst node is 0 and the best is 1. A lone node is 0.5.
2. **Score.** `puct = rank_score + c_puct · (1/N) · √(Σvisits) / (1 + visits)`.
3. **Select.** `argmax(puct)` over **all** nodes — there is no descent from the
   root, which is what "flat" names.
4. **Expand.** The model rewrites the selected node's program; the sandbox runs
   it; the resulting score makes a new node whose parent is the selected one.
5. **Backpropagate.** The new node and every ancestor take one visit.

Two choices in there are doing the work, and both differ from ordinary UCT:

**Exploitation is a rank, not a value.** Scores enter the formula only through
their order, so the exploration constant means the same thing whether the metric
is RMSE, log-likelihood or accuracy — and one bad candidate scoring `-inf`
cannot swamp the term the way a raw value would.

**The prior is uniform.** AlphaZero's `P(s, a)` needs a policy network to say
which sibling is promising. There is none here, so `P = 1/N` and the exploration
term reduces to a visit-starvation bonus.

## Algorithm mapping

| ERA mechanism | AgentDescent representation |
|---|---|
| `futs.search`'s select step | [`selection.FlatPuct`](selection.md), a shipped `SelectionPolicy` |
| `futs.Node` list | `EraTree`, the aggregator's shared archive |
| `futs.Solution` (a program string) | `Task` rollouts over a source-code artifact |
| `PlaygroundGenerator.__call__` | `propose(rendered, task, output, reward)` |
| Full program replacement | `EraStrategy.to_diff()` |
| `PlaygroundExecutor.__call__` | `run()` plus `reward_program()` |
| `Sandbox.run` (upstream: `NotImplementedError`) | `_era_support.sandbox_command` — Bubblewrap / Seatbelt |
| `num_iterations` | `evolve(rounds=iterations // workers)`, `--budget-rollouts` |
| Concurrent expansions | `evolve(max_concurrency=...)` |
| Completion-order commits | `async_evolve(async_ratio=...)` |
| Best node so far | AgentDescent `Ledger` dev head |

The artifact is generated executable code, so the port declares
`blast_radius=0.6` and is classified as an L1 change. The RMSE evaluator remains
the acceptance authority, as it is upstream.

## How it plugs into `evolve()`

```python
result = evolve(
    build_tasks(shards),               # one task per held-out shard
    reward_program,                    # 1 / (1 + RMSE), from the runner payload
    run=make_run(...),                 # train in the sandbox, predict one shard
    propose=make_propose(tree, ...),   # FUTS select -> mutation prompt -> program
    strategy=EraStrategy(),            # the program is the artifact
    aggregator_factory=factory,        # EraTreeAggregator: execute, append, commit
    blast_radius=0.6,
    rounds=iterations // workers,      # total expansions fixed as workers vary
)
```

Four plug-ins, and one thing deliberately *not* reused:

* **`strategy=EraStrategy`** — a single-slot program artifact. Not `SingleSlot`,
  because `to_diff` has to carry the parent's tree index alongside the code:
  where a node attaches is part of the algorithm, not metadata.
* **`aggregator_factory=`** — `EraTreeAggregator` owns the tree. It re-executes
  every surviving card against the held-out shards, appends the node, and commits
  the best-scoring program to the `dev` head.
* **`selection.FlatPuct`** — the shipped policy, called by the tree under its own
  lock so the visit reservation and the pick are one atomic step.
* **`reward_program`** is custom rather than one of
  [`agentdescent.rewards`](rewards.md): those score a *text answer* against a
  gold string, and this scores a vector of predictions against a vector of
  truths. Reaching for `numeric_close` here would have meant scoring the
  candidate's printed output rather than its predictions.

## Fidelity and boundaries

Preserved mechanics:

1. The PUCT formula, `c_puct = 1.0`, the rank normalisation including the
   single-node 0.5 case, the uniform prior, and visits backpropagated up the
   parent chain. Pinned by
   `tests/test_era_example.py::test_rank_scores_match_the_upstream_unit_test`
   and `::test_puct_matches_the_upstream_unit_test`, which are upstream's own
   `futs_test.py` fixtures.
2. **A node is appended for every expansion, including a failed one.** Upstream
   returns `float('-inf')` from `PlaygroundExecutor` when the sandbox fails and
   appends the node anyway; dropping it would change the rank denominator and
   the prior on every later iteration.
3. The task: Playground S3E1, the 80/20 head/tail split of `train.csv`, the
   `MedHouseVal` target dropped from what the candidate reads,
   `train_and_predict(train_path, test_path)`, RMSE scored on the host, and the
   mutation prompt — including the ban on `xgboost`/`lightgbm` and the three
   speed constraints.
4. `score = -RMSE`, because FUTS maximises. The engine's `[0, 1]` reward is
   `1 / (1 + RMSE)`, which is strictly decreasing in RMSE and therefore induces
   *exactly* the ranking `-RMSE` does — the tree and the acceptance gate cannot
   disagree about which of two programs is better.

Intentional differences:

1. **Upstream ships no sandbox.** `implementation/sandbox.py` is an abstract
   class whose `run` raises `NotImplementedError("Must provide a sandbox for
   executing untrusted code.")`. This port supplies one and refuses to run
   without it. See the boundary note below — it is the most important thing on
   this page.
2. Upstream reports one RMSE over its whole 20% tail, which is also the split it
   optimised against. Here the tail is cut into equal contiguous shards; the
   first `--shards` are what the search can score and the last `--test-shards`
   are never shown to it, so the reported number is on unseen rows.
3. A shard is one AgentDescent task, so `run()` trains the current program and
   predicts one shard. Upstream has a single train-and-predict per candidate;
   this makes the same candidate measurable per-task, which is what the engine's
   held-out gate and `eval_concurrency` need.
4. Upstream is serial. With N workers a visit is reserved at **selection** time
   rather than after execution — see below.
5. Candidate threads are pinned to one (`OMP_NUM_THREADS=1` and friends), because
   `RLIMIT_CPU` counts CPU seconds across threads: an OpenBLAS that helpfully
   starts eight would burn a 60-second budget in eight wall-clock seconds and the
   candidate would be killed for being fast. Upstream sets no thread policy and
   has no CPU limit to protect.

### The visit reservation, and why it is not a semantics change

Upstream backpropagates the new node's visit *after* `execute_fn` returns. This
port increments the selected node and its ancestors at *selection*, and gives
the inserted node `num_visits = 1` without re-walking the chain.

With one proposal in flight, nothing can observe the tree between those two
points, so every selection sees identical visit counts — the two are the same
algorithm. `tests/test_era_example.py::test_serial_tree_reproduces_upstream_futs`
pins that: it drives this port's tree and a line-by-line transcription of
`futs.search` with the same mock generator and executor, and asserts the same
node is expanded at every step, with the same final visit vector.

With N in flight it is the standard parallel-MCTS virtual loss, and it is the
minimum needed for N workers to mean anything: without it, `argmax(puct)` is
deterministic and every worker in a batch would be handed the same parent.

!!! danger "The AST gate is not the boundary here, and must not be read as one"
    The OpenEvolve port's gate allows six standard-library modules, so the gate
    and the sandbox are two real layers. This benchmark *requires*
    `pandas`, `numpy` and `scikit-learn` — a stack that can read files and spawn
    processes — so admitting it admits most of what a gate would otherwise stop.

    What the gate still buys is that the ordinary accidents (a candidate that
    shells out, calls `open`, or reaches for a dunder) fail in-process with a
    readable message. What actually confines a candidate is the sandbox:
    **Bubblewrap** (`bwrap`) on Linux, **Seatbelt** (`sandbox-exec`) on macOS,
    both denying network access and confining writes to a scratch directory,
    with CPU / address-space / file-size / fd / process limits from `setrlimit`
    inside the runner. A host with neither backend raises rather than running
    model-written code unconfined.

    Unlike the OpenEvolve port, the Bubblewrap profile here binds the root
    **read-only** rather than a handful of directories: the candidate's imports
    live wherever the interpreter was installed, and enumerating those would be
    a guess that fails differently on every host. Reads are therefore open on
    both platforms; writes and the network are not. That is a weaker boundary
    than OpenEvolve's and it is stated rather than glossed.

    It is checked against the kernel rather than by reading the profile back:
    `test_the_sandbox_blocks_the_writes_and_network_it_claims_to_block` runs a
    probe under the real profile and asserts a write outside the scratch
    directory fails, a write inside it succeeds, and `socket.create_connection`
    cannot reach the network.

## The second task — numerical solution of integrals

The ERA abstract lists six demonstrations. Five are scored against a leaderboard
or a held-out dataset; the sixth is not:

> "ERA also produced expert-level software for geospatial analysis, neural
> activity prediction in zebrafish, and **numerical solution of integrals**"

Upstream released `futs.py` and one task, `playground_s3e1.py`; there is no
integrals implementation to port. So what is faithful here is the **search** —
and the nine-family suite below is this repository's construction, stated as
such wherever it is reported. It lives in
[`examples/era/era_hard_integrals.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_hard_integrals.py),
and it runs on **the same search**: same flat-PUCT tree, same visit reservation,
same aggregator, same governance layer, same sandbox profile. The seam is a
[`Domain`](https://github.com/Birfy/agentdescent/blob/main/examples/era/_era_domain.py)
— seed program, sandboxed evaluator, mutation prompt, metric name — and nothing
algorithmic lives in it. `domain=None` is upstream's Kaggle task, so a caller
who names no domain gets the port upstream ships.

### What the candidate is asked for

```python
def integrate(f, a, b):     # a may be -inf, b may be +inf
    ...                     # -> one float
```

`f` is a **black box**: a scalar function of one float, with no formula, no
parameters and no family name attached. Nine integrals make a problem set, one
from each of nine difficulty classes:

| Class | What it breaks |
|---|---|
| algebraic singularities at **both** endpoints | boundedness, twice |
| logarithmic singularity of integer power | boundedness, and every derivative |
| oscillation accumulating at an endpoint | any fixed node count near the corner |
| interior peak of width 1e-7 to 1e-4 | adaptive subdivision that never samples it |
| barely damped oscillation on `[0, inf)` | tail truncation |
| oscillation that **never** decays on `[0, inf)` | tail truncation, harder |
| endpoint singularity × fast oscillation, damped | both at once |
| cancellation over `(-inf, inf)` | relative accuracy, by 4 orders of magnitude |
| endpoint singularity **and** a heavy tail | one substitution cannot fix both |

### Why the score means something

**The reference is a closed form, not another integrator.** Every family has an
exact value in terms of `Γ`, `atan`, `π/sin(πs)` and friends, so a candidate is
scored against arithmetic rather than against a rival method it might legitimately
beat. `tests/test_era_integrals.py::test_the_closed_form_matches_high_precision_quadrature`
checks each identity against mpmath at 30 digits — through a per-family
substitution, because mpmath's own quadrature on the raw integrand disagrees
with the closed form in the third decimal place on the Fresnel family. That
disagreement is the benchmark working.

**The metric is correct significant digits**, `min(12, -log10(relative error))`,
averaged over the problem set. The cap is the precision of the references
themselves: reporting 15 digits against a `math`-library reference would be
measuring the reference's rounding. A problem that raises, returns `nan` or
overruns its budget scores 0 and the rest of the set still counts — a quadrature
suite is nine independent facts, and partial credit is what makes the tree's
ranking informative.

**Every problem has a hard cap on calls to the integrand** (200,000 by default,
enforced in the runner, not trusted to the candidate). Without it the best
program is whichever one is allowed to spend the most, which is not a question
about method. `quad` on defaults uses ~1,000.

**The family is never named to the model.** The prompt describes the *classes*,
which is the task description an expert would be handed; the candidate sees a
callable and two limits; and the failure report the search feeds back carries the
interval and the digit count but not the family. Problems are shuffled within a
shard, so position is not family either. Together those are what keep the task
"write a quadrature rule" rather than "write a dispatch table".

### Deviations this task adds

1. **The gate is loosened in one place and narrowed in another.**
   `literal_top_level=False` admits computed module-level constants, because a
   Gauss-Legendre node table built once at import is ordinary numerics; the
   import allowlist drops `pandas`/`scikit-learn` and adds `cmath`. The sandbox
   is the boundary either way, and module-level work runs under the same CPU
   limit as everything else.
2. **The problem file is copied into the sandbox scratch** before the runner is
   started. The Bubblewrap profile mounts a fresh tmpfs over `/tmp`, so a suite
   living there would be invisible inside the sandbox and the candidate would be
   blamed for a `FileNotFoundError`.
3. **The problems are drawn, not downloaded.** A shard is a seeded draw from the
   catalogue, reproducible from `--seed` alone, written once under the
   dataloader cache. There is no dataset to fetch and no network in the loop.
4. `score = mean digits` with no sign flip — FUTS maximises and more digits is
   better — and the engine's `[0, 1]` reward is `mean_digits / 12`, which is
   again exactly order-preserving with what the tree ranks on.

## The third task — 2F1, and what replaces a leaderboard

The two tasks above are scored against a dataset and against arithmetic. The
third is scored against **an independent arbitrary-precision computation**, and
it exists to answer a specific objection: *a benchmark nobody else has run
proves nothing, because the people who built it also set the bar.*

[`examples/era/era_hypergeometric.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_hypergeometric.py)
asks for a double-precision routine `hyp2f1(a, b, c, z)` — the Gauss
hypergeometric function, real parameters, over a wide declared range. Three
properties are what a leaderboard would otherwise have supplied.

**The problem is hard, and not on this repository's say-so.** The standard
survey — Pearson, Olver & Porter, *Numerical methods for the computation of the
confluent and Gauss hypergeometric functions*, Numerical Algorithms 74:821–866
(2017) — exists because no single method covers the parameter space: the Taylor
series diverges outside the unit disc and cancels well inside it, every
transformation has a bad region of its own, and the recurrences are unstable in
one direction.

**The baseline is the state of the practice, not a strawman.**
`scipy.special.hyp2f1` — Cephes underneath, in production for decades, the
function a working scientist already calls. On the declared distribution it
**loses more than six digits on roughly a third of points**, and some of those
land at zero correct digits. That is measured in
`tests/test_era_hyp2f1.py::test_the_baseline_is_not_a_strawman_and_not_perfect_either`,
which also fails if SciPy ever gets good enough that the numbers here need
restating.

**The reference cannot be argued with.** Every value comes from mpmath, which
shares no code with SciPy and none with any candidate, computed at **30 and at
60 decimal digits**, and kept only where the two agree to 25. Nothing in the
sandbox can reach it — the file a candidate's shard is built from carries four
parameters per point and no values. And the whole set is committed, so the
claim is checkable rather than reported:

```bash
python -m tools.gen_hyp2f1_stress --check   # redraws and demands the file back
```

Two tests hold that down: one re-derives every stored value from mpmath at 60
digits, the other reruns the generator and requires the committed file **byte
for byte**, which is what rules out points having been chosen after seeing how
an implementation did on them. The distribution — `a, b, c ~ U(-30, 30)`,
`z ~ U(-40, 0.999)` — was fixed before anything was measured and is recorded in
the data file beside the values.

### The one constraint that keeps the comparison honest

`mpmath`, `decimal` and `fractions` are **off this task's import allowlist**.
The deliverable is a float64 routine, comparable with SciPy's; a candidate that
reimplemented arbitrary-precision arithmetic would be answering a different
question and would be scored against a reference produced the same way it was.
`tests/test_era_hyp2f1.py` asserts each of those imports is refused by the gate.

Everything else in `scipy.special` — including `hyp2f1` itself — is allowed, on
purpose. Using the baseline where it is reliable and something better where it
is not *is* the expert answer here; the search's job is to find where the line
falls and what to do on the far side of it, from `(a, b, c, z)` alone, with no
sight of the answer.

## The fourth task — LLM-SRBench, and a benchmark this repository did not build

The three tasks above are scored against a Kaggle split, against arithmetic, and
against an arbitrary-precision reference. Two of the three run on suites
constructed *here*, which is stated wherever they are reported and is the
standing objection to both: the people who built the task also set the bar.

[`examples/era/era_llm_srbench.py`](https://github.com/Birfy/agentdescent/blob/main/examples/era/era_llm_srbench.py)
answers that objection directly. It runs
**[LLM-SRBench](https://arxiv.org/abs/2504.10415)** (ICML 2025 Oral) — a
published benchmark for scientific equation discovery, built by other people,
with its own metrics and its own leaderboard of LLM-based methods. Nothing about
the problems, the splits, the tolerance or the metric definitions is this
repository's choice.

| | |
|---|---|
| Problems | 240 in five subsets: `lsr_transform` (111 Feynman equations rearranged into unfamiliar forms) and `lsr_synth` (chemistry 36, biology 24, physics 44, materials 25) — the paper's abstract says **239**, see below |
| Samples | LSR-Synth: 4 000 train / 500 in-domain test / 500 **out-of-distribution** test. LSR-Transform: 80 000 / 20 000, no OOD split |
| Metrics | The paper's own: `NMSE = Σ(ŷ−y)² / Σ(y−ȳ)²`, and `Acc_τ = 1(max_i \|(ŷ_i−y_i)/y_i\| ≤ τ)` with τ = 0.1 |
| Baseline node | Sequentially thresholded least squares over a fixed nonlinear library — SINDy's fitting step (Brunton et al., 2016) without its domain-chosen library |

The **240 is the released data's count, and the paper says 239.** The gap is one
physics problem: the paper's text reads "111 problems in the first category
(LSR-Transform), and 128 problems in the second category (LSR-Synth), spanning
… chemistry (36), biology (24), physics (43), and material science (25)", while
the benchmark's own gated HuggingFace dataset card lists `lsr_synth_phys_osc` at
44 — and the ungated mirror agrees with the card. This port follows the data,
because the data is what gets scored: dropping a problem to make the abstract's
arithmetic work would be reporting a benchmark nobody published.
`test_the_released_data_holds_240_problems_where_the_paper_says_239` pins it so
it stays a recorded fact rather than something quietly "fixed" later.

### What the candidate is asked for

```python
def discover(x, y, spec):   # x: (n, d) float64, y: (n,) float64
    ...                     # -> a closed-form equation, as a string
```

`spec` carries the column names, the output name, the benchmark's own
one-paragraph description of the science, the per-problem time budget, and
`spec["evaluate"]` — the grader's parser, so a method can score the forms it is
proposing with the same code that will score its answer.

The **answer is a string that is parsed, never executed**. The grammar admits
numeric constants, the problem's variables, `pi`/`e`, `+ - * / **`, and a fixed
list of elementary functions; it refuses comparisons, indexing, attribute access
and every name outside that list. Two things follow, and both are load-bearing:

* **It stays equation discovery.** A gradient-boosted regressor or a
  nearest-neighbour table would win on in-domain test points and has discovered
  nothing. Neither can be written down in this grammar. Nor can a piecewise
  answer with enough branches, which is why `where` and the comparison operators
  are absent rather than merely undocumented.
* **The held-out samples stay out of reach.** The candidate is handed the
  training arrays and nothing else; the test and OOD arrays are opened *after*
  `discover` returns, and only ever by an interpreter walking an AST that has
  already been validated. There is no moment at which candidate code and
  held-out data are both live.

### Why the score means something

**The benchmark is somebody else's, and so is the yardstick.** `Acc_0.1` and
NMSE are the paper's definitions, transcribed from §3 and checked in
`tests/test_era_srbench.py`. The paper reports both per domain for LLM-SR, LaSR,
SGA and direct prompting across three model backbones, so a number from here has
somewhere to sit — with the caveat in the next paragraph attached to it.

**The protocol is ERA's, not the benchmark's.** LLM-SRBench evaluates searchers
that see *one problem at a time*, with the data in the model's context and a
per-problem sample budget. Here the model never sees a data point: it writes one
program, and that program is run sandboxed against every problem in the set.
Same benchmark, same splits, same metrics, **different experiment** — which is
recorded in every result file under `comparability`, in the module docstring, and
here. The two are not interchangeable, and a table that put them in the same
column without the note would be misleading.

**The data is checked, not trusted.** The benchmark's own HuggingFace release
(`nnheui/llm-srbench`) is gated and returns 401 without a token, so this task
reads an ungated re-upload (`pkuHaowei/llm-srbench`), pinned to a revision.
`tests/test_era_srbench.py::test_the_published_equations_reproduce_the_published_samples`
re-evaluates every ground-truth expression that parses against the samples
shipped beside it and requires NMSE < 1e-6 — which is what ties the mirror to the
published benchmark. Two subsets' ground-truth *strings* are damaged in that copy
(36 chemistry expressions carry a mangled parameter, `0.189…_z`; 44 physics
expressions are templates whose `F0`/`beta`/`omega0` have no values), so they are
excluded from that check and the test asserts they are still broken — a mirror
that quietly fixed them should make the note stale, not silently pass. Scoring
never touches those strings: it is numeric throughout.

**The tree ranks on `min(12, -log10(NMSE))`, averaged over the problem set.**
`Acc_0.1` is an indicator per problem — flat almost everywhere, so a search
cannot descend it — and raw NMSE spans twelve orders of magnitude, so a mean of
it is whichever problem failed worst. The cap is the precision of the published
samples themselves, which are stored as float32: a ground-truth expression
re-evaluated on them lands at NMSE ≈ 1e-13. Both benchmark metrics are reported
beside it, for in-domain and OOD alike.

### Deviations this task adds

1. **A non-finite prediction fails the problem here; upstream drops the point.**
   `bench/pipelines.py` computes NMSE over `~isnan(y_pred)`, so an equation with
   a pole inside the test range is scored on the points either side of it. This
   port counts a non-finite prediction as a failed problem, because a search told
   otherwise learns to place poles. Both numbers are reported —
   `nmse` under this rule, `nmse_upstream` under the paper's — so a result here
   can still be laid beside a result there.
2. **The candidate is handed an evaluator.** `spec["evaluate"]` exists because
   the AST gate refuses `eval`, and a symbolic-regression method that cannot
   evaluate its own candidate forms would have to re-implement the grammar and
   hope its copy matched. A near-miss there shows up as a good method scoring
   zero, which is a defect in the harness rather than a fact about the method.
3. **The per-problem budget is enforced with `SIGALRM` in the runner**, not
   trusted to the candidate. Without it one method that fails to return on one
   problem costs every remaining problem in the shard its score.
4. **The runner imports numpy**, unlike its two siblings, which are standard
   library only: the candidate is handed float64 arrays and its answer is scored
   against more of them.
5. **Shards are dealt per domain**, not by shuffling the pooled list: every
   shard holds each of the four domains to within one problem. A draw that
   handed one shard no physics at all is otherwise perfectly ordinary at these
   sizes, and a node's score would then depend on which shards the verifier drew.
6. `score = mean digits` with no sign flip — FUTS maximises — and the engine's
   `[0, 1]` reward is `mean_digits / 12`, exactly order-preserving with what the
   tree ranks on.

## Measured results — Playground S3E1

### The method

| Setting | Value |
|---|---|
| Model | `glm-5.2`, Anthropic-shaped API |
| Sampling | temperature 0.7, **thinking disabled**, `--max-tokens 16000` |
| Mode | `async_evolve(n_workers=3, async_ratio=1)`, `--staleness full` |
| Budget | 6 expansions, hard-capped; `--max-seconds 1800` |
| Search | `c_puct = 1.0`, `--candidate-timeout 60` (upstream's `Sandbox(timeout_seconds=60)`) |
| Data | upstream's full 80% split — 29,709 training rows |
| Scoring shards | 8 of the 20% tail (4 rollout, 4 held-out gate), 619 rows each |
| Independent test | 4 further shards, 2,476 rows, never scored during the search |
| Isolation | Seatbelt (`sandbox-exec`), macOS |
| Replay | none; a single live engine run |

The recorded output is
[`bench/results/era-quality-run.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-quality-run.json).

### The result

6 expansions, 6 mutation calls, 0 failures, 8,962 tokens, 99.0 s of model time,
427.3 s of wall clock. Every figure below is scored on the **test** shards,
which the search never saw:

| | baseline | best found | |
|---|---:|---:|---:|
| test RMSE | 0.72968 | **0.59133** | −19.0% |
| held-out gate RMSE | 0.73915 | **0.58245** | the split the tree ranked on |
| framework reward `1/(1+RMSE)` | 0.5750 | **0.6320** | |

The baseline is upstream's own `LinearRegression` seed. The winner is a
`GradientBoostingRegressor` with early stopping over ten engineered features —
income × age / rooms / occupancy interactions, a longitude × latitude location
term, a distance-to-origin term, a high-income flag — and predictions clipped to
the target's known `[0, 5]` bounds. It is written to
`era-agentdescent-result-best.py` beside the JSON.

The tree machinery is live: 7 nodes, all 7 valid, root visited 6 times, max
depth 2, and `--staleness full` considered 6 cards and discarded none.

!!! note "Two things this run measured that the score does not show"
    **The gate is 95% of the wall clock, and workers are what starve.**
    `merge_gate_seconds` was 406.1 s of a 421.9 s run, and
    `worker_starved_seconds` 128.8 s. Every surviving card is re-executed across
    four held-out shards, and each of those is a full training run on 29,709
    rows inside the sandbox — on the merger thread. The parallelisable part is
    the model call (99 s across all six), so on this port **more `--workers` buys
    almost nothing**; `--eval-concurrency` and `--pipelined-gate` are the levers,
    and a speedup row here would be measuring the sandbox, not the scheduler.

    **Asynchrony makes the tree root-heavy.** Five of six expansions attached to
    the root, because sweep 1 dispatched four proposals before any sibling had
    been inserted — with one node in the tree, `argmax(puct)` can only return the
    root. The offline canned-model run showed the same effect more sharply
    (depth 1 async against depth 2 sync at the same budget). It cost nothing
    here, since the best node happened to be a root child, but it is a real
    semantic effect of the barrier-free schedule, and it is why tree depth
    belongs beside the score rather than in a footnote.

!!! warning "One run, one seed, and no serial control"
    This is a single live run. No `--serial` arm was measured, so **no speedup or
    parallel-efficiency claim is made here** and the ERA row in
    [port-fidelity.md](port-fidelity.md#the-parallelisation-matrix) is empty
    rather than filled from one arm.

## Measured results — hard integrals

### The method

| Setting | Value |
|---|---|
| Model | `glm-5.2`, Anthropic-shaped API |
| Sampling | temperature 0.7, **thinking disabled**, `--max-tokens 16000` |
| Mode | `async_evolve(n_workers=3, async_ratio=1)`, `--staleness full` |
| Budget | 12 expansions, hard-capped; `--max-seconds 3600` |
| Search | `c_puct = 1.0`, `--candidate-timeout 60` (upstream's `Sandbox(timeout_seconds=60)`) |
| Problems | 9 families × 12 problem sets, `--seed 0` |
| Scoring sets | 8 (4 rollout, 4 held-out gate), 9 integrals each |
| Independent test | 4 further sets, 36 integrals, never scored during the search |
| Per problem | ≤ 200,000 integrand calls, ≤ 5 s |
| Isolation | Bubblewrap, Linux |
| Replay | none; a single live engine run |

The recorded output is
[`bench/results/era-integrals-run.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-integrals-run.json),
and the winning program
[`bench/results/era-integrals-run-best.py`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-integrals-run-best.py).

### The result

12 expansions, 13 mutation calls (one stalled on the endpoint and was retried by
`with_retries`), 23,877 tokens, 615.9 s of model time, 391.0 s of wall clock.
Every figure below is on the **test** problem sets, which the search never saw:

| | baseline | best found | |
|---|---:|---:|---:|
| test mean correct digits | 8.862 | **10.205** | +1.34 |
| test problems at 10+ digits | 20 / 36 | **28 / 36** | |
| held-out gate mean digits | 8.681 | **10.139** | the split the tree ranked on |
| framework reward `digits / 12` | 0.7234 | **0.8449** | |
| integrand calls over the test set | 33,828 | 221,277 | of 7.2 M allowed |

The baseline is `scipy.integrate.quad` on defaults. The winner keeps `quad` as
its kernel and supplies what `quad` cannot infer: an explicit `t/(1 - t²)` map
for `(-inf, inf)` and `t/(1 - t)` for a half-line, a singular point declared at
the join, a ladder of `(limit, epsabs, epsrel)` settings that stops as soon as
the returned error estimate is below `1e-12`, and a wrapper that turns a raised
or non-finite integrand value into a zero rather than a dead program. That is
the shape of the task: not a better formula, but the transformation and the
error control an expert would add around a library routine.

The tree machinery is live: 13 nodes, 12 valid, **depth 3**, root visited 12
times, and `--staleness full` considered 12 cards and discarded none.

!!! note "Two things this run showed that the score does not"
    **The class that survived is the one that needs real analysis.** All four
    of the winner's zero-digit failures are the same family — an oscillation on
    `[0, inf)` whose amplitude never decays. Truncating that tail is wrong at
    any truncation point, and no tolerance setting rescues it; it wants
    half-period splitting plus a convergence acceleration, which is a different
    program rather than a better-tuned one. The remaining headroom is therefore
    concentrated and identifiable, which is what a benchmark is for.

    **A node died on an import the gate had allowed.** One expansion wrote
    `from scipy.interpolate import pade`, which the AST gate admits — `scipy` is
    on the allowlist — and which does not exist at that path in the installed
    version. It failed inside the sandbox, scored `-inf`, and was appended to
    the tree as upstream requires. That is the gate and the sandbox doing
    exactly what this port says they do: the gate is not the boundary, and a
    program that cannot run is a node rather than a crash.

!!! warning "The first run of this task found a hole in the task"
    The `(-inf, inf)` family was `exp(-x²)·cos(bx)`, which is **even**. The
    first live run's winner mapped both halves of the line onto `[0, inf)` and
    doubled — a plain bug — and scored 12 digits on it. The family now carries
    a nonzero offset, `exp(-(x - m)²)·cos(bx)`, with
    `sqrt(pi)·e^(-b²/4)·cos(bm)` as its closed form, and
    `test_the_whole_line_family_is_not_symmetric_about_the_origin` keeps it
    that way. The numbers above are from a rerun on the corrected suite; the
    first run's are not reported, because they were measured against a suite
    that could be passed without solving it.

!!! warning "One run, one seed, and no serial control"
    As with the S3E1 numbers above: a single live run, so **no speedup or
    parallel-efficiency claim is made from it**. The gate is a much smaller
    share of the wall clock here than on S3E1 (31.3 s of 391 s, against 406 s of
    422 s) because scoring nine integrals is milliseconds where training a
    regressor on 29,709 rows is seconds — so on this task the model call, not
    the sandbox, is what parallelism has to hide.

## Measured results — 2F1

### The method

| Setting | Value |
|---|---|
| Model | `glm-5.2`, Anthropic-shaped API |
| Sampling | temperature 0.7, **thinking disabled**, `--max-tokens 16000` |
| Mode | `async_evolve(n_workers=3, async_ratio=1)`, `--staleness full` |
| Budget | 18 expansions, hard-capped; `--max-seconds 5400` |
| Reply guard | `--reply-attempts 4` (see below — it was needed) |
| Points | 20 per set; 8 sets scored (4 rollout, 4 gate) — **the old suite**, see below |
| Independent test | 4 further sets, **80 points**, never scored during the search |
| Reference | mpmath 1.4.1 at 30 and 60 dps, kept where they agree to 25 |
| Isolation | Bubblewrap, Linux |
| Replay | none; a single live engine run |

Recorded in
[`bench/results/era-hyp2f1-run.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-hyp2f1-run.json).

### The result, which is mostly a negative one

18 expansions, 20 mutation calls (**2 arrived damaged and were redrawn**),
79,378 tokens, 1,148 s of model time, 404 s of wall clock.

| | baseline | best found | |
|---|---:|---:|---:|
| held-back mean correct digits | 9.692 | **9.826** | +0.13 |
| held-back points at 10+ digits | 51 / 80 | **52 / 80** | +1 point |
| gate mean digits | 10.185 | 10.202 | the split the tree ranked on |

That is a real improvement on points the search never saw, and it is **small**.
The comparison that says how small: a **five-line decidable rule** — apply Pfaff
when `z < -1`, pick the branch by which one has the smaller parameters — scores
10.148 against the baseline's 9.732 over all 240 points, about **+0.42**. A
numerical analyst writes that in five minutes. Eighteen expansions of `glm-5.2`
found roughly a third of it.

The tree says where the budget went: 19 nodes, 18 valid, depth 3. **Seven scored
exactly the baseline to six decimals** — programs that build a transformation and
then wrap it in a fallback whose condition never fires, so every point takes the
`scipy` branch. Five scored far *worse* (0.19, 0.88, 0.92, 4.20, 6.52): genuine
attempts at the connection formulas that got a sign or a branch wrong. Only one
node beat the root at all.

The winner is not a trivial program — it tracks the sign of `loggamma` through
negative arguments, sums with log-sum-exp to control cancellation, and keeps a
series with a convergence test. It is simply not yet better than Cephes at what
Cephes is good at.

!!! note "The gate refused a candidate that reached for arbitrary precision"
    One node died with `gate: import 'mpmath' is not allowed`. That is the
    constraint working exactly as designed: the model correctly identified that
    arbitrary precision would solve the problem, and the allowlist refused,
    because a routine that reimplements mpmath is not comparable with SciPy and
    would be scored against a reference produced the same way it was. The
    refusal is worth more than the node would have been.

!!! warning "Roughly one reply in five arrived damaged, and it was not the model"
    Measured on this endpoint: replies of a few thousand characters came back
    with bytes spliced into the middle of tokens — `return val9.3192`,
    `c_orig,0$ zG$C$F1_orig` — at about **19%**, pooling every certain case (a
    reply that does not parse) over 58 sampled replies. The rate is the same
    through the Anthropic SDK, through its streaming API, and through a
    hand-rolled `urllib` request, while 25 fetches of a similarly-sized file
    over the same proxy hashed identically. So it is the endpoint.

    The **first** 2F1 run was made without the guard, and 3 of its 15 expansions
    died on a `SyntaxError` the model did not write; it finished at 9.692 →
    9.692, no improvement at all. That run is not reported as a result, because
    it measured a channel. `--reply-attempts` redraws a reply that is not Python
    at all and never redraws a program that merely fails — the latter is still a
    node scoring `-inf`, as upstream requires — and every run now records
    `reply_damage` beside its numbers.

    A splice that lands inside a numeric literal still parses, and nothing here
    can catch it. Results measured through this endpoint carry that caveat.

!!! warning "One run, one seed, one sampling mode"
    No `--serial` arm, so no speedup claim. And `--thinking disabled` is a
    choice that changes the model's output, not only its latency: on a task
    whose difficulty is a chain of transformation identities, it may well be the
    binding constraint. This row is `thinking=disabled` and says so.

### The 80-point split had no resolution, and both rows above are inside its noise

The two rows above were measured on a suite of 240 points: 80 for the gate the
tree ranks on, 80 held back. That was not enough, and the arithmetic is not
close:

```
per-point correct digits, standard deviation  : 3.20
  standard error of an   80-point mean        : 0.358 digits
  standard error of a  1000-point mean        : 0.101 digits
```

The outcome per point is close to **bimodal** — a program either handles a
region and scores near the 12-digit cap, or misses it and scores near zero — so
the spread is enormous and averaging 80 of them settles very little. The
smallest gain an 80-point gate can separate from noise at two standard errors is
**0.72 digits**. Every number in the two rows above is smaller than that.

Pairing does not rescue it: the paired difference between two programs on the
same points has an SD of 3.05 against the unpaired 3.20, because when a program
changes a point it changes it by ten digits rather than by a tenth.

**So the suite was regenerated at 250 points a shard — 3000 points, a 1000-point
gate and 1000 held back** (`tools/gen_hyp2f1_stress.py`, twelve minutes of
arbitrary-precision arithmetic, committed). Evaluation cost is not the reason it
was small: a 250-point shard takes the baseline 0.30 s and the best evolved
program 0.36 s, and scoring a node across the whole 1000-point gate takes 1.7 s.

### What the resolving gate says about the run that was already made

Re-scored on **1000 fresh points** it had never seen:

| | mean digits | 10+ digits | vs baseline |
|---|---:|---:|---|
| baseline `scipy.special.hyp2f1`, 1000-point gate | 9.801 | 642 / 1000 | — |
| baseline, 1000 held back | 9.836 | 659 / 1000 | — |
| **the 48-expansion winner, on 1000 fresh points** | **10.148** | **737 / 1000** | **+0.347 ± 0.10, 3.2 SE** |

**That reverses the reading recorded here earlier.** The 48-expansion search did
produce a genuinely better program — about a third of a digit, and 95 more
points solved out of 1000 — and the old split could not see it. What the old
numbers showed (gate +0.56, held-back −0.15) was two draws from a distribution
with a 0.36-digit standard error. The earlier text called that a winner's curse;
it was an unresolved measurement, and the diagnosis was wrong.

The corroborating detail: on the new suite the gate half and the held-back half
score the baseline at **9.801 and 9.836**, a gap of 0.035. On the old suite the
same two halves differed by **0.49** — the "gate is easier than the test set"
effect visible in the earlier rows was itself noise.

### And what it does for a run made under it

48 expansions again, same model and sampling, `--workers 6` this time, scored
against the 1000-point gate
([`era-hyp2f1-run48-gate1000.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-hyp2f1-run48-gate1000.json)):

| | baseline | best found | |
|---|---:|---:|---:|
| gate mean digits (1000 points) | 9.801 | **11.737** | **+1.936** |
| **held-back mean digits (1000 points)** | 9.836 | **11.771** | **+1.935** |
| held-back points at 10+ digits | 659 / 1000 | **965 / 1000** | |
| over all 3000 points | 9.849 | **11.738** | +1.889, 1970 → 2880 solved |

**The gate and the held-back set now agree to 0.001 digits** (+1.9359 against
+1.9346). That is the whole point of the resize: the two halves of the same
distribution finally say the same thing, so a gate improvement is evidence about
the program rather than about which 80 points were drawn.

**What the program actually found — corrected after review.** An earlier
version of this section claimed the program was "past the transformation-picking
ceiling" because an oracle over four textbook transformations reaches 10.98
while the program reaches 11.74. That comparison was invalid: the oracle basis
excluded the identity the program itself uses. Measured over a basis that
includes it:

| on all 3000 points | mean digits | 10+ digits |
|---|---:|---:|
| `scipy.special.hyp2f1` | 9.849 | 1970 |
| **the z→1/z connection formula, applied blindly, no selection at all** | **11.253** | 2772 |
| the evolved program | 11.738 | 2880 |
| oracle over 4 textbook transformations *(the old, invalid basis)* | 10.981 | 2485 |
| oracle over those 4 **plus z→1/z** — still unreachable, it needs the answer | **11.919** | 2961 |

So the evolved program sits **below** the transformation-picking ceiling, not
above it, and **74% of its gain (+1.40 of +1.89) comes from one identity applied
without any selection at all**. Instrumenting the winner confirms the mechanism:
the z→1/z branch answers 2775 of 3000 points, SciPy 101, the direct Taylor 106,
the Pfaff branch 18.

The honest result is still a real one, and it is this: **the search rediscovered
the z→1/z connection formula and a `z < −1` switching rule**, which is worth
+1.9 digits on this distribution. It is not evidence that the program carries
machinery beyond the identities — a review found that `_safe_gammaln` is defined
and never called, `numpy` and `poch` are imported and never used, the
"complex" Taylor summation has an imaginary part of exactly zero at all 3000
points (because `cmath.log(-z)` is real for `z < 0`), and `_analytic_cont` is
invoked **zero** times on the suite.

Cost: 598 s wall, 52 calls, 210 k tokens, 4 replies damaged. The same 48
expansions at `--workers 3` took 1,280 s, so raising concurrency to the
endpoint's measured knee bought **2.1×**.

!!! warning "Better on 1284 points, worse on 111, and 13 of those catastrophically"
    Across all 3000 points the winner gains 6,142 digits and loses 475. The
    losses include **13 points where the baseline had 10+ digits and the winner
    has under 1**. A mean is the right headline and the wrong acceptance rule: a
    numerical library does not ship a change that breaks thirteen inputs it used
    to get right, however good the average. The lever for that is an acceptance
    condition of "mean improves **and** no new catastrophic regression", which
    is a counting statistic and far less noisy than the mean it guards.

!!! danger "Two wrong identities in the winning program, in branches the suite never reaches"
    A specialist review of the artifact found two mathematical errors, both
    confirmed here independently.

    **The z→1−z connection formula has both Γ coefficients wrong**
    (`_analytic_cont`). DLMF 15.8.4 requires
    `C₁ = Γ(c)Γ(c−a−b)/(Γ(c−a)Γ(c−b))` and `C₂ = Γ(c)Γ(a+b−c)/(Γ(a)Γ(b))`; the
    program divides the first by `Γ(a)Γ(b)` and writes the second as
    `Γ(c)Γ(1−c)/(Γ(c−a−b)Γ(1+a+b−c))`, which is neither. At
    `a=0.3, b=0.7, c=1.9, z=0.6` it returns −0.0965 where the value is 1.0895 —
    **zero correct digits**. It computes `Γ(c−a)` and `Γ(c−b)` and then never
    uses them.

    **The `a ≈ b` guard applies the wrong Pfaff transformation.** Pfaff needs
    `₂F₁(a, c−b; c; z/(z−1))`; the branch sums `₂F₁(a, a; c; z/(z−1))`, which
    agrees only on the line `c = 2a`. At `a = b = −2.3, c = 4.1, z = −6` it
    returns 198.85 where the value is 0.7448 — 267× too large, and SciPy gets
    that point right.

    **Neither is visible to the benchmark.** `_analytic_cont` is never called on
    the 3000 points, and the probability of `|b−a| < 10⁻⁵` under `U(−30,30)²` is
    about 3×10⁻⁷, so the suite contains no point that would exercise either. A
    benchmark that cannot reach a branch cannot penalise it — which is an
    argument about this suite's coverage (see below), not a defence of the code.

!!! warning "What the suite does not sample at all"
    Measured on the committed file: **0 points** with `z ≥ 1` (excluded by
    construction, `Z_HIGH = 0.999`), **1 point** above `z = 0.99`, **0 points**
    with `a` or `b` a non-positive integer (the terminating/polynomial case —
    the most common practical use of ₂F₁, and measure zero under a uniform
    draw), **0 points** with `b−a` or `c−a−b` within 10⁻⁶ of an integer (the
    logarithmic cases where both connection formulas degenerate), and **0.5%**
    with all of `|a|, |b|, |c| < 5`, which is where most real calls live.
    92.5% of the suite sits in `z < −2.2`, the single region the winner's main
    branch covers. The +1.9-digit headline is a true statement about this
    distribution and a weak proxy for "beats SciPy at computing ₂F₁".

!!! note "A silent splice, in the winning program"
    Line 65 of the winner reads `w = z / (z - 1/ 1.0)`. Nobody writes `1/ 1.0`;
    it has the shape of the transport damage documented below, and it parsed, so
    the guard could not see it. It is harmless here — `1/1.0` is `1.0`, so the
    expression is the Pfaff argument it was meant to be — but it is a concrete
    instance of the limitation stated there: a splice inside a numeric literal
    survives every check this port has.

!!! note "What survives from the earlier reading"
    Two observations do not depend on the resolution and still stand. The tree
    spent its budget badly: the best score appeared at **expansion 8** and the
    remaining 40 expansions produced exact copies of it, chains 14 deep in which
    every node scores identically because each rewrite preserved the parent's
    behaviour. And the reply channel damaged **12 of 60** replies, 20%, matching
    the ~19% estimated over 58 earlier samples.

!!! tip "The two levers still worth pulling"
    A **behavioural signal in the prompt** — "your last program changed nothing
    on 19 of the 22 points its parent failed" is computable host-side without
    revealing an answer, and aims straight at the plateau. And **storing the
    top-K node programs**, so gate-versus-test correlation can be measured
    directly rather than inferred from whichever program happened to win.

## Measured results — LLM-SRBench

### The method

| Setting | Value |
|---|---|
| Model | `glm-5.2`, Anthropic-shaped API (`--provider claude`), `--thinking disabled` |
| Search | sync, 12 expansions, 3 workers, `c_puct=1.0`, `--staleness guarded`, `--seed 0` |
| Data | every problem in the category — 129 (LSR-Synth) and 111 (LSR-Transform) |
| Shards | 6 scored + 2 held back, dealt per domain; `held_out_frac=0.5`, so a node is gated on 3 shards |
| Budget | 4 s per problem, 120 s per problem set, `--max-tokens 16000` |
| LSR-Transform only | `--train-points 4000`, matching LSR-Synth's training size — it ships 80 000 |

Two runs, one per category, so between them they cover **all 240 problems** —
129 and 111, no cap, no overlap. Note what that does and does not mean: every
problem was in a run, but each run scores its search against 6 shards and
holds 2 back, so the before/after numbers below rest on the **59 held-back
problems** (32 + 27) and the other 181 are what the search was gated on.
Everything below is the **held-back** shards: problems no expansion was ever
scored against. Files:
[`bench/results/era-srbench-synth.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-srbench-synth.json)
and
[`bench/results/era-srbench-transform.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-srbench-transform.json),
with the winning programs beside them.

### The result

| | LSR-Synth, 32 held-back problems | LSR-Transform, 27 held-back problems |
|---|---|---|
| mean `min(12, -log10 NMSE)` | **7.225 → 8.439** | **1.058 → 2.453** |
| median NMSE | 2.85e-07 → **4.83e-11** | 0.1101 → 0.0974 |
| Acc(0.1) | 50.0% → **59.4%** | 0.0% → **14.8%** |
| OOD mean digits | 4.718 → **5.463** | no OOD split in this category |
| OOD Acc(0.1) | 53.1% → **68.8%** | — |
| problems at 6+ digits | 20/32 → 22/32 | 0/27 → 3/27 |
| paired per problem | 18 better, 5 worse, 9 tied — sign test **p = 0.011** | 15 better, 9 worse, 3 tied — sign test p = 0.31 |
| wall / model calls / tokens | 584 s / 14 / 76 230 | 570 s / 14 / 83 092 |
| damaged replies redrawn | 2 of 14 | 2 of 14 |
| expansions that scored zero | 2 of 12 | 4 of 12 |

**The pooled means are the weaker half of that table.** Thirty-two problems put
one problem at 3.1 points of Acc(0.1), so a 9-point move is three problems and
proves very little on its own. The paired column is what carries the LSR-Synth
result: the same 32 problems, the same splits, 18 improved against 5 that
regressed, which a two-sided sign test puts at p = 0.011. On LSR-Transform the
same test says **p = 0.31** — the mean nearly doubled and the median NMSE barely
moved, which is the signature of a method that solved three more problems
outright while losing ground on others. Reported as such rather than as a win.

Per domain on LSR-Synth, with the caveat that a domain here is 6 to 12 problems
and each one is worth 8 to 17 points of Acc:

| Domain | Held-back | mean digits | Acc(0.1) |
|---|---|---|---|
| chemistry (reaction kinetics) | 8 | 8.97 → 11.03 | 87.5% → 87.5% |
| biology (population growth) | 6 | 7.61 → 9.42 | 66.7% → 83.3% |
| material science | 6 | 6.39 → 7.92 | 33.3% → 50.0% |
| physics (nonlinear oscillators) | 12 | 6.29 → 6.48 | 25.0% → 33.3% |

The gate agreed with the held-back set on direction and overstated the size:
the winning node scored 8.726 on the three shards it was gated against versus
8.439 on the two it never saw, against a root at 7.364 / 7.225. Four rounds,
committing at rounds 1 and 3 (0.6137 → 0.6938 → 0.7272 held-out reward).

### What the search actually found

Both winners are recognisably the seed program with two things done to it, and
neither is a new idea about symbolic regression:

* **A wider library.** The seed offers `1, v, v², v³, sin, cos, log, √, 1/v,
  e^-v` per variable plus pairwise products. The LSR-Synth winner adds `v⁴`,
  `tanh`, `e^+v`, `sin(v_i)·v_j`, `cos(v_i)·v_j`, `v_i²v_j`, `v_iv_j²` and
  triple products — which is what the four synthetic domains want, since their
  ground truths are ODE right-hand sides that are *additive* in terms like
  these.
* **A validation split and an escalation schedule.** Both winners hold back a
  quarter of the training samples, select on that rather than on the fit, and
  try term sets in increasing size under a wall-clock check against
  `spec["seconds"]`, keeping the best answer found so far. The LSR-Transform
  winner is almost entirely this: constant, then linear, then a growing set with
  `1/v`, `1/v²`, `√|v|`.

Both call `spec["evaluate"]` on the assembled expression before returning it, so
the answer is checked by the grader's own parser first. That idiom is in the
seed program, which is where they got it.

### Neither winner transfers to the other category

The prompt asks for "one method that works across every domain above, not a
dispatch on the problem id", and within a category that is what comes back. The
categories are another matter. Each winner, run on the *other* run's held-back
shards — problems it never saw, in a category it was never scored on:

| Program | LSR-Synth (32 held back) | LSR-Transform (27 held back) |
|---|---|---|
| seed (STLSQ over a fixed library) | 7.225 digits, 50.0% Acc | 1.058 digits, 0.0% Acc |
| LSR-Synth winner | **8.439**, 59.4% | 0.788, 0.0% |
| LSR-Transform winner | 5.238, 37.5% | **2.453**, 14.8% |

Each is best on its own category and **worse than the seed on the other**. Twelve
expansions of program search against one category produce a method tuned to it:
the LSR-Synth winner's wide additive library is dead weight on equations that are
products and quotients, and the LSR-Transform winner's escalation schedule never
reaches the terms the synthetic right-hand sides need inside four seconds. This
is the honest cost of the protocol — one program for a whole distribution — and
it is measurable here only because the benchmark ships two categories that
differ in kind. `--dataset all` runs the search against both at once; that run
has not been made.

### One search for the whole benchmark, not one per problem

The single most important number for reading the next table is the **LLM budget**,
because the two protocols do not sit on the same axis at all.

The paper's methods evolve **per problem**: each of the 240 problems gets its own
search. From the benchmark's own configs (`configs/llmsr_gpt4omini.yaml` and
friends) —

| Method | LLM budget, per problem | For a 129-problem category |
|---|---|---|
| LLM-SR | 1 000 samples (`global_max_sample_num`), 4 per prompt → 250 prompts | ~32 000 prompts |
| LaSR | 2 000 samples (`max_num_samples`), 25 iterations × 10 populations | ~258 000 samples |
| Direct prompting | 5 samples in 1 prompt | 129 prompts |
| **this port** | — | **12 expansions = 14 prompts, for the whole category** |

So the run measured above spends **14 model calls to cover 129 problems** —
about **0.11 calls per problem**, roughly 2 300× less LLM per problem than
LLM-SR, and an order of magnitude less than *direct prompting does for a single
problem*. There is one program, written without the model ever seeing a data
point, and it is run over every problem in the benchmark. Each problem then gets
four seconds of CPU and no network.

That is not a better way to run LLM-SRBench. It is a different question — "can a
model write one method that solves a whole scientific distribution?" rather than
"can a model find this equation?" — and the numbers below are only interesting
with the budget column attached.

### The same benchmark under its own protocol (`--per-problem`)

The section above says the LSR-Transform deficit is the protocol's rather than
the search's. That is a testable claim, so it was tested: the same tree search,
the same seed program, the same grammar and the same sandbox, run **once per
problem** over all 111 LSR-Transform problems.

| Setting | Value |
|---|---|
| Search | 6 expansions per problem, 3 workers, 6 validation shards from 25% of train |
| Budget | 8 s per problem per expansion, `--train-points 4000`, `--problem-concurrency 2` |
| Cost | 584 model calls for 111 problems — **5.26 per problem**, 1.62M tokens, 3 810 s wall |
| Reported on | each problem's own benchmark test split, which no expansion is scored against |

| LSR-Transform, all 111 problems | seed program | after 6 expansions |
|---|---|---|
| Acc(0.1) | 0.9% | **43.2%** |
| median NMSE | 0.0941 | **0.00568** |
| mean `min(12, -log10 NMSE)` | 1.074 | **5.391** |
| problems at 6+ digits | 0 | **44** |
| problems at the 12-digit cap | 0 | **41** |
| paired per problem | — | 59 better, 1 worse, 51 tied, sign test **p = 1e-16** |

Against the same category under the whole-category protocol — 8.1% Acc(0.1),
0.0753 median NMSE — that is a **five-fold** move, and it is the protocol that
moved rather than the search: identical tree, identical seed, identical grammar.

**And it recovers equations, not just fits.** The answers are median 3 terms and
126 characters against a ground truth of 1 term and 33; under the whole-category
protocol they were 9 terms and 240+. Forty-one problems land on the 12-digit cap,
and reading them shows why:

| ground truth | what the search returned |
|---|---|
| `-sqrt(q1*q2/(F*epsilon))/(2*sqrt(pi))` | `-sqrt(q1*q2/(4*pi*epsilon*F))` |
| `x1*cos(theta1 - theta2) - sqrt(x**2 + x1**2*cos(theta1 - theta2)**2 - x1**2)` | `x1*cos(theta1 - theta2) - sqrt(x**2 - x1**2*sin(theta1 - theta2)**2)` |
| `(A + x2*y2 + x3*y3)/x1` | `0.999999995*(A/x1) + 0.999999986*(x2*y2/x1) + 1.000000024*(x3*y3/x1)` |
| `-c*p*sqrt(1/(c**2*m_0**2 + p**2))` | `-0.99999999977*p/sqrt(m_0**2 + (p/c)**2)` |

The first is the same expression, since `1/(2*sqrt(pi))` is `1/sqrt(4*pi)`. The
second is the same expression through the Pythagorean identity — a rearrangement
the benchmark was built to reward and that no library fit produces. These are the
symbolic recoveries the whole-category protocol never once produced, and they are
what the paper's symbolic-accuracy column is asking for. That column is still not
measured here, but the answers are now the right shape to be asked about.

Result file:
[`bench/results/era-srbench-per-problem-transform.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-srbench-per-problem-transform.json).

### What a bigger budget buys

The run above spends 5.26 model calls per problem. Raising it to **24
expansions and 20 s per problem** — about three times the calls and two and a
half times the compute — was run over the same 111 problems, same seed, same
splits, and nothing else changed:

| LSR-Transform, all 111 problems | seed | 6 expansions, 8 s | **24 expansions, 20 s** |
|---|---|---|---|
| model calls per problem | 0 | 5.26 | **16.46** |
| Acc(0.1) | 0.9% | 43.2% | **49.5%** (55 of 111) |
| median NMSE | 0.0941 | 5.68e-03 | **9.75e-06** |
| mean `min(12, -log10 NMSE)` | 1.074 | 5.391 | **6.271** |
| problems at 6+ digits | 0 | 44 | **52** |
| problems at the 12-digit cap | 0 | 41 | **48** |
| paired against the seed | — | 59 better, 1 worse | **75 better, 0 worse** (p = 5e-23) |
| median answer length | 9 terms | 3 terms, 126 chars | **2 terms, 98 chars** |
| wall / tokens | — | 3 810 s / 1.6M | 8 801 s / 5.6M |

Three times the budget buys **six points of Acc(0.1) and a 580-fold drop in
median NMSE** — the second number being the more informative one, because
Acc(0.1) is an indicator that a problem either clears or does not, while the
median NMSE says the typical answer went from "roughly right" to "right to five
decimal places". Seven more problems reach the 12-digit cap. Nothing regressed:
75 problems improved on the seed and **zero** got worse.

The answers also keep getting shorter — 9 terms under the whole-category
protocol, 3 at the small per-problem budget, 2 here, against a ground truth of
1. More search is not buying a longer fit; it is buying the right structure.

Result file:
[`bench/results/era-srbench-per-problem-transform-24x.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-srbench-per-problem-transform-24x.json).
Both sweeps write their result file **after every problem**, so an interrupted
run keeps what it has paid for and `--resume` picks it up; the budget is
fingerprinted in the file and a resume under a different one is refused.

### Against the paper's own tables

Two of this port's three programs can be compared to the paper's numbers over
the **full** category, which is what the paper reports on:

* the **seed program** is a fixed algorithm that never saw anything — no
  selection, no contamination, directly comparable;
* each **winner** was selected on 6 of that category's 8 shards, so its full-set
  row is optimistic and is marked as such; its clean number is the held-back
  column beside it.

Acc(0.1), best published result per column across all four methods × three
backbones in the paper's Table 2. Every row below this port's own is in
[`bench/results/era-srbench-fullset.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/era-srbench-fullset.json):

| Acc(0.1) ↑ | LSR-Transform (111) | chemistry (36) | biology (24) | physics (44) | materials (25) |
|---|---|---|---|---|---|
| Direct prompting, best backbone | 6.31% | 13.88% | 4.16% | 9.09% | 0.0% |
| SGA, best backbone | 8.11% | 16.66% | 12.51% | 9.09% | 36.11% |
| LaSR, best backbone | **50.45%** | 38.92% | 20.83% | 31.81% | 72.04% |
| LLM-SR, best backbone | 39.64% | 66.66% | 58.33% | **36.36%** | **88.28%** |
| **seed program here** (no LLM at all, full set) | 0.9% | 80.6% | 54.2% | 18.2% | 48.0% |
| **whole-category winner here** (full set, selection-contaminated) | 8.1% | **83.3%** | **79.2%** | 34.1% | 52.0% |
| whole-category winner here (held-back only, clean) | 14.8% | 87.5% | 83.3% | 33.3% | 50.0% |
| **per-problem here**, 5.26 calls/problem | 43.2% | — | — | — | — |
| **per-problem here**, 16.46 calls/problem | **49.5%** | — | — | — | — |

Median NMSE, same rows (the paper does not state how it aggregates its NMSE
column; median is used here, and the per-problem values are in the result files
so either aggregation can be recomputed):

| NMSE ↓ | LSR-Transform | chemistry | biology | physics | materials |
|---|---|---|---|---|---|
| best published (method, backbone) | 0.0011 (LaSR) | 4.12e-06 (LLM-SR) | 1.04e-06 (LLM-SR) | 7.62e-05 (LLM-SR) | 3.21e-09 (LLM-SR) |
| seed program here (full set) | 0.0941 | 5.47e-10 | 2.72e-10 | 3.91e-05 | 5.55e-08 |
| whole-category winner here (full set) | 0.0753 | **7.53e-13** | **7.78e-13** | **4.59e-06** | **1.22e-10** |
| per-problem here, 16.46 calls/problem | **9.75e-06** | — | — | — | — |

### The column this protocol does not contest

The paper reports **three** metrics and the two tables above compare two of them.
The third is **symbolic accuracy** — a GPT-4o judge asking whether the discovered
equation is mathematically the same as the ground truth — and it is not
implemented here, because it needs a judge model in the scoring loop that
nothing else in this port requires.

It is left out, but it is not left unsaid, because there is every reason to
think this protocol scores near zero on it. Measured on the held-back answers:

| | median terms | median characters |
|---|---|---|
| ground truth, LSR-Synth | 2 (max 5) | 104 |
| ground truth, LSR-Transform | 1 (max 2) | 33 |
| this port's seed and winner | 9–11 | 240+ (truncated in the files) |

The winner returns a nine-to-eleven-term library fit where the truth is
`0.316*P*exp(-0.054*t) + 0.316*P**2/(9.87*P + 1)`. It reaches NMSE 1e-13 on that
problem — and it has not discovered the equation, it has interpolated it. The
published methods' symbolic accuracy is low in absolute terms (LLM-SR's best
column is 31.53% on LSR-Transform, 11.11% on chemistry) but it is not zero, and
that difference is the whole distance between fitting and discovery.

So: on **data fidelity** this protocol is competitive with and often ahead of the
state of the art on LSR-Synth. On **symbolic discovery** it does not compete, and
a reader should assume it loses that column outright until someone implements the
judge and measures it.

Three more things fall out, and the third is the one worth arguing about:

**On LSR-Synth this protocol matches or beats the published state of the art on
four of five columns of NMSE and two of four on Acc(0.1)** — at 0.11 LLM calls
per problem, and on data fidelity only, with the symbolic-accuracy caveat above
attached. Chemistry and biology are ahead of every published method on both
metrics; physics is level on Acc and an order of magnitude better on NMSE.

**And a plain sparse-regression baseline gets most of the way there on its own.**
The seed program contains no LLM anywhere, and it scores 80.6% on chemistry
against LLM-SR's 66.66% and a median NMSE four orders of magnitude below it.
That is a fact about the benchmark rather than about this search: LSR-Synth
problems are ODE right-hand sides built as *a known term plus a novel term*,
which is the exact shape sequentially thresholded least squares was designed
for, and the published methods are working under a per-problem sample budget
that a direct least-squares fit does not have. Anyone quoting LSR-Synth numbers
should have that row in view.

**Materials science is where the two metrics disagree, and it is instructive.**
The winner's median NMSE there (1.22e-10) is better than LLM-SR's (3.21e-09),
while its Acc(0.1) (52.0%) is far worse than LLM-SR's (88.28%). Acc(0.1) is an
indicator on the *worst* relative error over 500 test points, so an equation that
is excellent in L2 and blows up relatively at a single near-zero target scores
0 — and a library-fit equation does that where a structurally correct one does
not. The two columns are measuring different virtues, and this protocol has the
first without the second.

**On LSR-Transform the whole-category protocol loses, and the protocol is why.**
The seed scores 0.9%, twelve expansions take it to 8.1% (14.8% on the held-back
shards), against LaSR's 50.45%. Those problems are Feynman equations rearranged
into products, quotients and nested roots — forms no linear-in-a-library method
can express at all, where proposing a *structure* from the variable names is the
entire task. Switching to the benchmark's own protocol closes it: **49.5%**, one
problem short of LaSR's 50.45% (55 against 56 of 111) and ahead of LLM-SR's
39.64%, with a median NMSE two orders of magnitude below LaSR's — at 16.46 model
calls per problem against LLM-SR's 250 prompts. The gap was the protocol's and
the budget's, not the tree search's, and it was tested rather than asserted.

## Run it

Preview without an API key, network access, or sandbox process:

```bash
python -m examples.era.era_empirical_software --dry-run
```

```bash
python -m examples.era.era_empirical_software --provider claude --model glm-5.2 \
    --yes --iterations 6 --workers 3 --async --async-ratio 1 --staleness full \
    --shards 8 --test-shards 4 --candidate-timeout 60 --max-tokens 16000
```

Add `--serial` for the upstream serial algorithm (one worker, nothing to merge),
or drop `--async` for the synchronous barrier. `--train-rows N` caps the
training file for a quicker look — it is a difficulty knob, so a capped run is
not comparable to an uncapped one.

`--provider claude` selects an Anthropic-shaped endpoint (`ANTHROPIC_BASE_URL`
+ `ANTHROPIC_API_KEY`); `--provider openai`, the default, uses
`OPENAI_BASE_URL` + `OPENAI_API_KEY`.

The integrals task takes the same flags, plus `--eval-budget` and
`--problem-seconds`:

```bash
python -m examples.era.era_hard_integrals --dry-run

python -m examples.era.era_hard_integrals --provider claude --model glm-5.2 \
    --yes --iterations 12 --workers 3 --async --async-ratio 1 --staleness full \
    --shards 8 --test-shards 4 --candidate-timeout 60 --max-tokens 16000 \
    --eval-budget 200000 --problem-seconds 5
```

LLM-SRBench takes the same flags, plus `--dataset`, `--problems`,
`--problem-seconds` and `--train-points`:

```bash
python -m examples.era.era_llm_srbench --dry-run

python -m examples.era.era_llm_srbench --provider claude --model glm-5.2 \
    --yes --iterations 12 --workers 3 --shards 6 --test-shards 2 \
    --problem-seconds 4 --candidate-timeout 120 --max-tokens 16000
```

`--dataset lsr_synth` (the default) runs the 129 synthetic problems, the only
ones with an OOD split; `--dataset lsr_transform` runs the 111 rearranged Feynman
equations; `--dataset all` runs both. `--problems N` caps the count evenly across
subsets — a difficulty and cost knob, so a capped run is not comparable to an
uncapped one, and every run records exactly which problems it used. Reading the
benchmark's parquet needs `pyarrow`.

`--per-problem` switches to **the benchmark's own protocol**: one independent
search per problem, rather than one program for the whole category.

```bash
python -m examples.era.era_llm_srbench --provider claude --model glm-5.2 \
    --per-problem --dataset lsr_transform --shards 6 --iterations 6 \
    --workers 3 --problem-concurrency 2 --problem-seconds 8 --yes
```

There, `--iterations` is expansions *per problem*, `--shards` is how many slices
of that problem's validation pool the search is gated on, and
`--problem-concurrency` trades endpoint load for wall-clock. Note that rollout
tasks are `shards * (1 - held_out_frac)` and a round proposes one expansion per
task, so `--shards 4 --workers 3` leaves a worker idle.

Offline tests: `tests/test_era_example.py`, `tests/test_era_integrals.py`,
`tests/test_era_hyp2f1.py`, `tests/test_era_srbench.py`.
