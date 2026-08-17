# Self-evolution algorithms — nineteen ports

AgentDescent is a *general* engine for parallel, merge-based evolution. To show
it is faithful to the field — not a toy — nineteen published **skill**,
**program** and **harness** self-evolution algorithms run on it, each as one
runnable example with a dedicated page. Eight reproduce their paper's own
benchmark; eleven preserve the mechanism on a compact domain and say so. What
each one follows and where it departs is recorded per port in
[port fidelity](port-fidelity.md).

**All nineteen run through the AgentDescent evolution engines.** No example
bypasses the engine, and they reach it two ways:

* the **eight benchmark-faithful ports** are each a custom `strategy=` and/or a
  custom `aggregator_factory=`, with their parent/gate rules extracted as named
  policy classes at the standard seams ([selection](selection.md),
  [acceptance](acceptance-policies.md));
* the **eleven microports and analogues** are declarative
  [`MethodPolicy`](policies.md) definitions over one shared runner — their
  mechanisms plug in as `Policies(...)` fields, their artifacts as shared
  [strategies](strategies.md), and the [runtime matrix](matrix-overview.md)
  measures them under all three schedulers.

**All nineteen are parallel — and can run async.** In synchronous mode their workers
run **concurrently** (overlapping LLM rollouts) with the aggregator merge as the
barrier (*synchronous data-parallelism*). Add **`--async`** and the same example
runs **barrier-free** through
[`async_evolve()`](evolution.md#the-barrier-free-runtime-async_evolve) — workers
never wait for the merge, and the staleness policy rebases/discards stale diffs.
Their custom optimizers keep shared state thread-safe, so both modes work
unchanged.

```bash
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5           # synchronous DP
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5 --async   # barrier-free
```

### The eight benchmark-faithful ports

| Algorithm | Port author | Kind | Domain (faithful) | `evolve()` plug-ins | Page |
|---|---|---|---|---|---|
| **ACE** (Agentic Context Engineering) | chendanyang | skill / context | FiNER-139 (XBRL tagging) | `strategy=ACEPlaybook`; Curator = default aggregator | [→](algo-ace.md) |
| **GEPA** (Reflective Prompt Evolution) | chendanyang | skill / prompt | HotpotQA (EM) | `aggregator_factory=` Pareto optimizer | [→](algo-gepa.md) |
| **EvoSkill** (Automated Skill Discovery) | chendanyang | skill library | OfficeQA (Treasury), FinQA without HF access | `strategy` + `aggregator_factory=` bounded top-K frontier | [→](algo-evoskill.md) |
| **SkillOpt** (ReflACT) | chendanyang | skill document | SearchQA (EM/F1) | `strategy` (edits) + `aggregator_factory=` strict gate | [→](algo-skillopt.md) |
| **ADAS** (Meta Agent Search) | chendanyang | harness (L1) | MGSM, GPQA Diamond | `strategy` + `aggregator_factory=` keep-all archive | [→](algo-adas.md) |
| **DGM** (Darwin Gödel Machine) | chendanyang | harness (L1) | SWE-bench Verified ids; vendored bugs w/ pytest | `strategy` + `aggregator_factory=` archive + selection | [→](algo-dgm.md) |
| **OpenEvolve** (Program Evolution) | cyanneko | program (L1) | Function minimization | `strategy` + `aggregator_factory=` MAP-Elites islands | [→](algo-openevolve.md) |
| **ERA** (Empirical Research Assistance) | chendanyang | program (L1) | Kaggle Playground S3E1 (RMSE) | `strategy` + `aggregator_factory=` FUTS tree, `selection.FlatPuct` | [→](algo-era.md) |

### The eleven microports and analogues

All eleven are `MethodPolicy` definitions over
[`examples/_method_runner.py`](https://github.com/Birfy/agentdescent/blob/main/examples/_method_runner.py),
so they share a runner, a budget contract and a command line. Port author:
`cyanneko`.

| Algorithm | Fidelity class | Domain | Mechanism seams | Page |
|---|---|---|---|---|
| **PromptBreeder** | `mechanism_microport` | GSM8K | binary tournament as the population layer; `FieldSlots` genome | [→](algo-promptbreeder.md) |
| **AFlow** | `mechanism_microport` | GSM8K | `SoftMixed` selection; per-parent experience | [→](algo-aflow.md) |
| **Self-Refine** | `mechanism_microport` | GSM8K | two-call FEEDBACK→REFINE, stop signal | [→](algo-self-refine.md) |
| **Reflexion** | `mechanism_microport` | GSM-Hard | `WindowedMemory` (bounded append-only) | [→](algo-reflexion.md) |
| **SICA** | `self_edit_analogue` | GSM-Hard | AST gate; `Archive('best')` selection | [→](algo-sica.md) |
| **Gödel Agent** | `self_edit_analogue` | GSM-Hard | AST gate; optional `--gateless` acceptance | [→](algo-godel-agent.md) |
| **Voyager** | `environment_analogue` | crafting world | `SkillLibrary`, `DifficultyWeighted`, self-verify critic | [→](algo-voyager.md) |
| **SkillWeaver** | `environment_analogue` | settings site | `SkillLibrary`, `DifficultyWeighted`, self-verify reward model | [→](algo-skillweaver.md) |
| **Absolute Zero** | `inference_analogue` | self-play carts | frozen self-play evaluation; learnability signal | [→](algo-absolute-zero.md) |
| **R-Zero** | `inference_analogue` | self-play carts | `AdvantageAcceptance` (GRPO shape), `DifficultyWeighted` | [→](algo-r-zero.md) |
| **Agent0** | `inference_analogue` | self-play carts | `DifficultyWeighted`; calculator stop-and-go | [→](algo-agent0.md) |

### The shared command line

Every port's shared flags have one definition, in
[`add_standard_args`](https://github.com/Birfy/agentdescent/blob/main/examples/_common.py),
and the behaviour behind each lives there too — a flag declared centrally and
honoured locally is how a port grows a `--yes` it never reads.

| flag | what it does | where the behaviour lives |
|---|---|---|
| `--provider` / `--model` | pick Claude or any OpenAI-compatible endpoint | `completion_for` |
| `--seed` | the run's seed | the port |
| `--async` / `--async-ratio` / `--max-seconds` | barrier-free runtime and its lag budget | the port |
| `--serial` | **the upstream algorithm's own semantics**: one worker, nothing to merge | `worker_count` |
| `--budget-rollouts` | total rollouts, held fixed as workers vary | `budget_kwargs` |
| `--val-cap` | shrink the gate's split without shrinking test | `capped_val` |
| `--eval-concurrency` | held-out evaluations in flight at once; wall-clock only | the port |
| `--reflective-merge` | merge contradicting diffs with a model instead of ranking them | `merge_kwargs` |
| `--eval-cache DIR` | memoise held-out scores across processes; off by default | `eval_cache_kwargs` |
| `--no-thinking` | ask an Anthropic-shaped endpoint for no reasoning tokens | `completion_for` |
| `--dry-run` | print the plan, with no model call | the port's early return |
| `--yes` | skip the confirmation before real API calls | `confirm` |

The iteration count is deliberately *not* standardised: `--rounds` (ACE, GEPA),
`--generations` (ADAS, DGM), `--iterations` (EvoSkill, OpenEvolve, ERA), `--steps`
(SkillOpt) and `--candidates` (the eleven) each keep their own vocabulary, which
for the eight is part of being a faithful port. `--budget-rollouts` maps onto
whichever one a port uses, so a sweep can pin every arm with one flag.

`--serial` is the control every one of these ports was missing. They all
parallelise an algorithm that was published as a serial loop, and until this flag
existed none of them could run that loop — so every claim about parallelising
them had no baseline in the repository at all. It is refused together with
`--async`: the barrier-free runtime's concurrency *is* `n_workers`, so
`--serial --async` is a one-worker *asynchronous* run whose diffs can still go
stale against a moved head, and staleness in the control arm is the one thing a
control must not have.

**`--serial` on its own is still not a comparison.** Six of these eight ports pass
a fixed iteration count and let the worker count multiply it, so an `N=8` run does
eight times the rollouts of the serial one — measured on the engine at
`rounds=24`: 192 rollouts against 24. A wall-clock read across that gap is eight
times the model spend reported as parallel efficiency. `--budget-rollouts N` pins
both arms to the same total; pass it to both, at the same value, and report
`result.rollouts` rather than the budget, because the synchronous path checks at
the round barrier and can overshoot by up to `n_workers - 1`.

OpenEvolve is the exception and needed no fixing: it derives
`rounds = iterations // workers`, so its total work was already fixed and the flag
simply sets `--iterations`.

Every example takes `--dry-run`, which prints its configuration and returns
**without a model call and without an API key**. Whether it also avoids the
network depends on which runner it is on: the eight return before any dataset is
touched, and say so (`Data: deferred`). The eleven build their `MethodPolicy`
first, so a port whose domain is a real benchmark — PromptBreeder, AFlow,
Self-Refine, Reflexion, SICA, Gödel Agent — loads and caches its split during a
dry run, and prints that it did.

Datasets go through the shared [**`agentdescent.dataloader`**](dataloader.md)
layer — dependency-free (`urllib` only), cached under `~/.cache/agentdescent/`,
from each benchmark's canonical source. Every port has an offline test suite
exercising its pure logic, named on its page. Where a paper's full setup needs
heavy infrastructure, the boundary is documented in the example's module
docstring — never hidden.

### The MethodPolicy command line

The eleven declarative ports share
[`build_parser`](https://github.com/Birfy/agentdescent/blob/main/examples/_method_runner.py),
so their command line is the same on all eleven: the shared flags above plus
`--workers`, `--candidates` (a synonym for `--budget-rollouts`),
`--no-reflective-merge`, `--staleness`, `--temperature`, `--max-tokens` and
`--timeout`. Two ports add one switch of their own — Reflexion's
`--per-instance` and Gödel Agent's `--gateless`, both controls for a declared
departure.

| flag | what it does here |
|---|---|
| `--async-ratio N` | the lag budget, passed to `async_evolve`. **Defaults to 1**, not the shared 3 — see below |
| `--eval-concurrency N` | held-out evaluations in flight at once; wall-clock only. Left off, the runner's own rule applies: **1** under `--serial`, `--workers` otherwise |
| `--eval-cache DIR` | memoise the gate to a directory two processes can share, merged onto the method's own `Policies` bundle. Off by default: a cache that outlives the run makes a rerun return the first run's numbers |
| `--val-cap` | **not offered.** These ports freeze train/held-out/test in `build()`, before the parser is consulted, so it now fails as an unrecognised argument rather than parsing and moving nothing |

The first three used to be declared by `add_standard_args` and never passed to
`run_port`, so on these eleven a run that set all three was byte-identical to one
that set none — the same defect as Gödel Agent's `--gateless`, which five
documents described while the parser rejected it. All four are now wired or
withdrawn, and `tests/test_method_runner_flags.py` enumerates the parser and
fails on a flag with nowhere recorded that reads it, so the next one added here
cannot be decorative. `run_port` also records all three in `framework`, and
`--dry-run` prints them: a flag absent from the plan is a flag nobody checks.

!!! note "Why `--async-ratio` defaults to 1 here and 3 for the eight above"
    Two entry points reach `run_port` — this command line and
    `bench.candidate_methods` — and a lag budget on which they disagree is a
    trap, because the same nominal configuration would then run two different
    searches depending on which one launched it. `run_port`'s signature says 1
    and `bench.candidate_methods` defaults to 1, so this parser says 1 too;
    `add_standard_args`' 3 stays the default for the eight ports above, which
    is where it was measured. It is one argument away and it is a real change,
    not a label: the budget bounds both how far a worker's snapshot may drift
    behind head *and* how many cards may sit un-merged ahead of the merger.

!!! warning "Fifteen recorded rows said `async_ratio: 2` and ran at 1"
    They were attributed to `bench.candidate_methods`, which passes the flag
    through. They did not come from there. **`bench.candidate_methods` has no
    `--staleness` and never passes `staleness=` to `run_port`**, so every run
    through it is `guarded` — and all fifteen blocks record `full`.

    What they did come from is the line `standard_main` prints, and that is
    checkable rather than argued. It formats qualities at `.3f`, seconds at
    `.1f` and calls as an int, and across all 45 cells in those files not one
    value carries more precision than that; its fields are `test`/`validation`
    pairs, `accepted`, `invalid`, `wall_s`, `engine_s`, `calls`, which is the
    cell schema exactly. `bench.candidate_methods` JSON-dumps
    `MethodRunResult.compact()` unrounded — in the one file it produced, 198 of
    198 wall/engine values are full floats. That command line dropped
    `--async-ratio`, so the runs took the runner's default of 1.

    Those files now record 1, with a note. The **Run it** commands pass
    `--async-ratio 1` explicitly for the same reason a config block should never
    have been typed from memory: the value matters and a default can move.

    The underlying defect was not the flag. It was that a run printed what it
    *reached* and nothing about how it was *set up*, so the config block beside
    those numbers had to be remembered. A live run now prints its resolved
    configuration as one JSON line, in these key names, to be copied:

    ```
    config: {"arm": "async_pipeline", "seed": 0, "budget_rollouts": 80, "workers": 8,
             "async_ratio": 1, "staleness": "full", "reflective_merge": true, ...}
    ```

---

## Measured results — all nineteen

Every port has been run and every number below is linked to the run that
produced it. **The two halves of this page are not comparable with each other**,
and within each half only rows on the same domain are: a gain is bounded by the
headroom its baseline leaves, and the ports sit on benchmarks whose baselines
run from 0.000 to 0.641.

### The eleven, on one runner and one budget

Three seeds each, `async_pipeline`, 80 rollouts, 8 workers, `--staleness full`,
`deepseek-v4-flash` at temperature 0.7 with thinking disabled. Test columns are
the mean over the three seeds; `moved` counts seeds whose held-out test score
rose at all.

| Method | Domain | test, before → after | gain | moved | accepted | calls / seed |
|---|---|---:|---:|---:|---:|---:|
| [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k) | GSM8K | 0.474 → **0.969** | +0.495 | 3/3 | 8/240 | 1444 |
| [AFlow](algo-aflow.md#measured-results-gsm8k) | GSM8K | 0.510 → **0.969** | +0.458 | 3/3 | 10/240 | 2069 |
| [Self-Refine](algo-self-refine.md#measured-results-gsm8k) | GSM8K | 0.552 → **0.943** | +0.391 | 3/3 | 5/240 | 1050 |
| [Reflexion](algo-reflexion.md#measured-results-gsm-hard) | GSM-Hard | 0.583 → **0.599** | +0.016 | 2/3 | 3/240 | 465 |
| [SICA](algo-sica.md#measured-results-gsm-hard) | GSM-Hard | 0.641 → **0.667** | +0.026 | 2/3 | 5/240 | 899 |
| [Gödel Agent](algo-godel-agent.md#measured-results-gsm-hard) | GSM-Hard | 0.625 → **0.687** | +0.062 | 2/3 | 9/240 | 985 |
| [Voyager](algo-voyager.md#measured-results-crafting-world) | crafting world | 0.000 → **0.667** | +0.667 | 2/3 | 3/240 | 1234 |
| [SkillWeaver](algo-skillweaver.md#measured-results-settings-site) | settings site | 0.000 → **0.771** | +0.771 | 3/3 | 8/240 | 1170 |
| [Absolute Zero](algo-absolute-zero.md#measured-results-self-play-carts) | self-play carts | 0.042 → **0.354** | +0.313 | 3/3 | 9/240 | 624 |
| [R-Zero](algo-r-zero.md#measured-results-self-play-carts) | self-play carts | 0.062 → **0.292** | +0.230 | 3/3 | 6/240 | 937 |
| [Agent0](algo-agent0.md#measured-results-self-play-carts) | self-play carts | 0.042 → **0.542** | +0.500 | 3/3 | 6/240 | 1626 |

**Read the columns against each other, not down the gain column.** The three
GSM8K rows converge on 0.94–0.97 from a real model baseline and separate on
*cost*: AFlow spends twice Self-Refine's calls for 0.026 more. The three
GSM-Hard rows move a fraction as far because their headroom is a fraction as
wide — and the noise floor there is ±0.02 against GSM8K's ±0.09, which is why a
+0.026 on GSM-Hard is a result and a +0.05 on GSM8K would not be. The two
environment analogues start at 0.000 because a seed agent that has discovered no
skill solves none of their goals; the three self-play rows do not, because a
trusted renderer generates their evaluation carts and a fresh solver already gets
some of them — which is also why neither of those groups has a 1.000 ceiling to
read a final score against.

**`accepted` is the shape of the mechanism, not a yield.** Between 3 and 10 of
240 proposals commit across every row, because the gate scores each candidate on
the full held-out split and admits only a strict improvement. One accepted skill
is all Voyager's run needs.

### The eight, each on its own benchmark

Different budgets, different datasets, one seed each — these are single runs
that establish the port works end to end, not a comparison.

| Method | Domain | measured | budget |
|---|---|---|---|
| [ACE](algo-ace.md#measured-results-finer-139) | FiNER-139 | val 0.719 → **0.766**, test 0.786, 10 bullets curated | async N=4, 120 rollouts |
| [GEPA](algo-gepa.md#measured-results-hotpotqa) | HotpotQA | **1.85× / 3.22×** concurrency (sync / async) against a true serial arm; test EM 0.600 → 0.850 on 20 items | 16 rollouts pinned on all three arms |
| [EvoSkill](algo-evoskill.md#measured-results-finqa) | FinQA | val 0.527 → **0.707**, test 0.633, the frontier filled 5/5 | async N=4, 120 rollouts |
| [SkillOpt](algo-skillopt.md#measured-results-searchqa) | SearchQA (`--hard`) | val 0.053 → **0.211**, test 0.316, 3 edits accepted of 43 | async N=4, 60 rollouts |
| [ADAS](algo-adas.md#measured-results-gpqa-diamond) | GPQA Diamond | **no lift number** — MGSM is saturated and GPQA costs 49 s / 5,116 tokens per call, and the two constraints are opposed | — |
| [DGM](algo-dgm.md#measured-results-vendored-bugs-objective-real) | vendored bugs, real pytest | seed agent 0.844 → best archived child **0.906** held out; its own `solve.py` 18 → 79 lines | async N=2, 16 rollouts |
| [OpenEvolve](algo-openevolve.md#measured-results-function-minimization) | function minimization | combined score 0.9638 → **1.4995** against a 1.5 ceiling, held-out seeds | async N=4, 24 rollouts |
| [ERA](algo-era.md#measured-results-playground-s3e1) | Kaggle Playground S3E1 | test RMSE 0.7297 → **0.5913** (−19.0%) on 2,476 unseen rows; 7-node tree, all valid | async N=3, 6 expansions |

!!! warning "One run per seed, and one seed on the bottom table"
    Nothing here is a paper-scale result. The eleven carry three seeds and report
    the spread on their own pages; the eight carry one, and a single run does not
    pin a number on a sampled model. The quality columns are evidence the
    mechanism runs and moves the metric it should, not evidence about how much.

---

## Mechanism coverage

Every mechanism family in the original backlog now has at least one implemented
port:

| Mechanism | Benchmark-faithful | Microports / analogues |
|---|---|---|
| Evolution / program search | [OpenEvolve](algo-openevolve.md) | [PromptBreeder](algo-promptbreeder.md), [AFlow](algo-aflow.md) |
| Reflection / refinement | [GEPA](algo-gepa.md) (reflective) | [Reflexion](algo-reflexion.md), [Self-Refine](algo-self-refine.md) |
| Skills / lifelong learning | [EvoSkill](algo-evoskill.md), [SkillOpt](algo-skillopt.md), [ACE](algo-ace.md) | [Voyager](algo-voyager.md), [SkillWeaver](algo-skillweaver.md) |
| Self-play / unlabeled data | — | [Absolute Zero](algo-absolute-zero.md), [R-Zero](algo-r-zero.md), [Agent0](algo-agent0.md) |
| Self-modifying code / harness | [DGM](algo-dgm.md), [ADAS](algo-adas.md) | [SICA](algo-sica.md), [Gödel Agent](algo-godel-agent.md) |

The label-free path now exists — the three self-play ports derive reward from a
grounded local verifier with no gold labels — but only as inference analogues:
a benchmark-faithful label-free port (real RL updates, real task domains)
remains open. TextGrad remains unimplemented in the reflection family.

### Deferred pending released code

* **CoEvoSkills** (arXiv:2604.01687, "Self-Evolving Agent Skills via
  Co-Evolutionary Verification") — the Skill Generator + co-evolving Surrogate
  Verifier + opaque pass/fail oracle is a compelling fit for AgentDescent's
  aggregator, but the **official code is unreleased** ("coming soon") and its
  benchmark (SkillsBench) requires a Claude Code / Codex agent harness. With no
  original repo to be faithful to, it is intentionally deferred until the authors
  release code, rather than reconstructed and mislabelled as faithful.

## Fidelity principles

Use the short [porting checklist](porting-checklist.md) before adding a row here,
and record the result in [port fidelity](port-fidelity.md) after — the checklist
is the standard, that page is what each port was actually found to do.

1. **Faithful to the repo, not just the paper.** Where the released code diverges
   from the paper's claims (e.g. EvoSkill's frontier is top-K aggregate, not
   per-instance Pareto), the example follows the **code** and says so.
2. **Declared dataset identity.** A benchmark-faithful port uses the paper's own
   benchmark from its canonical source; a microport or analogue uses a compact
   or substituted domain and carries the fidelity class that says which it is.
3. **Documented boundaries.** Where the full setup needs heavy infra (AppWorld,
   SWE-bench Docker, gated data), the example states the boundary and, when
   needed, substitutes a clearly-labelled surrogate — the algorithm stays
   faithful and runnable; nothing is passed off as a benchmark result it is not.
