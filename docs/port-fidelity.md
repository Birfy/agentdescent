# Port fidelity — what each port follows, and where it departs

Eighteen published self-evolution algorithms run on this engine — seven as benchmark-faithful ports, eleven as declared microports and analogues. Every one of them
was published as a **serial** loop, and every one of them here runs in parallel
with a merge step the original does not have. That is only an interesting claim
if the algorithm is otherwise untouched — a "parallelised GEPA" that quietly
replaced GEPA's Pareto selection is not evidence about parallelisation, it is a
different algorithm with a familiar name.

So this page is the record of what "untouched" means, per port. It is the
*after*: the observed differences. [`porting-checklist.md`](porting-checklist.md)
is the *before*: the standard a new port is held to. Each section reads the same
way — **paper says / released code does / this port follows** — because that is
the axis the differences actually fall on, and because the answer is not always
"the paper".

!!! info "The rule, and why it is the code"
    Where a paper and its released code disagree, these ports follow the **code**
    and say so. A port faithful to a paper that the authors' own implementation
    contradicts reproduces something nobody ran.

!!! note "Fidelity is a per-port property, not a tier"
    All eighteen ports sit side by side in
    [Self-evolution algorithms](self-evolution-examples.md); what differs is
    each port's recorded fidelity class. Eleven of them run compact domains or
    substituted environments and say so on their pages -- a mechanism
    microport or an environment, inference, or self-edit analogue is not a
    faithful benchmark port, and the [runtime matrix](matrix-overview.md)
    measures them without pretending otherwise.

---

## What parallelising is allowed to change

One answer, for every port: **the scheduling of rollouts and the timing of the
merge.** Nothing else.

| may change | must not change |
|---|---|
| how many proposals are in flight at once | which candidate is selected next |
| when diffs are merged back (per round, or barrier-free) | how a candidate is scored |
| where rollouts execute (threads, processes, sandboxes) | the acceptance/admission rule |
| — | the dataset, its split, or the metric |

The "semantics changed" column of the matrix below is expected to read *"rollout
scheduling and merge timing only"* on every row. A row saying anything else is
either a bug or a finding, and either way it has to be written down.

!!! note "That column is generated, not typed"
    It is [`bench.matrix_run.SEMANTICS`](https://github.com/Birfy/agentdescent/blob/main/bench/matrix_run.py)
    — one entry per row **per arm**, declared beside the rows the sweep runs, and
    `bench/matrix_report.py` refuses to render a row that has no entry. A column
    filled in afterwards is filled in from memory by whoever is writing up the
    results, with the numbers already on the page; a blank one reads as "nothing
    changed" rather than as "nobody checked".

    Auditing the arms that way found one: **EvoSkill's async arm ran a different
    admission rule from its own serial arm** (below). All seven ports now use the
    same aggregator on all three arms, which is what that column exists to
    enforce.

Until [#75](selection.md)'s selection seam is wired to a multi-head ledger, each
port's candidate-selection rule still lives in its own example file — so for now
"we did not change it" is a claim a reader verifies by reading, and the last
column of each section below says exactly where to look.

---

## ACE — Agentic Context Engineering

* **Paper**: an evolving *playbook* of bullets, grown and refined incrementally.
* **Released code**: incremental delta updates, never a monolithic rewrite; a
  de-duplication pass on the bullet set.
* **This port follows**: the code. `ACEPlaybook` is a `Strategy` whose ops are
  per-bullet, so two workers editing different bullets do not conflict; the
  Curator role is the engine's default aggregator.
* **Departures**: the dataset is FiNER-139 as upstream, with `--top-k` capping
  the concept vocabulary — a *difficulty* knob, not an algorithm change, and one
  the page documents because the measured lift depends on it (nothing to learn
  at 10; the baseline only leaves headroom once `--pool` is wide enough to
  surface rare concepts).
* **A departure that was the acceptance rule, now fixed and behind a flag.**
  Upstream's Curator (`ace/core/curator.py`) validates the Reflector's output
  *structurally* — reasoning is a string, operations is a list — and applies it.
  There is no held-out evaluation of a bullet before it enters the playbook;
  utility is tracked afterwards by per-bullet `helpful=X harmful=Y` counters and
  size by de-duplication plus a token budget. This port instead ran the engine's
  shipped `DefaultAcceptance`, a Beta posterior requiring each bullet to raise
  held-out reward — which is the **acceptance rule**, the row the table above
  puts under *must not change*.

  It is not a harmless substitution. A bullet teaches one XBRL concept, so it can
  only move a validation split containing that concept: measured, five committed
  bullets covered 4 of 32 val tasks, and widening the gate to 64 tasks dropped
  commits from 5 to **0**. The gate becomes more correct as it gains statistical
  power, and the playbook empties — while ACE's entire claim is accumulation.
  `--grow-and-refine` restores upstream's rule; it is opt-in because turning it
  on changes what a row measures. See [algo-ace.md](algo-ace.md#measured-results-finer-139).
* **Selection rule lives in**: nowhere separate — ACE has no candidate archive.
* **Details**: [algo-ace.md](algo-ace.md)

## GEPA — Reflective Prompt Evolution

* **Paper**: reflective mutation over a multi-module compound system, with
  Pareto selection over held-out instances and a rollout budget.
* **Released code**: the Algorithm-1 acceptance test is a **minibatch of one**.
* **This port follows**: the code, with the compound system reduced to a single
  instruction module; the minibatch is the per-round worker sample, so
  `--workers` is the minibatch size. The Pareto set is the held-out split.
* **Departures**: one module rather than many. Recorded because "GEPA on
  HotpotQA" without it would imply the multi-module search ran.
* **Selection rule lives in**: `ParetoWinFrequency` — GEPA's Algorithm-2 sampling as a named `SelectionPolicy` ([selection](selection.md)); `ParetoAggregator` delegates to it
  `examples/gepa/gepa_prompt_evolution.py` — per-instance domination, the
  `per_instance` mode of [`ParetoFrontier`](selection.md).
* **Details**: [algo-gepa.md](algo-gepa.md)

## EvoSkill — Automated Skill Discovery

* **Paper**: per-instance Pareto selection, and joint skill + prompt mutation.
* **Released code**: **neither**. `manager.py:update_frontier` is a leaderboard
  on one scalar — mean validation accuracy — admitting if the frontier has room
  and otherwise replacing the worst member only if strictly greater. Skills
  mutate; prompts do not.
* **This port follows**: the code. Top-K aggregate frontier, and the unit-aware
  numeric scorer with its exact tolerance ladder
  (`[0.05, 0.01, 0.1, 0.0, 0.025]`, weight `1/(1+20·tol)`).
* **Why this one matters most**: it is the clearest case of the rule. A port
  faithful to the paper here would be a *better-sounding* algorithm that the
  authors' own code does not implement.
* **A departure that was the admission rule, now removed.**
  `run_evoskill` used to pick its aggregator off the `asynchronous` flag:
  `TopKFrontierAggregator` (upstream's rule) on the serial and synchronous arms,
  an SGD-style `SgdSkillAggregator` on the asynchronous one. The second applied
  every proposed skill to the head immediately, amortised held-out validation
  over `val_every` steps, rolled a whole batch back when it failed to improve —
  and had **no frontier at all**, one checkpoint in its place.

  That is three of upstream's mechanisms at once, and the frontier is not an
  implementation detail: `update_frontier` plus `select_from_frontier` *is*
  EvoSkill. Upstream evaluates each child on the full validation split before
  the admission decision, so a barrier-free schedule changes *when* those
  evaluations happen, not whether they do — there was never a reason for the
  schedule to pick the optimizer. The frontier now runs on every path and the
  SGD variant is gone rather than optional. Pinned by
  `tests/test_matrix_report.py::test_the_evoskill_frontier_is_the_algorithm_on_every_arm`,
  which reads the source and fails if the aggregator is keyed off the schedule
  again. The mechanism itself is
  [written up in the aggregator page](aggregator.md#the-async-optimizer-variant-sgd-style-descent)
  as a thing to build deliberately, never to install by schedule.
* **Selection rule lives in**: `FrontierBest` — the frontier's best member as a
  named `SelectionPolicy` in `examples/evoskill/evoskill_skill_discovery.py`; the
  bounded top-K admission stays on `Frontier`, the `topk_aggregate` mode of
  [`ParetoFrontier`](selection.md).
* **Details**: [algo-evoskill.md](algo-evoskill.md)

## SkillOpt / ReflACT — Skill-Document Training

* **Paper and released code** agree: edit-based updates to a skill document,
  with a learning-rate schedule on the number of edits per step.
* **This port follows**: both, including the cosine schedule (`--lr-mode`).
* **Departures**: `--hard` filters SearchQA to questions the seed skill answers
  wrong. This is a *measurement* decision, not an algorithm one — plain SearchQA
  scores ~0.900 for a strong model, leaving no headroom to detect a change — and
  it is stated because a reported lift on a filtered set is not a lift on the
  benchmark. It takes `--hard-passes` measurements per item (default 3) because
  filtering on one selects the model's unlucky answers rather than hard
  questions: at one pass the filtered val split scored **1.000** on re-measurement
  and the strict gate could accept nothing above it.
* **Checked against upstream, and clean.** All four load-bearing invariants match
  the released code line for line: the op set `{append, insert_after, replace,
  delete}` (`optimizer/skill.py`), the strict `candidate > current` gate with no
  tolerance or tie-break and `gate_metric=hard` by default
  (`evaluation/gate.py`), the textual learning rate as an integer cap on edits
  per step (`optimizer/scheduler.py`), and the per-epoch rejected-edit buffer fed
  back to the optimizer (`engine/trainer.py`'s `_format_step_buffer`). Unlike ACE
  and EvoSkill, nothing here had to be put back.
* **Selection rule lives in**: best-of-batch in the strict-gate aggregator;
  the gate itself is `StrictImprovement`, a named `AcceptancePolicy`
  ([acceptance](acceptance-policies.md)).
* **Details**: [algo-skillopt.md](algo-skillopt.md)

## ADAS — Meta Agent Search

* **Paper and released code**: the meta agent writes Python `forward()`
  functions, which are `exec`'d.
* **This port follows**: the loop, the seed archive, MGSM scoring and the
  keep-all archive — but **not** the substrate. An agent here is a composable
  control-flow program in a small validated DSL (`AGENT_BLOCKS`) run by a safe
  interpreter.
* **Departures**: this is the largest departure of any port, and it is a safety
  one: running model-written Python through `exec` is arbitrary code execution.
  It bounds what the port demonstrates — the *search* is faithful, the space it
  searches is smaller than upstream's.
* **Checked against upstream, and clean otherwise.** Keep-all archive, the
  meta-agent conditioned on the entire archive *with* fitness, exactly two
  Reflexion refinement rounds, bootstrap-CI fitness, and the seven hand-designed
  seeds by name (`_mgsm/search.py`, `_mgsm/mgsm_prompt.py`) all match. `--dataset
  gpqa` is a *domain* change within ADAS's own four, not a dataset swap: GPQA
  Diamond ships in the ADAS repo.
* **Not measurable on `deepseek-v4-flash`, and the page says why.** MGSM is
  saturated in every language (CoT 1.000), which does not merely remove the lift
  — it ties all seven seeds and so removes the meta-agent's conditioning signal.
  GPQA has headroom but costs 49 s and 5,116 completion tokens per call, and a
  candidate is `|val| x program_cost` of those. Shrinking `|val|` to afford it
  returns the split to a ceiling. The two constraints are opposed.
* **Selection rule lives in**: the shipped [`Beam(1)`](selection.md) over the
  keep-all archive — best-of-archive, byte-identical to the inline
  strictly-greater tracking it replaced.
* **Details**: [algo-adas.md](algo-adas.md)

## DGM — Darwin Gödel Machine

* **Paper and released code**: an archive of self-modifying agents, parents
  sampled by performance × novelty, scored by running candidate patches inside
  the SWE-bench Docker harness.
* **This port follows**: the algorithm exactly, including the parent-selection
  weights (`DGM_outer.py:91-100`: a sigmoid steep at ×10 centred on 0.5, divided
  by `1 + children`) and the staged-eval subset sizes (10 / 50 / 140 with
  `test_more_threshold = 0.4`) — both pinned by `tests/test_port_fidelity.py`
  against the upstream lines.
* **Departures**: the objective is a **transparent surrogate**, not SWE-bench.
  Each real instance has a latent required-capability set the agent must cover.
  The algorithm runs and is tested offline; the *scores* are simulated. Pass a
  real `evaluate_fn` to `run_dgm` for the actual harness.
* **A past departure, now fixed**: the port once had a third staged-eval rung
  upstream's self-improve loop does not use. It passes exactly two subsets, and
  the test above pins that it stops at medium.
* **The surrogate objective is monotone, and `--objective real` is the way out.**
  Hashing an instance id into a capability set means adding a capability can
  never un-resolve a task, so a self-modification can never regress -- which
  removes the reason DGM keeps an archive at all. `--objective real` evolves the
  agent's own Python source against vendored bugs with real pytest runs: not
  SWE-bench, but real execution, real regressions, and self-edits that can leave
  the agent unable to run. Measured there, the seed scores 0.844 and the best
  archived child 0.906, and an earlier run archived children at 0.875 and 0.500
  -- the worse one kept, which is what `keep-all` is for and what the surrogate
  cannot produce. See [algo-dgm.md](algo-dgm.md#measured-results-vendored-bugs-objective-real).
* **Selection rule lives in**: `DGMParentSelection` — `sigmoid(10·(s−0.5)) × 1/(1+children)` as a named `SelectionPolicy` over the archive
  `examples/dgm/dgm_self_improve.py` — [`Archive`](selection.md) with
  `sampling="novelty"`.
* **Details**: [algo-dgm.md](algo-dgm.md)

## OpenEvolve — Program Evolution (AlphaEvolve)

* **Reference**: commit `411fb59c886c18704caaffb611e17cf9e7d824d2`,
  `examples/function_minimization` plus the database implementation.
* **This port follows**: Python source as the genome; the evaluator's value /
  distance / reliability / basin-multiplier formula verbatim; exploitation-vs-
  exploration parent mixing; per-island MAP-Elites grids over program length and
  code diversity; children staying on their island with ring migration of elites.
* **Departures**: the genome is rewritten whole rather than patched with
  SEARCH/REPLACE; feature bins use fixed length boundaries and insertion-time
  token-Jaccard diversity instead of evolving min/max scaling; candidate
  execution is deterministic, budgeted, AST-gated and sandbox-isolated —
  Bubblewrap on Linux, Seatbelt (`sandbox-exec`) on macOS, and a refusal to run
  at all on a host with neither. Both backends deny network access and confine
  writes to a scratch directory; the CPU / memory / file-size / fd limits come
  from `setrlimit` in the runner, so they are identical on both, except that
  Darwin has no `RLIMIT_AS` and the runner reports the refusal rather than
  claiming it applied.
* **Selection rule lives in**: `EpsilonGreedy` for the in-pool pick, on the MAP-Elites island archive (the archive structure is the mechanism and stays)
  `examples/openevolve/openevolve_program_evolution.py`.
* **Details**: [algo-openevolve.md](algo-openevolve.md)

---

## The parallelisation matrix

What the seven ports can jointly show that no single one can: **parallel merging
is not tied to this repository's own domains — it is a layer that goes around an
existing self-evolution algorithm.** That is the answer to "why would I use this
instead of just using GEPA", and the answer is that it is not a choice between
them.

Every port now takes [`--serial`](self-evolution-examples.md#the-shared-command-line),
which runs the published loop: one worker, nothing to merge. That is the control
column. Without it the speedups already in [results](results.md) had nothing to
be speedups over.

!!! danger "`--serial` alone does not make the two arms comparable"
    Six of the seven ports pass a **fixed `rounds`** and let `n_workers` multiply
    it, so an `N=8` arm performs **eight times the rollouts** of the `--serial`
    arm. Measured on the engine directly, `rounds=24`:

    | workers | no budget | `--budget-rollouts 24` |
    |---|---|---|
    | 1 | 24 rollouts | 24 |
    | 2 | 48 | 24 |
    | 4 | 96 | 24 |
    | 8 | **192** | 24 |

    Comparing wall-clocks across the left column reports eight times the model
    spend as parallel efficiency, and comparing final quality credits the extra
    spend to parallelism. It is the confound
    [`agentdescent.baselines`](results.md) was built to remove, and the warning
    on that page — that a speedup table cannot distinguish merging from sampling
    — applies here first.

    **So every cell below has to be produced with `--budget-rollouts`, on both
    arms, at the same value.** `evolve(max_rollouts=)` has existed since the
    equal-budget work and no port passed it;
    `tests/test_example_entrypoints.py::test_every_port_can_hold_its_rollout_budget_fixed`
    now refuses a port that cannot. The synchronous path checks at the round
    barrier, so an `N`-worker arm may overshoot by up to `N-1` — report
    `result.rollouts`, not the budget.

| Algorithm | Dataset | Serial (upstream) | AgentDescent N=8 | Speedup | Final held-out Δ | Semantics changed |
|---|---|---|---|---|---|---|
| ACE | FiNER-139 | — | — | — | — | scheduling and merge timing; budget must be pinned |
| GEPA | HotpotQA | 609 s / 85 calls / **1.00× concurrency** | sync 239 s / 75 · async 140 s / 83 | **1.85× / 3.22×** concurrency | 0.600 → 0.600 / 0.850 (5 tasks of 20; 1 seed, not a result) | round's diffs merged into one pool candidate (`--reflective-merge`); empty seed instruction; `--staleness guarded`, measured before the stale counters existed — [full setup](algo-gepa.md#measured-results-hotpotqa) |
| EvoSkill | FinQA (OfficeQA is HF-gated) | — | async N=4: 0.527 → 0.707 val over 120 rollouts | — | — | scheduling and merge timing; budget must be pinned. `--reflective-merge` offers the frontier one fused candidate per sweep instead of one per worker (`update_frontier` and the parent draw unchanged). The async arm used to swap in `SgdSkillAggregator` — no frontier, per-batch validation — which is now removed rather than optional |
| SkillOpt | SearchQA (`--hard` subset) | — | async N=4: 0.053 → 0.211 val over 60 rollouts | — | — | scheduling and merge timing; budget must be pinned. `--minibatch` is this port's name for the worker count, not upstream's minibatch of tasks. `--reflective-merge` scores one fused patch per step, which is *upstream's* shape (a ReflACT step emits one patch of up to `lr` edits) rather than a departure from it |
| ADAS | GPQA Diamond (MGSM is saturated) | — | **not measurable on this model** — see [algo-adas.md](algo-adas.md#measured-results-gpqa-diamond) | — | — | scheduling and merge timing; budget must be pinned. `--reflective-merge` is deliberately *not* passed: the archive is keep-all and is the meta-agent's whole conditioning signal, so fusing a round's designs would remove archive entries rather than change merge timing |
| DGM | vendored bugs w/ pytest (`--objective real`) | — | async N=2: seed 0.844, best archived child **0.906** over 16 rollouts | — | — | scheduling and merge timing; budget must be pinned. `--serial` sets `selfimprove_size=1`, which is a population of one, so this row's control is the degenerate archive rather than upstream's default |
| OpenEvolve | function minimization | — | — | — | — | **none** — `rounds = iterations // workers` already fixes total work, so this row's speedup is the only one that was equal-budget before the flag existed |

**The quality column is allowed to go down, and a table of all-green is probably
wrong.** Asynchrony has to cost something somewhere: stale diffs get discarded, a
rebased delta may no longer hold, and archive-style algorithms select against the
newest head. If not one of the seven shows it, the likeliest explanation is that
the measurement is too coarse or the serial arm was not really serial — not that
the cost is zero.

**Budget and variance follow the same rule as everywhere else.** Rollouts *and*
calls, both reported as measured; ≥ 3 seeds; the spread rather than a point
estimate. `docs/algo-ace.md` records one configuration moving 4.8 points between
two runs of itself.

The cells are empty because they have not been measured. Filling them in with a
one-seed run would be worse than leaving them.

Once they are, `python -m bench.matrix_report --json bench/results/matrix.json`
renders them — speedup off wall-clock net of time lost in failed calls, quality
as a median with its range, the async arm's stale rate with its denominator, and
a row daggered whenever it rests on fewer than three seeds. It withholds a
speedup outright where the cells show the arms did not spend the same budget,
which is the one check that cannot be done by reading the flags that were
passed.

## The eleven MethodPolicy ports

Their departures are not repeated here. Each one's page carries a **Boundaries**
section naming exactly what its compact or substituted domain gives up, and a
`!!! danger` block for every defect found in it and fixed — which is the same
*paper says / released code does / this port follows* audit, written where the
port is. Start from the
[table of all eleven](self-evolution-examples.md#the-eleven-microports-and-analogues);
their measured results are
[in one place](self-evolution-examples.md#measured-results-all-eighteen), and the
scheduler comparison they exist for is the [runtime matrix](matrix-overview.md).

What they share is the thing the fidelity class encodes: **a `mechanism_microport`
preserves the algorithm on a smaller domain, and an `environment_analogue`,
`inference_analogue` or `self_edit_analogue` substitutes something the paper does
not have** — a crafting world for Minecraft, verbal memory for a GRPO update, one
AST-gated function for a codebase. None of them is a paper-benchmark
reproduction, and none may be cited as one.
