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
    faithful benchmark port, and the [runtime matrix](matrix-report.md)
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
  the page documents because the measured lift depends on it (0.844 → 0.889 at
  `--top-k 120`; nothing to learn at 10).
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
* **Selection rule lives in**: `FrontierBest` — the frontier's best member as a named `SelectionPolicy`; the bounded top-K admission stays on `Frontier`
  `SgdSkillAggregator` (async) in
  `examples/evoskill/evoskill_skill_discovery.py` — the `topk_aggregate` mode of
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
  benchmark.
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
  execution is deterministic, budgeted, AST-gated and Bubblewrap-isolated
  (Linux-only — the offline suite skips only the sandbox test elsewhere).
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
| GEPA | HotpotQA | 1424 s / 97 calls | sync 1022 s / 70 · async 742 s / 95 | 1.39× / **1.92×** | 0.75 → 0.60 / 0.65 (1–3 tasks of 20; noise-range) | round's diffs merged into one pool candidate (`--reflective-merge`); empty seed instruction; 1 seed — [full setup](results.md#merging-as-a-cost-lever-serial-vs-8-wide-sync-and-async-gepahotpotqa) |
| EvoSkill | OfficeQA | — | — | — | — | scheduling and merge timing; budget must be pinned |
| SkillOpt | SearchQA | — | — | — | — | scheduling and merge timing; budget must be pinned. `--minibatch` is this port's name for the worker count, not upstream's minibatch of tasks |
| ADAS | MGSM | — | — | — | — | scheduling and merge timing; budget must be pinned |
| DGM | surrogate | — | — | — | — | scheduling and merge timing; budget must be pinned. `--serial` sets `selfimprove_size=1`, which is a population of one, so this row's control is the degenerate archive rather than upstream's default |
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

The eleven MethodPolicy ports *are* measured under all three schedulers — see the [runtime matrix](matrix-report.md).
