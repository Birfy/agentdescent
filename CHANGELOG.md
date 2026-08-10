# Changelog

All notable changes to AgentDescent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **`async_evolve` listed `selection` among the policies it honours and never
  asked it.** `selection` is in `_ASYNC_WIRED_POLICIES`, so
  `Policies.require_supported` let it through, and nothing in the barrier-free
  loop read `_pol.selection` — the field was accepted and dropped. The two
  drivers therefore disagreed about the same bundle: a policy naming a starting
  point other than the head raised `NotImplementedError` from `evolve()` and ran
  to completion under `async_evolve()`, returning a reward. That is precisely the
  outcome `require_supported` and `_check_selection` were each written to make
  impossible, arriving through the driver that called neither. The merger now
  asks once per sweep — a sweep being that loop's round, which is what `history`
  is indexed by.

### Added

- **`MultiHeadUnsupported`**, raised where `_check_selection` used to raise a
  bare `NotImplementedError`. It subclasses both `NotImplementedError` (what
  callers already catch) and `ContractError` (how it gets out). The second base
  is not cosmetic: the merger runs on a background thread that absorbs ordinary
  exceptions as backend failures and retries past them, so a plain
  `NotImplementedError` would have been filed as a transient, retried until the
  sweep budget ran out, and reported as though a provider had flaked.

## [0.4.2] — 2026-08-10

### Fixed

- **A run reported what it reached and nothing about how it was set up, so
  fifteen recorded rows attributed themselves to the wrong lag budget.** Every
  `bench/results/*.json` written from a MethodPolicy port pairs a `cells` list
  with a hand-typed `config` block; the cells were transcribed from the line
  `standard_main` prints, which states results only. Fifteen blocks recorded
  `async_ratio: 2` for runs that took `run_port`'s default of 1 -- 0.4.1 fixed
  the dropped flag but left the attribution, which named
  `bench.candidate_methods` as the source. It was not: that harness has no
  `--staleness` and never passes `staleness=` to `run_port`, so every run
  through it is `guarded`, and all fifteen blocks record `full`. The transcribed
  line is what produced them, and that is checkable rather than argued -- it
  formats qualities at `.3f`, seconds at `.1f` and calls as an int, and across
  all 45 cells not one value carries more precision, while the one file
  `bench.candidate_methods` did write has 198 of 198 wall/engine values at full
  float precision. Those files now record 1, with a note.
- **`run_port` now records `staleness` in `framework`.** It was the one setting
  a run could not state about itself, and the one that made those blocks
  impossible to attribute either way.

### Added

- **`examples._method_runner.run_config` and the `config:` line.** A live run
  prints its resolved configuration as one JSON object, in the key names those
  results blocks already use, so a block is copied out of a run instead of
  remembered about it. `tests/test_results_provenance.py` keeps the forensic
  invariant executable: a future results file whose cells carry more precision
  than that print line came from somewhere else, and its config block cannot be
  read as describing a command-line run.

## [0.4.1] — 2026-08-10

### Fixed

- **Four shared flags reached the MethodPolicy runner and were dropped.**
  `examples/_method_runner.py`'s `standard_main` declared `--async-ratio`,
  `--eval-concurrency` and `--eval-cache` through `add_standard_args` and passed
  none of them to `run_port`, so on all eleven declarative ports a run that set
  every one of them was byte-identical to a run that set none -- and
  `--val-cap`, which those ports cannot honour at all because their splits are
  frozen in `build()`, parsed and moved nothing. The same defect as Gödel
  Agent's `--gateless`, which five documents described while the parser rejected
  it. All three are now threaded and recorded in the run's own `framework`
  block, `--dry-run` prints them, and `--val-cap` is withdrawn rather than
  accepted. `--async-ratio` cost reproduction specifically: the rows in
  `bench/results/` were recorded at 2, and until now no
  `python -m examples.<port>` command could ask for that. It defaults to **1**
  here, not `add_standard_args`' 3, because `run_port`'s signature and
  `bench.candidate_methods` both say 1 and two entry points that disagree on a
  lag budget make the same nominal configuration run two different searches.
  `--eval-concurrency` defaults to unset for a related reason: taking the shared
  8 would make `--serial` -- the control arm, the upstream algorithm's own
  one-at-a-time loop -- score its gate eight ways at once.
  `tests/test_method_runner_flags.py` now enumerates the parser and fails on any
  flag with nowhere recorded that reads it, and every `Run it` command on the
  eleven algorithm pages passes `--async-ratio 2` outright.
- **A work-budget stop on the async path abandoned the evidence it had paid
  for.** `max_rollouts` counts a rollout when it completes, so when the budget
  trips, up to `n_workers - 1` rollouts are still in flight -- legitimately
  started, their model calls already billed. The merger drained the intake once
  and returned, dropping whatever landed after; and since a failing rollout runs
  propose + self-verify after solve (three sequential calls to a success's one),
  the abandoned set was enriched with exactly the rollouts that produce
  evidence. Measured on an 8-worker GEPA run: 8 cards produced, **7 abandoned**,
  the pool never grew past the seed, and the arm reported a wall-clock it had
  not earned. The merger now keeps draining until the workers exit whenever the
  stop was a work budget (bounded by `max_seconds`, the run's own outer limit);
  a time-budget stop keeps the short grace, because the caller bounded
  wall-clock and waiting would overshoot it.
- `evolve(policies=..., aggregator_factory=...)` silently ignored the merge-side
  policies: a custom factory replaces the optimizer, so `conflict`/`fusion`/
  `acceptance`/`promotion` never reached anything. It now warns, naming the
  dropped policies -- a caller who paid for a model-merging run deserves to know
  they got the factory's behaviour instead.

### Added

- **`Usage.failure_seconds`** -- model time spent inside calls that ultimately
  failed (retry waits, hung connections, timeouts), kept apart from `seconds`
  because `wall - failure_seconds` is the wall-clock net of endpoint weather:
  the number a comparison between two arms should quote when one was unlucky.
  Printed by `Usage.summary()` as `N failed (X.Xs lost)`.
- **`bench.matrix_run`** -- the parallelisation-matrix runner: one row per port,
  serial vs N-wide arms, `--budget-rollouts` pinned on both, results written
  after every cell (a sweep this long must survive interruption), per-cell
  transcripts kept because the parsed row answers "what did it cost", not "what
  did it do".
- **GEPA: `--reflective-merge` merges the round's diffs into one pool
  candidate.** Admission is what `ParetoAggregator` pays for -- every candidate
  is scored across all of D_pareto -- so a round from N workers costs N sweeps;
  merged, it costs one. Measured at equal budget (16 rollouts, seed 0, net of
  failures): serial 1424s/97 calls, sync N=8 1022s/70 calls (-28%), async N=8
  742s/95 calls, test EM within noise across arms. Pareto selection itself is
  untouched -- what changes is what the pool contains, recorded in
  `docs/port-fidelity.md`. The merger must be `ReflectiveFusion`: `fuse_diffs`
  on a one-key artifact keeps the last rewrite and drops the rest.
- GEPA's `_score_rows` batches D_pareto scoring across the round's candidates in
  one flattened pool (in-flight = `eval_concurrency`, not its square); admission
  order stays deterministic because dedup runs first and sequentially.
- Shared port flags: `--eval-concurrency` (held-out evaluations in flight),
  `--val-cap` (trim the gate split without touching test, via
  `examples._common.capped_val`), `--reflective-merge`, and GEPA's
  `--seed-instruction`.
- A static name-resolution guard over every port
  (`test_every_name_main_uses_is_actually_imported`): `--dry-run` returns before
  `main()`'s real body, so a missing import passes every test and then raises
  `NameError` twenty minutes into a paid sweep -- which is exactly how one
  shipped.


### Added

- **`--budget-rollouts` on every algorithm port**, forwarded as
  `evolve(max_rollouts=)` through `examples._common.budget_kwargs`. Without it the
  parallelisation matrix in `docs/port-fidelity.md` cannot be filled honestly: six
  of the seven ports pass a fixed iteration count and let `n_workers` multiply it,
  so an `N=8` arm runs **eight times** the rollouts of the `--serial` arm.
  Measured on the engine at `rounds=24` — 1 worker: 24 rollouts; 2: 48; 4: 96;
  8: **192**; with the budget, all four land on exactly 24. Comparing wall-clocks
  across that gap reports eight times the model spend as parallel efficiency, and
  comparing final quality credits the extra spend to parallelism — the confound
  `agentdescent.baselines` exists to remove. `evolve(max_rollouts=)` had shipped
  and **no port passed it**, the same shape of miss as `cheap_eval_tasks`.
  `tests/test_example_entrypoints.py::test_every_port_can_hold_its_rollout_budget_fixed`
  now refuses a port that cannot hold its budget fixed.

  OpenEvolve needed no fixing and is recorded as the exception: it derives
  `rounds = iterations // workers`, so its total work was already fixed and the
  shared flag simply sets `--iterations`. Its speedup row is therefore the only
  one in the matrix that was equal-budget before this existed — which is itself a
  "semantics changed" entry, since one row of a table meaning something different
  from the other six is the failure the column is there to prevent.


### Changed

- **The fusion tournament is off by default; the union goes straight to the
  acceptance gate.** Work out what the tournament decides that the gate does not
  and it is one case — the fusion beats the artifact but loses to one of the
  singles — so it is a *selection refinement*, not a safety mechanism: the gate
  already scores on the full held-out set and refuses a measured regression. Even
  that case is recoverable, because `fuse_diffs` is `ops.update()` and the union
  is a **superset** of every single, so committing it unranked loses no proposal.
  Against that, the ranking costs one cheap sweep per candidate every round,
  unconditionally. Measured on `tests/test_fusion_stats.py`'s multi-key fixture:
  **88 → 48 task evaluations, same `final_reward=1.000`**. On an `AppendRules`
  run the same change committed the round's four proposals as one merge instead
  of four, reaching the identical six-rule artifact in 2 commits where it took 6.
  `evolve(fusion_tournament=True)` (or `AggregatorConfig(fusion_tournament=True)`)
  restores ranking — and is the only way to get `FusionStats.win_rate`, since
  `best_single_score` exists only where a single was actually scored. That number
  is a property of the workload rather than of the mechanism, so it is worth
  measuring per workload and not worth paying for on every run.
- **`cheap_eval_tasks=None` now means 8**, or the whole held-out set when that is
  smaller — `ThreeLayerVerifier.rule_subset`'s own default, which `evolve()` had
  been overriding. It meant "score everything", which made the cheap layer cost
  exactly what the oracle costs: rule / learned / oracle were one full sweep
  wearing three names, ranking one candidate bought a full sweep of real agent
  calls, and `oracle_budget`'s documented fallback saved nothing because it was
  the same measurement. The knob to fix that shipped and **nothing in `bench/` or
  `examples/` ever passed it**, so every real run paid the full price. The cost is
  ranking resolution: 8 binary-scored tasks resolve 0.125, so candidates closer
  than that rank by whichever the sample favours. Both commit gates still read
  `eval_counts` on the full set. Pass `len(held_out)` for the old behaviour.

### Fixed

- **Identical proposals were counted as a fusion.** `fuse_diffs` is
  `ops.update()`, so N copies of one diff "fuse" into that diff — nothing is
  combined, and it went into `contested`, which is `win_rate`'s denominator.
  Measured on a mute-backend run: nine tournaments, every one of them two
  identical `{'value': 'rule-0'}` diffs, all nine counted. The same applied
  whenever the union merely equalled one of its inputs. `FusionStats` gained
  `nothing_to_fuse` to name it, counted apart from `contradiction` because the fix
  is the opposite one — workers duplicating each other, not a key space that is
  too coarse. This also made a `SingleSlot` run able to report a win rate for a
  mechanism `docs/results.md` says can never run there.
- `FusionStats.single_wins`, `neither` and `synthesized_wins` counted trials that
  ranked nothing, so a run where fusions and singles never met still reported
  singles beating them — `bench/results/equal-budget-hotpotqa-3seed.json` carries
  `single_wins: 3` from exactly that. All three are now guarded on `ranked`, like
  `fused_wins` already was. `unranked` counts by whether a union was *built*
  rather than by which policy built it, so it sees `DefaultFusion` too, and
  `summary()` no longer reports "fusion never ran" on runs that merged every
  round.
- `docs/results.md` presented "ACE playbook → `contested` > 0" under the heading
  "Measured on the two shipped artifacts". It comes from an offline unit test with
  a synthetic reward, not from a run on FiNER; the table now says so per row.

### Added

- `evolve(max_rollouts=, max_calls=)` — a budget the engine enforces, in the two
  units a comparison has to hold fixed. `rounds` is not one of them: configurations
  differ in how much model a round buys, so a budget fixed in rounds hands the wider
  one more model and then reports the extra model as a win for parallelism. The
  synchronous path checks at the round barrier and so overshoots by up to one round;
  the barrier-free path checks per rollout. `stop_reason` gained `"max_rollouts"` and
  `"max_calls"`, and `async_evolve` gained `max_calls=`.
- `agentdescent.baselines` — the control the efficiency numbers were missing.
  `serial` / `best_of_n_fork` / `merge_of_n` run over one shared `Workload` on one
  `Budget`, and fork reports both the oracle fork (best on test, unshippable) and the
  selected fork (best on dev, reported on test). `compare(fixed=...)` names the unit
  held fixed and prints the other unit's divergence as a confound — because the two
  cannot both be equalised, which is measured rather than assumed.
- `EvolutionResult.fusion_stats()` — the fusion tournament's record, with the
  denominators `RoundStat.fused` was missing. That counter tallied *committed*
  fusions, and the tournament only commits one that won, so it could not answer
  "does merging just average the improvements away?". The shipped `FusionPolicy`
  now records a `FusionTrial` per tournament (it was already computing the scores),
  reporting win rate, mean gain, the losing tail with its worst case, and fusions
  that fell below the baseline rather than merely below the best single diff.
  `win_rate` is `None` when nothing was contested, so a mechanism that never ran
  cannot be read as one that always lost.
- `--serial` on every algorithm port — the eighth shared flag, and the control
  they were all missing. Each port parallelises an algorithm published as a serial
  loop, and none of them could run that loop, so every parallelisation claim here
  was one-armed. `examples._common.worker_count` honours it (one worker, nothing
  to merge) and refuses `--serial --async`, which would be a one-worker
  *asynchronous* run — staleness in the control arm.

- `agentdescent.selection` — the decision `Policies` was missing: **which
  candidate the next batch of workers starts from**. `SingleHead` is the default
  and reproduces the current run exactly (asserted against a full run, not
  described); `Beam`, `ParetoFrontier(mode="per_instance"|"topk_aggregate")`,
  `Archive(sampling=...)` and `MCTS` are the ports' hand-written rules, as
  arguments. Selection is *under* merging, not instead of it: one selected
  starting point still has N/k workers merging diffs into it. A policy that names
  a starting point other than the head raises rather than being collapsed to it —
  the ledger holds one live branch, and multi-head support is separate work.

- `docs/port-fidelity.md` — one section per port, in the shape the differences
  actually take: paper says / released code does / this port follows. The notes
  existed, scattered across seven pages, a README paragraph and a test file, so
  "we did not change the algorithm" could not be checked in one place. Carries
  the parallelisation matrix (serial vs N=8, speedup, final held-out Δ, semantics
  changed) with its cells empty until measured, and states that a table with no
  quality regressions anywhere is more likely a measurement problem than a free
  lunch.

- `agentdescent.advantage` — three decision rules borrowed from PPO and GRPO,
  **all off by default and none validated**: group-relative advantage
  (`EvidenceCard.advantage`, consumed by opt-in `AdvantageAcceptance` /
  `AdvantageConflict`), `AdaptiveTrustRegion` for the two guessed diff-size
  constants, and `MergeContext.stable_distance` with `StableDistanceAcceptance`.
  There is no PPO and no GRPO here — no policy distribution, so no importance
  ratio and no clipping objective — and `docs/concepts.md` keeps these separate
  from the analogy table whose rows all land on code that runs. `bench.ab_run`
  is the A/B each has to pass; one that does not should be deleted with the
  negative result recorded.
- Measured while building it: a group is one base version and one task cluster,
  and the base version moves on every commit — so a group is at most one round of
  workers split across the clusters they landed in. Four workers over four
  clusters can never fill the default `min_group=4`. That bounds where the
  advantage signal can exist at all, and is pinned by a test rather than left to
  be discovered during an A/B.

- `docs/architecture.md` gains a diagram of the **two runtimes** — the barrier
  and the barrier-free path side by side — because everything the async path has
  to deal with follows from the barrier being gone, and the consequence that
  costs the most to rediscover (`η` is zero by construction on the synchronous
  path at `refresh_interval=1`, so the staleness policy cannot decide anything
  there) is a property of the runtime rather than of the algorithm. The data-flow
  diagram also shows the [selection seam](docs/selection.md), and names the
  difference between "which task" (`TaskScheduler`) and "which candidate"
  (`SelectionPolicy`), which the picture previously invited confusing.

- `agentdescent.fusion` — `ReflectiveFusion`, which merges **competing values for
  the same key** by asking a model to write one version keeping both. `fuse_diffs`
  is a dictionary update, so on a one-key artifact (GEPA's `InstructionSlot`) the
  last writer wins and no fused candidate is ever built: measured `contested = 0`
  for an entire run, which makes `merge_of_n` there per-round best-of-N selection
  rather than merging. Ships as a **pair** with `KeepContradictions` via
  `reflective_merge()`, because conflict resolution runs first and would otherwise
  hand the fusion policy a single diff. The synthesised candidate has no privilege
  in the tournament — a tie loses — and every failure path falls back to shipped
  behaviour. Off by default, not yet A/B'd.

- **The eleven issue #74 candidate methods now have an AgentDescent-native
  runtime study.** PromptBreeder, AFlow, Reflexion, Self-Refine, Voyager,
  SkillWeaver, Absolute Zero, R-Zero, Agent0, SICA, and Godel Agent reserve the
  same candidate and proposal-call budgets across
  `evolve(max_concurrency=1)`, synchronous `evolve(max_concurrency=workers)`,
  and `async_evolve`. The benchmark records paired end-to-end, engine-window,
  time-to-quality, provider, actor, staleness, and held-out quality metrics, and
  refuses to report a matrix with a changed source fingerprint or mismatched
  budget.

  The fidelity boundary is explicit: four are compact mechanism ports, while
  the environment, RL, and full-agent self-edit dependencies that do not fit the
  available host are labelled analogues rather than paper reproductions. Offline
  tests exercise every method in every runtime mode without an API key.

  The methods are now **declarative**: each lives in its own `examples/<name>/`
  folder as a `MethodPolicy` (artifact strategy, frozen datasets, pure
  solve/propose/reward, and the engine `Policies` seams its mechanism plugs
  into -- binary tournament and soft mixed selection, archive base selection,
  difficulty-weighted curricula, group-relative acceptance), with one shared
  runner owning phases, budgets, and modes. Validation happens once, in the
  strategy's `to_diff`: an unparseable proposal costs its candidate, is
  counted, and produces no diff -- no fallback substitution anywhere.
  Merge batches are worker-sized and text-valued artifacts install
  `reflective_merge`, so contradicting proposals are model-merged into one
  gate evaluation instead of each paying a ranking pass. Self-play evaluation
  splits are frozen per seed; the checked-in results predate this
  restructuring and are pending a rerun under the new fingerprint.

  Declared selection policies now actually run: `examples/_population.py`
  generalises the GEPA/DGM pattern into a `PopulationAggregator` -- the
  shipped merge pipeline plus an archive of committed heads, with any
  standard `SelectionPolicy` picking the next parent by ledger commit and
  `finalize()` landing the archive's best. The decision plane grew a
  documentation suite: a choosing guide (`docs/policies.md`), one page per
  policy kind listing every implementation, and an `aggregator_factory`
  page recording the single-head fact that makes the factory exit the home
  of population search.

### Changed

- OpenEvolve's `--serial` now means the shared thing (one worker) rather than
  `max_concurrency=1` with three. Its benchmark's `serial` mode is untouched and
  still isolates threading; `docs/algo-openevolve.md` states the difference.

## [0.4.0] — 2026-08-05

One engine. 0.3.0 made the numbers readable; this release removes the second
implementation they were being measured against.

`AgentDescent` and `AsyncAgentDescent` each had their own round barrier, worker
dispatch, snapshot staggering, merger thread, published head and backpressure.
`docs/architecture.md` called that "a known wart rather than a design intent",
and it kept costing: two measured fixes that had to be hand-ported between the
loops, two early-stop epsilons nobody chose, and three mechanisms the general
engine re-derived — and got wrong — because the reference stack already had
them. Both are adapters over `evolve()` now, the public surface is unchanged,
and the reference table still reads first 0.604, final 1.000, against a fork
baseline of 0.379.

Underneath that, the decisions a loop used to hard-code became replaceable
contracts, and the resources a rollout consumes got a plane of their own:
sandbox leases with an owner, a ceiling and a way to be reclaimed; a container
boundary verified against both podman and docker; a ledger and an evaluation
cache that survive more than one writing process; and an executor seam the
round's rollout actually goes through, which it did not before — a supplied
executor ran nothing and the run reported that it had gone fine.

**Breaking.** `agentdescent.worker.Worker`, `AsyncConfig.aggregator_interval`,
`AsyncConfig.worker_pause`, `EvidenceCard.version_annotations` and the
`domains.router.Task` alias are removed, and `full_eval` is no longer part of
the `Evolvable` protocol. Each entry below says what replaced it.
`Evolvable.cheap_eval` was renamed to `evidence_eval` but remains an alias, and
the three moved strategy import paths still work.

### Changed
- **Each faithful algorithm port now owns a directory.** `examples/` had grown to
  seven ports flat alongside six framework demos, and nothing in the name said
  which was which: `gepa_prompt_evolution.py` and `parallelism.py` sat at the
  same level, and OpenEvolve's two private helpers were `_openevolve_support.py`
  and `_openevolve_runner.py` — prefixed by hand precisely because a directory
  was not available to say it.

  So `examples/<algorithm>/` now holds the entry point, a `README.md`, and any
  helper only that port uses: `examples/ace/`, `examples/adas/`, `examples/dgm/`,
  `examples/evoskill/`, `examples/gepa/`, `examples/openevolve/`,
  `examples/skillopt/`. Module paths gain one segment
  (`python -m examples.gepa.gepa_prompt_evolution`), and the ~90 references
  across the README, docs, and tests moved with them. The framework demos
  (`run_demo`, `run_async`, `efficiency`, `parallelism`, `rq2_staleness`,
  `duration_scheduling`) stay at the top level, because they belong to no single
  algorithm.

  Each new `README.md` states the kind, governance layer, paper, upstream
  revision, dataset, and `evolve()` plug-ins, and links the port's `docs/algo-*`
  page and test file — the things a reader previously had to assemble from three
  places.

  One test needed more than a path rewrite.
  `test_ports_table_covers_every_standardised_entrypoint` globbed
  `examples/*.py`, so once the ports moved down a level it would have found
  nothing and passed by vacuum — the exact failure mode it exists to prevent. It
  walks the tree now and asserts the walk was non-empty, and a new
  `test_every_port_owns_a_directory_with_a_readme` holds the layout itself,
  since a port dropped back at the top level would otherwise still import, still
  run, and go unnoticed.

### Fixed
- **The OpenEvolve port joined the examples tree without joining the entrypoint
  contract, and `main` went red on the guard test written for exactly that.**
  `test_ports_table_covers_every_standardised_entrypoint` globs the examples
  directory for anything calling `add_standard_args` and demands it appear in
  `PORTS`; the port called the helper and never appeared, so the seventh entry
  was missing and the assertion failed on every push.

  The table was the smaller half of it. The port had also grown its own copies
  of two things `examples/_common.py` already owns: a `_make_completion` that
  branched on `provider in ("openai", "glm")` itself, and an inline
  `input("Proceed with paid model calls? [y/N] ")`. Both are what
  `test_no_port_reimplements_the_shared_behaviour` exists to catch — it just
  never ran on this module, because the module was not in the table. Adding the
  row without the cleanup would have traded a red gate for a silent exemption.

  `_make_completion` now dispatches through `completion_for()` and assembles
  only the genuinely one-sided option (GLM `thinking`), the prompt is
  `confirm(args)`, and `PORTS` became a `NamedTuple` whose `provider` and
  `async_ratio` carry the shared defaults. That last part is what keeps the
  entry honest: OpenEvolve really is measured on an OpenAI-compatible GLM
  endpoint with a lag budget of 1, so the deviation is now written down in the
  row rather than expressed by staying out of the table.

### Added
- **A test that runs the container execution chain end to end**, and the first
  time `sandbox_container.py` has been exercised against a real engine at all.
  Ten of its tests carry `skipif engine_available(...)`, and every machine that
  has run this suite skipped all ten -- the provider shipped, and stayed shipped,
  without an engine ever seeing it.

  What that hid is not in any one component. It is that `runners._sh` has to
  notice `sandbox.exec_prefix()`, that the workspace has to appear at
  `CONTAINER_WORKDIR`, and that a missing tool has to come back as a *scored*
  failure rather than an exception. The new test materialises a candidate, runs
  its frozen test suite inside the container, runs the entrypoint there, and
  asserts a broken gate returns `TEST_FAILURE_MARKER` in-band.

  It puts the workspace under `$HOME` rather than `$TMPDIR`, which is the fix the
  provider's own error message gives: on macOS and Windows the engine runs in a
  VM that shares only part of the host filesystem, and the system temp directory
  is usually outside it. Reaching that message was the first thing a live run
  found.

### Documentation
- **The algorithm-port results table now says which rows survive a change of
  model, because two of them do not.** Every row was measured with
  `deepseek-v4-flash` -- stated in the page's opening line and nowhere in the
  table, so a table copied out of context carried none of it.

  Re-running all five ports against `glm-5.2` (real API, original datasets, the
  published settings) reproduced three and flattened two:

      DGM        0.000 -> 0.300, test 0.200   identical to the digit
      GEPA       Pareto 0.500 -> 0.600        reproduced; test 0.800
      EvoSkill   val 0.500 -> 0.577           against a published 0.487 -> 0.573
      ACE        val 0.867 -> 0.867           against a published 0.844 -> 0.889
      SkillOpt   val 1.000 -> 1.000           0 edits accepted, 6 rounds

  The mechanism ran correctly in all five -- ACE spent 413 calls against a
  published 403, so it did the same work. What differs is that the two that
  flattened take their difficulty from a **knob calibrated against the published
  model**: `--hard` keeps the items the seed answers wrong, and `glm-5.2` answers
  95% of SearchQA correctly, so `select_hard` found 2 hard items in 40 and padded
  to its floor with items the model already solves. `--top-k 120` sets how many
  XBRL concepts compete, and `glm-5.2` starts above where the published run
  finished.

  The three that reproduce are the three whose difficulty is model-independent: a
  deterministic surrogate, multi-hop retrieval, and a decimal-place *convention*
  no capability can guess. The table marks the knob-dependent rows with ⚠︎ and
  the published numbers are unchanged -- they were measured, and nothing here
  falsifies them for the model they were measured on.

### Added
- **OpenEvolve function-minimization now runs through AgentDescent's real
  evolution engines.** The port maps generated Python programs to a `Strategy`,
  the MAP-Elites island archive to an `Aggregator`, sandboxed evaluation to the
  rollout/reward boundary, and supports both `evolve()` and `async_evolve()`.
  Its offline tests cover archive migration, strict mutation budgets, all three
  runtime modes, and Bubblewrap is skipped when unavailable. A compact live
  GLM-5.2 benchmark records three paired repetitions without model responses or
  generated source.

### Fixed
- **`claude()` was the one blocking boundary in the package with no timeout.**
  `_git` bounds a command at 120s, `_CliAgent` at 600s, `runners._sh` takes one
  per call, and `openai_compatible` has had `timeout=120.0` all along -- this one
  relied on the Anthropic SDK's 600s default, which the SDK then retries
  internally, and which `with_retries` retries again. One logical call against a
  stalled endpoint can block for well over half an hour while the log says
  nothing, and a run doing that is indistinguishable from a slow one.

  Measured against a hosted endpoint: a GEPA run sat **51 minutes without
  finishing five rounds** -- 1.07s of CPU across the whole time and one
  ESTABLISHED socket -- and the same run with `timeout=120.0` finished all five
  in **14 minutes**, 96 calls. Same model, same endpoint, same settings.

  `claude(timeout=120.0)` now matches its sibling adapter and is overridable for
  a backend that legitimately takes longer.

### Fixed
- **`evolve(staleness_policy=...)` could not decide anything on the synchronous
  path.** The loop snapshots the ledger at the top of every round and every
  worker proposes against that snapshot, so a diff's staleness `eta` is **0 by
  construction** -- measured over an 8-round run, all 15 staleness decisions saw
  `eta = 0` and returned ACCEPT, which makes Full, Guarded and Reflective
  identical runs. The `alpha` tolerances in `AggregatorConfig` and the
  `all-stale` outcome were equally unreachable there. The whole mechanism only
  ever bit on `async_evolve`, where `async_ratio` produces the drift.

  `evolve(refresh_interval=N)` is the missing half: a worker keeps its snapshot
  for N rounds, **staggered by worker id**, so the workers hold a spread of
  versions and their diffs arrive with a spread of `eta`. `1` is the default and
  is exactly the old behaviour. It costs no extra ledger read -- a worker either
  adopts the snapshot the round already took, or keeps the older one it has --
  and it is the same mechanism `AgentDescent` uses to make the staleness sweep
  meaningful, which is the first thing the general engine was missing before the
  two loops can be merged.

- **The async path counted every surviving card twice, so `stale_rate()` read
  about half the truth.** `async_evolve` runs its own staleness gate — it has to,
  because a custom `aggregator_factory` is promised only already-rebased cards —
  and then `Aggregator` runs its own over the survivors, on the same `Meter`.
  Measured: 20 cards reported `stale_considered = 40`. A true 50% stale rate read
  as 33%, and the end-of-run "you are discarding most of your evidence" warning,
  which fires at `discarded/considered > 0.5`, needed a *67%* true rate to trip.
  Each side now counts only what the other cannot see: the gate's discards, the
  aggregator's survivors.

- **An async worker read the ledger once per rollout to measure its drift.** A
  ledger read is a `git checkout` behind a process-wide file lock and an RLock
  every worker and the merger queue on, so the cost of asking "am I far enough
  behind to resync?" grew with the concurrency it exists to support. The merger
  is the only writer, so it now publishes the head after each sweep and workers
  read that — which is what `AsyncAgentDescent` has always done, and what the
  general engine reached for git to do instead. Measured on a 21-rollout run: 46
  ledger reads before, 22 after. A published head can lag by one sweep, delaying
  a refresh by at most one rollout; the refresh still takes a real snapshot.

- **An exhausted `oracle_budget` turned the audit gate into a sub-sample veto.**
  `oracle_eval` degrades to `rule_eval` once the budget is gone — by design, so a
  run cannot spend money it was told not to — but `rule_eval` is the
  `cheap_eval_tasks` sub-sample, and the audit gate vetoed on its verdict.
  Measured: a candidate that took the full held-out rate from 0.5 to 1.0 came
  back `oracle-rejected` because a two-task sample scored both sides at 0.5. Two
  sections of `docs/verifier.md` promise sub-sampling can never decide a commit;
  this was the path that made it false, and it is the same hole the acceptance
  step's regression guard was fixed for one entry below.

  The fix is that the audit stops buying a measurement it already has.
  `ThreeLayerVerifier.oracle_shares_full_set` records what that class has always
  done — `oracle_eval` and `eval_counts` are the same `eval_fn` over the same
  held-out set — and the aggregator reuses the full-set rates the acceptance test
  just computed. The verdict is identical while budget remains, and there is no
  degraded path to fall onto when it does not. A custom verifier whose oracle is
  genuinely independent leaves the attribute undefined and keeps being called.

  Consequences worth knowing: with the shipped verifier an L1 run now reports
  `oracle_calls_used == 0` — the audit ran, nothing had to be bought — so
  `AuditScheduler.audits` was added to answer "did the gate open", which is the
  question `oracle_calls_used` used to be a proxy for.

- **`async_evolve()` accepted `Policies(executor=...)` and ignored it.** Both
  engines shared one `_WIRED_POLICIES` tuple, and the barrier-free loop has no
  executor seam — its worker calls `eng.run` directly. So the field was declared
  supported for a loop that never read it: accepted, then dropped, which is the
  single outcome `Policies.require_supported` exists to prevent. It got sharper
  the moment a supplied executor started working under `evolve()`, because
  flipping `asynchronous=True` would then silently stop honouring it. The async
  path now has its own set and raises `NotImplementedError` naming the field.

- **A "cheap re-verification" that verified nothing did so in silence.** The
  staleness policies' `REBASE` branch keeps a rebased diff iff
  `before <= after`, and `EvolvingArtifact.score` returns `0.0` for an empty task
  list — so when an evidence card carries nothing the engine can score, the check
  compares `0.0 <= 0.0` and keeps *every* diff, including one that makes the
  artifact worse. `EvidenceCard.trajectory_refs` was annotated `List[str]` for a
  while, so a domain that followed the annotation and stored ids landed here with
  a non-empty list and nothing in it to score. The annotation was corrected
  earlier; `evidence_eval` now warns once per run when it happens, which is what
  would have made the original trap visible while it was running.

- **`policies=Policies(executor=...)` produced a finished run that measured
  nothing.** `evolve()` describes each rollout as a `RolloutSpec`, and the
  `Ref`s in it named `agents:echo` / `rewards:contains` as stand-ins — the
  caller's `run` and `reward` are closures and cannot be named. The built-in
  `ThreadExecutor` holds the actors directly and never resolved them, so nothing
  noticed; a *supplied* executor resolved the stand-ins, failed every rollout on
  an argument-count mismatch, and returned `rollouts=0` with a plausible
  `final_reward` from the gate and no exception raised.

  A supplied executor is now handed the run's actors via `attach_actors(run,
  reward)` — `evolve()`'s win over any passed to the constructor, so which actor
  runs does not depend on how the executor was built. One that cannot accept
  them (any cross-process executor: a closure does not cross a boundary) is
  **refused at build time** naming the fix, rather than producing a wrong answer
  in the shape of a right one. The spec's placeholder `Ref`s now point at
  `evolution:undescribable_actor`, which raises and explains itself if anything
  else resolves them.

- **`cheap_eval_tasks` could veto a commit, while four places promised it could
  not.** The aggregator's acceptance step refuses a candidate that scores worse
  than the incumbent — a guard worth having — but it read the *cheap* layer,
  which `cheap_eval_tasks` sub-samples. So a four-task sample could overturn a
  decision the full held-out Beta test had just approved, and the three
  `evolve()` entry points for directories default that knob to 4. The guard now
  reads the full-set rates `eval_counts` has already computed, which is what it
  meant all along; two source comments, one docstring and two doc pages that
  asserted the opposite are corrected.

  Its rejection message is also honest now. Both gates reported
  `P(delta>0)=… <= …`, so a regression rejection printed something like
  `P(delta>0)=0.97 <= 0.50` — self-contradictory to anyone reading `outcomes()`
  to find out why nothing committed, which is that method's only job.

- **`EvolvingArtifact.diff()` could not express a deletion**, so
  `a.apply(a.diff(b))` differed from `b` whenever `b` had dropped a key —
  silently, and for exactly the case a file tree cares about, where a key is a
  path. `apply` learned the `None` sentinel last release; `diff` now emits it.

- **The contract mechanism was declared and never enforced.** `Contract`,
  `Contract.is_compatible_with` and `ContractRejected` all existed, nothing
  called any of them, and the docstrings described the enforcement as if it were
  there. `Ledger.register` now records an artifact's contract and `commit`
  refuses a state whose major disagrees.

- **`domains.router.Task`** — the alias colliding with `evolution.Task` — is no
  longer used inside the package (`worker.py` takes `RouterTask`). The alias
  stays, marked deprecated.

- **`stable` never received the artifact a run produced.** Both reference
  runtimes call `Aggregator.finalize()` on a clean exit, and `docs/ledger.md`
  documents it — but `evolve()` and `async_evolve()`, the two engines a real
  workload actually reaches, did not. Promotion needs `promote_after_k`
  regression-free rounds and `target_reward` stops the run on the very commit
  that reaches it, so a clean run that hit its target left `stable` holding the
  *seed* artifact while `dev` held the one the run was for. Both engines now
  publish their head on a clean exit (and only on a clean one: a run that died
  leaves the confirmed branch where it was).

- **`result.forced_refreshes` was non-zero on every healthy async run.** The
  field, `docs/async.md`, `docs/staleness.md` and `concepts.md` all say a
  non-zero count means the lag budget and the staleness tolerance disagree — but
  it counted *every* snapshot refresh, including the ordinary lag-budget one that
  happens all run long. A converged three-second run reported 3. Only
  backpressure-forced resyncs are counted now, in both barrier-free runtimes, so
  the diagnostic is quiet when there is nothing to diagnose.

- **A stratified split could put the same item in `val` and `test`.** For a class
  too small for the ratios the cut points collapse, and `g[a:b] or g[a:a+1]`
  *copied* an item into `val` while `test` still took it — so the split
  `Dataset` documents as "fully held out, never seen by the optimizer" contained
  an item the acceptance gate had already scored. Rare classes are exactly what
  stratifying is for (a thin FiNER tag, a low-resource MGSM language), and three
  shipped ports stratify. The item is now *moved*, not copied.

- **`KeyedRules.keys()` returned a key space the strategy never writes.**
  Matching is case-insensitive and `to_diff` stores the folded key, so
  `categories=["Routing"]` under tensor parallelism built a section map on
  `"Routing"` while every diff wrote `"routing"`: no key had an owner and every
  proposal in the run was rejected as `section-violation`.

- **`document_agent` and `openhands()` leaked a workspace per call.** Each staged
  a fresh temp directory — with a copy of the document, routinely a megabyte of
  financial tables — and never removed it, while every other staging path in the
  package cleans up after itself. Both now do, and `openhands()` still leaves a
  caller-supplied `in_workspace()` directory alone, since that one holds the
  caller's files.

- **`EvidenceCard.trajectory_refs` was annotated `List[str]`, which quietly
  disables the staleness gate.** Every producer in the package puts task
  *objects* there and both consumers `isinstance`-filter for them, so a custom
  domain that follows the annotation and stores ids gets an `evidence_eval` over
  an empty list — the REBASE branch then compares `0.0 <= 0.0`, keeps every
  rebased diff, and a diff that makes the artifact *worse* survives the
  re-verification meant to catch it. Annotated `List[Any]`, documented, and
  pinned by a test that a harmful rebased diff is discarded.

- **`result.outcomes()` did not print the way any of its documentation shows.**
  It is a dict, and printing a dict calls `repr` on its keys, so the `MergeOutcome`
  enum rendered `{<MergeOutcome.COMMITTED: 'committed'>: 1}` where the README,
  both quickstarts, `evolution.md` and the enum's own docstring all show
  `{'committed': 1}`. `__repr__` is now the value's, as `enum.StrEnum` does on
  3.11+; equality, `str()`, f-strings and lookups by bare string are unchanged.

- **Three algorithm ports could crash in their final report, and three discarded
  `error`.** GEPA indexed `agg.scores[0]` (an `IndexError` when no round
  completed) behind a guard that would itself have raised `TypeError`
  (`f"{[]:.3f}"`), and ACE and `skill_evolution` indexed `history[0]` — all on
  the exact partial result the engine goes to lengths to return. GEPA also
  dropped `evolve()`'s return value entirely, and `EvoResult` / `SkillOptResult`
  carried no `error` or `stop_reason`, so a run killed by a rate limit printed
  the same confident "seed → best" line as one that converged. Every port now
  reports both. ACE and `skill_evolution` also label that first number "round 0"
  rather than implying it is the seed's score — it is measured *after* round 0's
  merge.

### Changed
- **The documentation grew an execution-and-resource plane, because the code
  had one and the pages did not.** Eight modules — `executor`, `supervisor`,
  `workspec`, `sandbox`, `sandbox_shared`, `sandbox_container` and the two
  evaluation ones — were reachable only through pages named for something else:
  where a rollout runs was a section inside *Customizable parallelism*, and how a
  sandbox is isolated was a section inside *Evolving a directory*. A reader
  running `evolve()` against a container had no reason to open either.

  There are now two pages, [Where rollouts run](docs/execution.md) and
  [Sandboxes](docs/sandboxes.md), grouped with async and scheduling under a
  *Running at scale* section; the pages they came from keep a pointer. The module
  map is redrawn along the same four planes — its ASCII diagram had listed
  `sandbox` twice and filed six execution modules under *how a change is
  accepted*.

  Also corrected against the code, rather than moved: the executors' status
  ("not yet wired into `evolve()`'s round loop" — they were wired in #89), a
  paragraph in the container warning that had escaped its admonition mid-sentence
  since #67, and three docstrings that described behaviour the code does not have
  (`Ledger._exclusive` calling its `RLock` a `Lock` and claiming the section
  nests — `flock` is per-open-file-description, so it does not;
  `EvaluatorGroup.map` claiming to report rather than raise; `bench.harness`
  claiming every row carries a fingerprint when `bench.run` populates none).

  Two limits that were true and unwritten are now written down: the shared
  sandbox pool's ceiling is [advisory](docs/sandboxes.md#one-ceiling-across-processes)
  — admission reads the lease directory and then acquires, with nothing holding it
  still in between — and `Ledger` [initialises its repository before the lock
  exists](docs/ledger.md#more-than-one-writer), so concurrent *creation* of one
  path still races.

- **Faithful algorithm ports now share one tested CLI contract.** Their provider,
  model, seed, async, dry-run and confirmation flags come from
  `examples._common`; upstream iteration vocabulary and per-port defaults remain
  local. A copyable code/test template and a short porting checklist make the
  same contract the starting point for future ports. The behaviour behind the
  flags is shared too, not just their declaration: `confirm` reads `--yes` and
  `completion_for` dispatches `--provider`, so a port cannot declare a flag it
  never honours — which is the drift that lost DGM its `--yes`. All six
  `--dry-run` paths now return before dataset/model setup, so they are genuinely
  zero-network on a cold machine, and the candidate backlog records mechanism
  gaps and owners.
- **The two barrier-free runtimes share their policies.** `async_evolve` and
  `AsyncAgentDescent` implemented the same shape independently; a previous
  release had to hand-port two measured fixes between them. The parts that are
  *policy* rather than plumbing now live in `agentdescent.pipeline`
  (`WorkerHealth`, `StallGuard`) and both call them, so the next fix lands in
  both. The runtimes themselves are still separate — unifying them would move
  every measured result's reproduction path, which is a decision, not a cleanup.

  `StallGuard` was the half of that sentence which was not yet true: both
  runtimes still counted their no-commit sweeps inline, so the module documented
  a convergence it had not reached. Both now count through it, on identical
  conditions to before — each still decides *which* sweeps count, because a sweep
  with no evidence in it is neither progress nor a stall and the two runtimes
  express "no evidence" differently.
- **One difficulty-weight formula.** `4·p·(1−p)` was written out in both
  `sampling.DifficultyWeighted` (over tasks) and `scheduler.TaskScheduler` (over
  clusters); it now lives in `stats.difficulty_weight`. The exploration constants
  still differ (0.2 vs 1.4) and the docstring says why: the sweep that produced
  0.2 was measured at task granularity and has not been repeated at cluster
  granularity.
- **One definition of which section owns a key.** `TensorParallelMerge` used the
  `section_of` hash bucket while `TensorParallel` and `evolve()` used the
  `assign_key_sections` partition — two answers to the same question, so a diff
  legal on one path could be rejected on the other. `TensorParallelMerge(keys=…)`
  now uses the same partition.
- **`AuditScheduler` queues only when asked** (`collect=True`). Nothing in the
  shipped runtimes calls `pop()`, so every `submit` was a heap push plus a
  periodic rebuild — once per merge decision — for a queue no one reads. The
  priority is still computed and returned; `force_oracle` and the trust update
  are unaffected.
- **`Evolvable` requires one method fewer.** `full_eval` was in the protocol and
  called by nothing: ground truth reaches the aggregator through the verifier's
  `eval_fn`, which the domain supplies. Implementations keep theirs.
- **`Evolvable.cheap_eval` is now `evidence_eval`**, because it collided with
  `ThreeLayerVerifier.cheap_eval` — different argument, different meaning, both
  called from the same forty lines of the aggregator. `cheap_eval` remains an
  alias.
- **The text strategies moved out of the engine** into `agentdescent.strategies`,
  so the rule is now "one module per strategy family, none of them in
  `evolution.py`" — `FileTree` already had its own. All three import paths still
  work.
- `parallel.py` no longer imports `RouterSkill`: a general primitive was typed to
  the reference domain.

### Documentation
- A second pass, this time reading the docs *against* the code they describe:
  - `docs/aggregator.md`'s stage list was mangled — steps 4 and 5 appeared twice
    and the commit came *before* the audit gate, which is the one thing
    `aggregator.py`'s own docstring insists on ("it holds a veto"). The README's
    copy had the same inversion, and additionally said promotion counts
    *commits* — it counts regression-free **rounds**, and a commit restarts the
    clock.
  - `concepts.md` said `duration_estimator=` is reachable from `evolve()`; it is
    an `async_evolve` parameter and `evolve()` raises `TypeError` on it. It also
    described the audit priority queue as always built, when `collect=False` (the
    default) computes the priority without queuing anything.
  - `docs/async.md` described `stall_patience` as counting *empty* merger sweeps
    — it counts sweeps that had cards and committed nothing, the opposite
    condition. `docs/usage.md` described `AsyncConfig.noise` as a fraction of
    workers; it is the per-op probability that one of the (every third) noisy
    workers proposes a wrong label.
  - `docs/evolution.md`'s "why did nothing commit?" table was missing `oversized`
    and `section-violation`, two of the eight categories `outcomes()` can return
    — and `oversized` points at the opposite fix from the `all-stale` it used to
    be folded into.
  - `docs/modules.md` claims to list every module and omitted `strategies`,
    `stats` and `pipeline`, still filing the text strategies under `evolution`
    after they moved out.
  - The `run_async` policy table in the README and `concepts.md` now carries the
    same caveat `efficiency.md` already gave its own numbers: the ratios are the
    result, the absolute counts scale with the machine.
  - `agentdescent/__init__.py` pointed readers at `agentdescent_design.md`, which
    is not in the repository.
  - `docs/verifier.md` advertised a three-method interface for a custom verifier
    (`cheap_eval` / `eval_counts` / `oracle_eval`); the aggregator also calls
    `learned_eval`, so building to the page raised `AttributeError` from inside a
    merge, after the run had spent its rollouts.
  - `docs/duration-scheduling.md` said an overrunning rollout is "set aside and
    resumed" and does not "block a worker" — contradicting its own warning three
    paragraphs later, which is the accurate one: the record is written *after*
    the rollout returns, and nothing resumes it. Its header also pointed only at
    `AsyncAgentDescent`, though `async_evolve` takes `duration_estimator=` too.
  - `docs/governance.md` says every algorithm port prints its governance layer at
    startup; ACE and GEPA did not, so they now do.
- A consistency pass over the new site found 14 problems and fixed them. The ones
  worth naming: `staleness.md` had the `alpha` tolerance **backwards** (it widens
  for L1/hot artifacts, not cold) and contradicted `concepts.md`; `sampling.md`
  had dropped the caveat `evolution.md` carried, that the difficulty-weighted
  numbers are a *targeting* measurement and not an accuracy claim; `evolution.md`
  kept full copies of sections that now have their own pages while linking away
  to them; and the `AggregatorConfig` reference omitted `trust_region_chars`,
  which is the per-file cap a `FileTree` caller has to know about.

### Documentation
- **The documentation site was rebuilt as a reference, not a tour.** It had 23
  pages covering 7 of 21 modules and no API reference at all: a reader who wanted
  `Ledger`, `ThreeLayerVerifier`, `StalenessPolicy`, `DifficultyWeighted` or the
  scorers had nowhere to go but the source. Now 38 pages on a
  concepts → quickstarts → module reference → API structure, with a page per
  module.

  - **New: [`docs/api.md`](docs/api.md), generated.** Every public name with its
    real signature, produced by `python -m tools.gen_api_docs` from the package's
    own signatures and docstrings. A hand-written reference is wrong the moment a
    signature changes and nobody notices until a reader copies a call that no
    longer exists — so `tests/test_api_reference.py` runs the generator's
    `--check` mode and fails when the page and the code disagree. It also proves
    every exported name appears, which is how the `evolve_skill` gap below was
    found.
  - **New module pages** for the parts that had none: the data model, ledger,
    verifier, governance, staleness, strategies, sampling, rewards, async,
    backends, and the reference orchestrator/domain.
  - **New entry pages**: install-and-first-run, a directory quickstart, and a
    module map with a reading order per goal.
  - `duration-scheduling.md` gained the **audit scheduler**, which was
    undocumented despite being half of `scheduler.py`.

### Fixed
- **`evolve_skill` was importable but missing from `agentdescent.__all__`.**
  `from agentdescent import evolve_skill` worked — it is what the README and the
  quickstart use — but `import *` and any tooling reading `__all__` did not see
  it. Found by the generated API reference, which asserts that every exported
  name is documented and had nothing to document.

### Added
- **`ClusterParallel`, and the feedback channel that makes it possible.**
  `ParallelStrategy.plan(n_workers, round_index, keys)` was a pure function of
  its arguments, so a strategy could not learn anything from the rollouts it
  dispatched -- enough to *shard*, not enough to *schedule*. That is why UCB over
  task clusters (design section 5.2, L-task) existed only in the reference
  runtime's `TaskScheduler`, which `evolve()` cannot reach, and why the general
  engine's answer to the task tail stopped at `DifficultyWeighted` over
  individual tasks.

  `ParallelStrategy` gains an **optional** `observe(unit, task_id, score)`;
  `evolve()` calls it after every rollout when the strategy defines one, so
  `DataParallel` and `TensorParallel` are untouched. `ClusterParallel` is the
  consumer -- it groups tasks by `cluster_of(task_id)`, leases whole clusters
  UCB-ordered with the same difficulty filter (all-pass and all-fail clusters
  carry no gradient), and feeds each rollout's reward back into the estimate. It
  composes with `task_sampler` rather than replacing it: one picks the cluster,
  the other picks the task inside it, and they share
  `stats.difficulty_weight`.

  Two stated differences from the reference scheduler: it learns from a
  rollout's **reward** rather than the before/after delta of the diff it produced
  (that delta needs `self_verify`, which the directory entry points turn off, so
  a scheduler depending on it would silently stop learning exactly there), and it
  learns per task rather than per lease.

- **`RoundInfo` reports what the merge *did*, not only what category it landed
  in.** Four numbers `MergeReport` computed all along and the driver threw away:
  `considered` (the denominator), `discarded_stale`, `conflicts_dropped`, and
  `fused` -- commits whose winning candidate was the fusion of several diffs
  rather than any single one, which is the model-soup question asked per round.
  The reference runtimes reported them (`RoundStat.fused`,
  `AsyncStats.conflicts_dropped`) and the engine every real workload uses did
  not, so a caller could see *that* nothing committed and never *how* the merge
  got there. `save`/`load` round-trip them, and a file written before they
  existed still loads.

  `fused` counts only fusions that **committed**: the tournament builds a fused
  candidate whenever the survivors are complementary, so counting the ones it
  built says nothing about whether combining them beat taking the best single
  diff.

  `_Engine.record_round` now takes the reports rather than a pre-chewed
  `committed` / `rejected` / `reasons`. Deriving those was the other thing both
  loops were doing separately -- and they disagreed on spelling, one counting
  `rejected` as `len(reports) - committed` and the other re-scanning for a
  missing `committed_version`. Same answer, two places for the next number to be
  added to one and not the other.

- **Evolving a directory: a skill folder, an agent folder, or its code.** Until
  now every artifact was text that ended up *in a prompt*. A skill directory is
  not that: it is only a skill directory if the agent can *read the files*, which
  needs the candidate on disk, in the layout the agent expects, for the duration
  of one rollout. Four new modules do that translation and nothing in the
  optimizer changed — `aggregator.py`, `ledger.py`, `async_evolve.py`,
  `parallel.py` and `governance.py` are untouched.

  The reason it drops in so cleanly is that the engine never interprets a state
  key: it only asks whether two diffs touch the same one. Make the keys **file
  paths** and file-level semantics arrive for free — two workers editing
  different files fuse, two editing the same file contradict and are resolved on
  held-out score.

  - `agentdescent.filetree` — `load_tree` / `materialize` / `canonical` /
    `parse_tree` / `TreeSpec`. A file that matches `include` but cannot be
    represented (binary, oversized) **raises** rather than being skipped: a
    silently dropped file is one `write_to` would later delete by omission.
    Paths are re-validated at every boundary, because a state key is a
    model-authored string and `materialize` turns it into a filesystem write.
  - `agentdescent.treestrategy` — the `FileTree` strategy, the `<EDITS>`
    multi-file proposal protocol (`parse_edits`), and `tree_reflector`, which
    shows the reflector the files it may change rather than the whole tree.
  - `agentdescent.runners` — `tree_runner` / `code_runner`. One throwaway
    workspace **per call**: `max_concurrency` worker threads and
    `eval_concurrency` evaluation threads share a `run`, so a fixed directory
    would let two candidates overwrite each other and produce scores that look
    ordinary and are wrong.
  - `agentdescent.skilldir` — `evolve_skill_dir` (L2), `evolve_agent_dir` (L1),
    `evolve_agent_code` (L1 + a test gate).
  - `EvolutionResult.write_to(path)` installs the evolved tree back, backing the
    directory up first, leaving unknown files alone unless `prune=True`, and
    refusing outright when the artifact is not a file tree.
  - `examples/skill_dir_evolution.py` — runs offline in seconds; the "agent" is a
    real subprocess bound to a workspace, so the staging, layout, overlay and
    isolation logic are all exercised without an API key.

- **`FileTree(frozen=[...])`, enforced twice.** `governance.py` freezes whole
  artifacts by id, which cannot say "this skill may evolve, but not its test
  suite" — and without that the shortest path to a high score is to weaken the
  thing measuring it. Filtering proposals only stops the *reflector*; candidate
  code can still rewrite `conftest.py` at run time. So the runner also **overlays
  the pristine frozen files after materialisation**, and `code_runner` invokes
  the gate from outside the tree. Both halves are needed; only the second is a
  security boundary.

- **`document_agent(..., skill_files=...)` — skills as files, not prompt text.**
  A skill *directory* is only worth more than a concatenated string if the agent
  can open one skill at a time. When the completion is a `WorkspaceAgent` the
  library is written to `.claude/skills/` in the same scratch directory as the
  document and the prompt carries a pointer; a backend with no workspace folds
  them back into the inline block rather than dropping them in silence.

### Changed
- **EvoSkill's skill library is now a directory** (`SkillLibraryTree`, a
  `FileTree` subclass): `{"defense-lookup": ...}` became
  `{"skills/defense-lookup/SKILL.md": ...}`. This was the acceptance test for the
  new abstraction — if the port could not be expressed in it, the shape was
  wrong — and it passed with all 12 existing EvoSkill tests unchanged.

  Two things it settled, both of which looked like blockers beforehand:

  * **A port keeps its own prompt format.** `render()` has to be the lossless
    serialisation because it is the evaluation-cache key, but `run` is where an
    artifact becomes a prompt — so `run` parses the tree and re-renders it in
    EvoSkill's own `### skill: <name>` format. The retriever path's prompt is
    byte-identical to before, and a test now asserts that.
  * **A port keeps its own proposal protocol.** `to_diff` still accepts the
    repo's `name :: body` rather than `FileTree`'s `<EDITS>` JSON. What is
    faithful about EvoSkill is the two-role Proposer/Generator induction, not the
    separator; switching would have changed the Generator's prompt, which is
    precisely the silent behaviour change a migration must not smuggle in.

  With `--backend claude-code` / `openhands` the skills now reach the agent as
  files in its workspace instead of inlined in every prompt.
- **A `None` op now deletes a key** instead of storing `None`
  (`EvolvingArtifact.apply`). `dict.update` could express "add" and "replace" but
  not "remove", which is invisible for a rules playbook and disqualifying for a
  file tree, where a key is a path and deleting a file is an ordinary edit.
  `None` was never a legal op value, so this is backward compatible.
- `evolve_skill_dir` / `evolve_agent_dir` / `evolve_agent_code` default to
  `self_verify=False` and `cheap_eval_tasks=4`, unlike the plain engine. Both
  defaults are right for a text artifact and expensive for this one: the first
  doubles the agent calls per proposal, and the second leaves the cheap layer
  pinned to the whole held-out set, which makes candidate *ranking* the dominant
  cost of a run when `eval_fn` runs a real agent.
- `TreeSpec.max_file_bytes` defaults to 28 000, below `AggregatorConfig`'s
  32 000-char trust region, and `validate_against` refuses a mismatch up front. A
  larger file could be loaded but never changed: every diff touching it is
  rejected as `oversized`, which surfaces five rounds later as "my reflector
  emits junk" rather than "that file was never editable".

### Added
- **`examples/efficiency.py` now produces every number `docs/efficiency.md`
  publishes.** Four of them had **no entry point in the repository at all** --
  the latency-distribution table, the `eval_concurrency` table, and the
  threads-and-the-GIL table were measured by hand and could not be re-run. Three
  new experiments, selectable with `--only`:

      --only distribution   overlap against four shapes of latency
      --only gate           wall-clock against eval_concurrency 1/4/8/16
      --only gil --model X  a real API round trip vs pure-Python arithmetic

  The GIL one needs a model (`ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`, or any
  id `agentdescent.agents.claude` can reach); the other two need nothing.

### Documentation
- **The offline measurements were re-taken on the ported runtimes, and two of
  them were being measured wrongly.** Both flattered the result, and both were
  found by re-measuring rather than by reading:

  * **`docs/efficiency.md`'s `stale%` column was understated by roughly a factor
    of two** — 86% and 10% for the two async rows, against a corrected 93% and
    25%. The async path ran its own staleness gate and `Aggregator` ran another
    over the survivors, both writing to the same meter, so every surviving card
    was counted twice in the denominator. Nothing about the runs changed; the
    numerator was always right.
  * **The throughput experiment's denominator included setup and the shutdown
    grace**, which are fixed costs and therefore fall hardest on the low-worker
    rows — which reads as superlinear speedup (8.2–9.4x for 8 workers against
    ~8.1x). It says "a fixed wall-clock window", so it now divides by the window
    it asked for.
  * **`self_verify` doubled what a counted rollout cost** in that experiment: the
    engine re-runs a proposal's own rollout for a before/after delta and counts
    only the first, so an injected 6 ms latency became 12 ms per counted rollout.
    The reference loop got that delta free. `AsyncConfig.self_verify` and
    `AgentDescent(self_verify=)` expose the knob; the throughput experiment turns
    it off, because dispatch rate is what it measures.

  Three more tables were re-measured, and two moved for reasons worth knowing
  rather than drift:

  * **the latency-distribution table** read 5.9x / 4.8x / 3.3x / 2.4x and now
    reads 1.8x / 2.4x / 2.2x / 1.7x. The old one isolated the rollout stage under
    a setup nobody could reproduce; the new one is end-to-end `evolve()` at
    default settings. Subtract the columns and the rollout saving is exactly what
    eight workers should buy -- the speedup is smaller because the ceiling is
    whatever in a round is *not* a rollout, which at default settings is the
    gate;
  * **the threads-and-the-GIL row** was 7.1x on `deepseek-v4-flash` and is 5.8x
    on `glm-5.2` (5.8 / 6.3 / 6.5 across three runs). The gap is the
    latency-distribution table restated: eight threads finish when the slowest
    finishes, and a reasoning model has a long tail. Its CPU row was 1.0x from a
    25 ms unit of work -- too small to measure -- and is 1.1x from a unit sized
    to take seconds;
  * **the `eval_concurrency` table** keeps its shape (serial, then linear, then
    flat past the held-out size) on a workload small enough to re-run.

  The stated variance band is honest about the machine now: across five runs the
  8-worker speedup landed between 7.83x and 9.15x, dominated by a single-worker
  baseline that itself varied 15%.

- **`docs/results.md`'s efficiency table is re-measured and says how.** Every row
  now names the command that produces it. Only one of the five needed a model at
  all -- the other four were stub-backend measurements that had simply never been
  scripted.

### Removed
- **Six resilience tests that had become duplicates.** `AsyncAgentDescent` was a
  separate implementation of the barrier-free loop when they were written, so
  driving failures through it tested something of its own. It is an adapter now,
  and they were asserting `async_evolve`'s behaviour through a wrapper --
  `tests/test_fault_matrix.py` asserts the same invariants directly against both
  engines, and `tests/test_worker_resilience.py` in more detail. One survives, as
  the only test of the adapter's `rollout=` seam: a broken seam would otherwise
  make every reference example quietly stop exercising failures.

### Changed
- **The two reference runtimes are adapters, not a second implementation of the
  loop.** `AgentDescent` and `AsyncAgentDescent` had their own round barrier,
  worker dispatch, snapshot staggering, merger thread, published head and
  backpressure, all feeding the shared `Ledger` and `Aggregator`.
  `docs/architecture.md` called that "a known wart rather than a design intent",
  and it kept costing: two measured fixes that had to be hand-ported, two
  early-stop epsilons nobody chose, and three mechanisms the general engine
  re-derived -- and got wrong -- because the reference stack already had them.

  They now describe the reference domain in the vocabulary `evolve()` and
  `async_evolve()` speak and run that. The public surface is unchanged
  (`RoundStat`, `AsyncStats`, `final_accuracy`, `buffer_pending`,
  `run_fork_baseline`), the sequential barrier is preserved so a seeded run stays
  reproducible, and both paths still converge on the same table with merge
  beating fork:

      AgentDescent (adapter)   first 0.604   final 1.000
      evolve() directly        first 0.604   final 1.000
      fork baseline (RQ1)      0.379

  Three things the translation does not preserve exactly, listed in
  `agentdescent/domains/router.py` rather than left to be discovered:
  `before_after_delta` and `evidence_eval` are measured over the whole cluster
  rather than the failing subset, and noise is per proposal rather than per
  worker -- the general engine has one `propose` for every worker and, by design,
  no worker identity to branch on. `run_fork_baseline` keeps per-fork noise,
  because a fork is one actor for its whole run.

  `AsyncAgentDescent` gains `rollout=` and `aggregator_factory=`, which are the
  seams `Worker.run` and a patched `aggregator.step` used to be: the resilience
  tests inject a failing rollout and a flaky merger through them.

### Removed
- **`agentdescent.worker.Worker`.** It modelled a rollout for a loop that no
  longer exists. What it did lives in
  `agentdescent.domains.router.router_propose` (the corrector) and `rollout=`
  (the latency injection the efficiency experiments need).
- **`AsyncConfig.aggregator_interval` and `AsyncConfig.worker_pause`.** They
  paced threads this module no longer owns; `async_evolve` owns its own sleeps.
  Nothing in the tests or examples set either.

### Removed
- **`EvidenceCard.version_annotations`**, described as "per-turn version
  annotations, used when the ledger hot-updates mid-rollout". Nothing wrote one
  and nothing read one -- no producer in the package, no consumer, no test, no
  doc page. A field on the object every worker constructs is not free: it reads
  as a capability, and the mid-rollout hot-update it names does not exist (a
  worker holds one snapshot for a whole rollout, by construction).

### Documentation
- **One table of everything provided, tested, and not in any engine path**, in
  `docs/modules.md`: `Ledger.commit_atomic`, `L1SerialGate`, `ResumeQueue`,
  `AuditScheduler.pop`, `EvidenceBuffer.settled`, `TaskScheduler`'s missing
  artifact axis, `PipelineParallel`. Each already said so in its own docstring,
  so finding out cost a read of the source, one class at a time. The rule they
  share is worth stating once: a primitive that is implemented and unreachable is
  honest, one that is reachable and silently does nothing is not.

## [0.3.0] — 2026-08-01

A measurement pass. 0.2.0 made the framework's claims honest; this release makes
the numbers behind them readable, and fixes the ports and engine paths where a
run could look like it had measured something when it had not.

The headline is that a silent failure mode ran through everything: on a reasoning
model, too small a token budget returns **empty visible content**, which does not
raise — it scores as a wrong answer. A starved run therefore reports a low
accuracy indistinguishable from a model that cannot do the task. ADAS's
meta-agent hit this on every call, so its search proposed nothing and the docs
recorded "no lift demonstrated" for a cause that was never the algorithm.

### Fixed
- **The single-sourced version was single-sourced onto the wrong number.**
  0.2.0 removed the duplicate version from `pyproject.toml` so it could not
  drift again — correctly — but pointed it at `agentdescent.__version__`, which
  had said `0.7.0` since the Concordia → AgentDescent rename while every wheel,
  the PyPI page and the badge said 0.2.0. Nothing surfaced it, because the value
  is only read at build time: the next GitHub Release would have published
  **0.7.0** to PyPI, and PyPI version numbers cannot be reused, so 0.3.0 through
  0.7.0 would have been unusable forever. Now `0.3.0`.
- **ADAS's meta-agent returned empty content on every call, so the search
  proposed nothing.** `deepseek-v4-flash` is a reasoning model: the token budget
  is spent on hidden reasoning first and the visible content is what is left. The
  example took the library default of 4096 — sized for "answer with the final
  number" — and used one completion for both the multi-step agent programs and
  the meta-agent, whose prompt is far longer. At 4096, **0 of 4** meta-agent
  calls returned anything at all; at 16384, 4 of 4 returned a well-formed design.
  The solver is affected too (blank replies 13/40 → 2/40), though the accuracy
  difference at n=40 is inside the noise. The defect is that it is *silent*: an
  empty completion does not raise, `_extract_int("")` is `None`, and `score_mgsm`
  scores `None` as a wrong answer — so a starved run reports a low accuracy
  indistinguishable from a model that cannot do the problems. Now `--max-tokens`
  (default 16384) and `--timeout` (300 s), blank replies are counted and warned
  about, and the pre-flight check sends a *reasoning* prompt and aborts if it
  comes back empty ("Reply with the single word: ok" passes at any budget, which
  is why it never caught this).
- **A design's identity ignored key order.** Every dedup in the search is string
  equality on the program's JSON, and a proposal's key order is whatever the
  model emitted, so two orderings of the same program were two designs. The
  duplicate passed the dedup, cost a full evaluation sweep, tied what it
  duplicated, and failed the acceptance test. Now `json.dumps(..., sort_keys=True)`.
- **A run that finished its search printed nothing if the test split failed.**
  The test split is scored after the search is over and paid for; letting a
  backend failure there escape discarded every validation number the run had
  produced. Test accuracy now degrades to `n/a` and the rest still prints.
- **ADAS reported "no lift demonstrated", and most of the reasons were not the
  algorithm.** The recorded run spent 791 calls to report `test accuracy 0.000`
  on three items. Fixed across the measurement, the search and the accounting:
    - The hard subset came from a 600-item sample of four languages, leaving 47
      items to split three ways. Full MGSM is 2750 items over 11 languages and
      `deepseek-v4-flash` answers 0.919 of it directly — a 222-item hard pool.
    - The split was 50/25/25, but the train half only ever *triggers* proposals:
      the meta-agent conditions on the archive, not on the task `evolve()` hands
      it, so a generation consumes two items and ignores the rest. `--train-frac`
      / `--test-frac` now default to 15/50/35, and splits are stratified by
      language (MGSM languages differ by 8 points of baseline accuracy, so an
      unstratified draw can hand val one mixture and test another — and that
      difference reads as a lift).
    - The run reported only the searched agent's test accuracy, which on a
      `--hard` subset cannot say whether searching helped: the structure-free
      baseline is 0.000 there by construction. It now scores the best seed on the
      same split and prints both rows and the delta.
    - The `--hard` baseline was scored with a digit-concatenating extractor while
      the agents used `_extract_int`, so "hard" partly meant "the two extractors
      disagree". One extractor for both sides. The printed direct accuracy was
      also derived from the subset size, so a pool `select_hard` had *topped up*
      reported the top-up as model error.
    - The seed archive was scored inside the first `step()` — after generation 0
      had already proposed — so the first generation designed against seven
      entries all reading `unevaluated`. Seeded before round 0 instead.
    - `propose_agent` returned whatever the last Reflexion round emitted,
      discarding a valid draft from an earlier round; `_parse_agent`'s greedy
      `\{.*\}` spanned first brace to last brace in the whole reply, so one brace
      in the closing prose threw the proposal away. A generation produces one
      proposal, so each of these costs a whole generation.
    - Nothing bounded a proposal's cost, which is multiplicative in this DSL (an
      ensemble of two 2-round debates is 14 calls per question against a
      most-expensive-seed of 5). Now capped, and the cap is in the meta-prompt.
    - `self_verify` ran an extra multi-step rollout per proposal for a delta the
      archive never reads; `--eval-concurrency` never reached `evolve()`; and
      `--hard` re-ran its entire baseline pass every invocation (2750 calls at
      full size), now cached per (model, question).
    - The budget line was computed *before* `--hard` narrowed the split, so it
      advertised ~9009 calls for a run that made 791. It is now derived from the
      real program costs, after the split, and reports a range.
    - `MetaSearchAggregator` left `MergeReport.accepted`/`category` empty while it
      was committing, so an unchanged best design printed `+0/-1` and
      `outcomes()` bucketed every generation as `unknown`.
- **The evaluation cache keyed on state the artifact does not render.**
  `_EvalCache` used the whole state dict, but `eval_one` only ever passes
  `render()` to `run` — so two states that render identically cannot score
  differently. ADAS keeps a design's name and rationale beside the design itself,
  and committing a candidate the aggregator had just scored therefore re-ran the
  entire held-out set because a label changed: a duplicate sweep of real model
  calls per committing round.
- **`RoundInfo.n_items` is the artifact's key count, not the sample size.** The
  verbose line printed it as `items=` directly beside the reward, where it reads
  as "measured on this many" — a 110-item measurement announced itself as 3. Now
  `size=`, with the held-out count printed next to the reward it belongs to.
- **The measured worker-retirement fix existed in only one of the two async
  pipelines.** `async_evolve` retires a worker only while *no* worker has ever
  succeeded, with a comment recording why: keyed on a worker's own history, an
  intermittent backend retires whoever loses its first few rolls, and since every
  worker shares one backend, shedding workers cannot relieve the throttling and
  only guarantees the run dies -- measured at a 1-in-3 call failure rate as all
  three workers retiring in 22s with nothing learned. `AsyncAgentDescent` still
  had exactly the blanket rule that paragraph describes. Ported, along with
  `retired_workers` so a run finishing at a fraction of its concurrency is visible.
- **The reference merger was still a single point of failure.** It ended the run
  on its first exception -- and it *calls the backend* every sweep, since it scores
  held-out. `async_evolve` removed that pattern after measuring a run end with 0
  sweeps while the workers were healthy; the same tolerance now applies here.

### Added
- **Backpressure on the general async path** (`async_evolve(stall_patience=)`).
  `concepts.md` documents this guard as what keeps a mismatched `async_ratio > α`
  from livelocking under Guarded -- workers propose against a snapshot too old for
  the policy to accept, every card is discarded, head never moves, so the lag
  budget never triggers a refresh either. It existed only in the reference
  runtime, which is not the one a real workload reaches. `result.forced_refreshes`
  counts how often it fired.
- **Duration-aware straggler detection on the general async path**
  (`async_evolve(duration_estimator=, straggler_factor=)` →
  `result.stragglers`). The design's L-traj mechanism was reachable only through
  `AsyncAgentDescent`, which accepts nothing but a `TaskUniverse`. Detection only:
  resuming a partial rollout would need it to expose its turns, and
  `run(rendered, task) -> output` is opaque.

### Fixed
- **`examples/rq2_staleness` swept a parameter the run never reads.** It varied
  `alpha_head`, which `Aggregator._alpha_for` consults only for an L1 artifact,
  while the reference `RouterSkill` is `blast_radius=0.2` -- so the live knob was
  `alpha_tail=min(alpha, 1)` and **three of the four published rows were the same
  configuration**. It sweeps both bands now, and alpha=0 genuinely separates:
  7 rounds to converge and 7 stale discards, against 4 rounds and 1 for alpha>=1.
- **Every published sweep reported `dev_acc=1.000` for every setting.** The router
  domain is reachable from all of them, including zero staleness tolerance, so the
  outcome column was constant and the experiments could not answer the question
  they were run to answer. `rq2_staleness` and `run_async` now report **cost** --
  rounds and rollouts to converge, and the share of rollouts discarded -- which
  does vary: `guarded` wastes 91% of its rollouts against `full`'s 0% and
  `reflective`'s 12%. Both scripts say plainly that accuracy saturates and the
  cost columns are the comparison.
- **The docstring-completeness guard was a substring match.** `test_api_docs.py`
  exists to keep `evolve` / `async_evolve` honest as their signatures grow, and
  checked `p not in doc` against the *whole* docstring. Delete `async_evolve`'s
  entire Parameters section and **20 of its 27 parameters still passed**, because
  its opening paragraph names them in prose; `evolve` kept 11 of 30 the same way.
  Substrings made it worse -- `run` matches "running", `agent` matches "agents",
  `parallel` matches "parallelism". It now parses numpydoc entries, plus a
  meta-test that fails if stripping the section leaves anything looking
  documented. `async_evolve`'s 15 cross-referenced parameters got a real entry
  rather than relying on the prose.

### Changed
- **The six custom optimizers in `examples/` take a lock.** They were safe only
  because every `ingest` happened to be a single `list.append`, atomic under the
  GIL -- which stops holding the moment `ingest` grows a counter or a dedup check,
  and is already not enough when `evolve(round_timeout=)` abandons a straggler
  that keeps running and can `ingest` mid-drain. `AggregatorProtocol` now states
  the contract it always relied on: `ingest` may be called from many worker
  threads, `step` from one, guard anything both touch.
- **`tests/faults.py` gained the three fault classes that had actually caused
  bugs.** Every primitive raised, so "the backend succeeds and returns nothing"
  -- the failure the codebase itself calls the most insidious, with a dedicated
  counter and warning in `LLMAgent.propose` -- had no way to be injected;
  `returns_nothing` covers it, and the matrix pins that a mute backend cannot
  produce a commit. Faults only ever wrapped `run`, so a reflector outage (a
  *separate* backend call, often to a different model) was untested;
  `flaky_propose` covers it. And nothing faulted the ledger -- the sole writer,
  on the critical path, with no retry anywhere -- whose failure escaping as an
  exception was found only by injecting it by hand; `ledger_dies_after` covers it.
- **`tests/test_bbh_example.py` tested `examples/skill_evolution.py`**, a stale
  name from a rename, so the test for that example was the one file nobody would
  look in for it. Renamed.
- **The five fully-offline examples had no test at all**, while all six algorithm
  ports -- which need credentials to do anything real -- had offline tests of
  their helpers, and CI runs `pytest` only. `tests/test_offline_examples.py`
  covers them, including the RQ1 merge-vs-fork advantage three doc pages quote.
- `docs/usage.md` said the algorithm ports "run offline with `--dry-run`". It
  still downloads the benchmark, and does not run the evolution loop.

### Fixed
- **The published version drifted five minor releases behind the code.**
  `__version__` said `0.7.0`; `pyproject.toml` said `0.2.0`, and the build backend
  reads the latter -- so every wheel, the PyPI page and the README badge were
  wrong. The version is now single-sourced from `agentdescent.__version__` via
  `dynamic = ["version"]`, and a test refuses a static version in `pyproject.toml`.
- **The README imported `LLMAgent` from the wrong module** (`agentdescent.agents`;
  it lives in `agentdescent.evolution`). Found by the new docs-import test the
  moment it was written, which is the point of it.
- **Trust-region rejections were counted as `all-stale`.** "My reflector emits
  500 KB values and every one is dropped" and "my lag budget is too tight" are
  opposite fixes, and only the second had a name. New `oversized` outcome.

### Added
- **`MergeOutcome`** -- a declared vocabulary for `MergeReport.category`, the keys
  of `result.outcomes()`. They were bare string literals written at six different
  return sites, so learning them meant reading `aggregator.py`; nothing could
  validate a typo, and a custom aggregator had no contract to meet. Subclasses
  `str`, so every existing lookup, comparison and format string is unchanged.
- **`evolve(solved_threshold=)`** and the `SOLVED` constant. `0.999` was written
  out four times -- twice in the drivers, once in a docstring, once as
  `DifficultyWeighted`'s default, whose own docstring says it "mirrors the engine".
  Right for a binary scorer; for a graded one (ROUGE, an LLM judge) nothing ever
  reaches it, so *every* rollout asks the reflector to fix an answer that scored
  0.95 and the run reports `below-threshold` as if the reflector were at fault.
- **`AggregatorConfig.anneal_half_life` and `accept_samples`.** `base_delta` was
  tunable but the half-life that turns it into the actual acceptance threshold was
  a default argument buried in `stats.annealed_delta`, unreachable from the object
  the docs call "tuning for the reference aggregator" -- and it sets the shape of a
  whole run (the threshold goes 0.505 at v1, 0.875 at v128, floors at 0.99).
- **A much wider top-level API.** `tasks_from` (documented, but importable only
  from `agentdescent.evolution`), the whole error hierarchy (`ContractError` and
  friends -- `evolve()` tells callers to distinguish a caller bug from a backend
  failure, and the base class was not reachable from the package that says so),
  the extension primitives `diffs_contradict` / `fuse_diffs` / `stable_hash` /
  `assign_key_sections`, and `GitError` / `LedgerFailure` / `FAST_MAX` /
  `FROZEN_IDS`.

### Changed
- **`domains.router.Task` is now `RouterTask`** (`Task` kept as an alias). It
  shadowed `agentdescent.Task` -- disjoint fields, no relationship, same name --
  and `orchestrator.py` and `worker.py` imported the other one, so a reader
  following `AgentDescent -> Worker -> Task` from the architecture page landed on
  the wrong class with no signal.
- **The docs now use the top-level API**: 53 `from agentdescent import ...`
  against 14 submodule imports, up from 3 against 63. `evolve` was never once
  shown as `from agentdescent import evolve`, which is why the top-level surface
  went untested and gaps in it went unnoticed. The remaining submodule imports are
  the deliberately module-scoped ones (`dataloader`, `rewards`, `backends`,
  `domains.router`).
- A new test resolves **every** `from agentdescent... import` across all 70 doc
  code blocks. 68 of them were executed by nothing at all, so a rename, a typo or
  an unexported name was invisible.

### Fixed
- **The DGM port ran a staged-eval rung upstream does not have.** `DGM_outer.py`
  passes exactly two subsets to each self-improve attempt -- `small` (10) and
  `medium` (50), one `test_more_threshold = 0.4` -- and `big.json` (140) belongs
  to the separate full-evaluation path, gated by the *archive-relative*
  `get_full_eval_threshold(...)`. The port ran `big` as a third rung on the same
  0.4, which changed what `agent.score` means: a high scorer's became a
  140-instance number while a low scorer's stayed a 10-instance one, and both then
  fed the same `dgm_parent_weights` sigmoid. The example's own docstring described
  upstream correctly ("big=140 for top agents") while its code did something else.
- **The GEPA port's admission test was a minibatch of one.** GEPA's Algorithm 1
  compares a child against its parent on a feedback minibatch of size *b*;
  `evolve()` rolls out one task per worker per round, so `before_after_delta` is a
  single-instance measurement -- exactly `{-1, 0, +1}` for a binary reward like
  HotpotQA EM. Gating on `> 0` therefore demanded that the one sampled instance
  flip wrong-to-right, and it is the instance the mutation was generated *from*.
  A prompt that helps broadly but does not fix that particular question was
  discarded before it was ever scored: rejected candidates never enter the pool,
  never get a `_score_row`, and so can never reach the Pareto frontier -- which is
  precisely the complementary specialist the frontier exists to keep alive. Now
  `>= 0`, which filters obvious regressions and leaves selection to domination
  pruning. Algorithm 2 itself is scored on the full `D_pareto` row and is
  unaffected.
- **EvoSkill's frontier bound was 3, upstream's is 5**
  (`src/registry/manager.py:379`), including the `--frontier` default.
- **ADAS seed name.** `Self-Consistency with CoT` is
  `Self-Consistency with Chain-of-Thought` in `get_init_archive()`.
- **Upstream citations pointed at paths that do not exist.** EvoSkill's
  `runner.py:79` / `:319` are `src/loop/runner.py`, and `registry/manager.py` is
  `src/registry/manager.py`. The **line numbers were exact** -- `:79` really is the
  tolerance ladder and `:319` really is the 0.8 pass/fail -- so only the prefix was
  missing, but it made the citations un-followable.

### Changed
- ADAS's bootstrap resample count (2 000 against upstream's 100 000) is now named
  as a deliberate speed trade in the docstring and on the fidelity page, rather
  than left for a reader to diff against the repo.

### Added
- `tests/test_port_fidelity.py` pins the constants and control flow that have an
  exact upstream source: DGM's selection weights, subset sizes and where the
  ladder stops; ADAS's seven MGSM seeds and the documented resample deviation;
  EvoSkill's tolerance ladder, pass threshold, weight formula and frontier bound
  (a top-K leaderboard, not the Pareto front the paper's abstract describes).

### Fixed
- **The L1/L2 boundary was defined three times, with two different numbers.**
  `governance.classify` drew it at `FAST_MAX = 0.30`; the aggregator's staleness
  tolerance re-derived it as `blast_radius > 0.5` and the audit gate as
  `blast_radius >= 0.5`. An artifact at 0.4 was therefore **L1 by governance** --
  the slow, conservative layer -- and treated as L2 by both mechanisms that decide
  what being L1 means: it got the staleness tolerance meant for a cold L2 skill,
  and no oracle audit at all. `evolve()`'s docstring papered over the gap by
  recommending 0.2 and 0.6, the two values where the thresholds happen to agree.
  Both sites now call `classify`.
- **A reserved artifact name failed late and blamed the wrong thing.** The L0
  frozen ids are ordinary words -- `oracle` is a plausible name for an evolving
  judge prompt -- and `evolve(artifact_id="oracle")` surfaced a `GovernanceError`
  on the first round that named governance rather than the cause, which is the
  *name*. Now refused beside the other `artifact_id` rules, before any rollout,
  with a message that says to rename it. Still a `GovernanceError`: refusing to
  mutate L0 is the safety claim, and callers are told to catch that type.

### Removed
- `governance.SLOW_MAX`. It was defined with a comment describing a frozen-layer
  rule, and `classify` never read it -- so 0.31 and 0.99 classified identically
  while the comment documented behaviour that did not exist. L0 is reached by id,
  not by radius, so one threshold is all there is.

### Changed
- **Documentation now matches the scheduler.** `TaskScheduler` was described as
  "UCB over (task-cluster x artifact)" in four places -- both `architecture.md`
  diagrams, `concepts.md` §5 and its own module docstring. There is no artifact
  dimension: `TaskCluster` has no such field, and both reference runtimes register
  exactly one artifact, as does `evolve()`. The missing axis is the one L-task is
  *about* ("head skills flooded, tail skills starved"), so the mechanism operates
  on clusters while the problem statement is about artifacts. Documented as
  not-implemented, alongside the tail canary set and partial-rollout resume.
- **`select_batch` no longer promises distinct leases.** It cycles when asked for
  more than there are clusters, and `TaskUniverse.clusters` drops empty hash
  buckets -- so on the default 24-keyword universe distinctness stops holding at
  12 workers, and at 24 workers 7 of them (29%) duplicate another's cluster,
  rolling out the same deterministic tasks for no extra evidence.
  `AgentDescent.__init__` now warns when it has fewer clusters than workers.
- **`L1SerialGate` is documented as a primitive, not as something in the path.**
  "At most one L1 diff in evaluation at a time" holds today by construction --
  every merge decision runs on one thread -- and the gate is what would enforce it
  once merges run concurrently. `concepts.md` said "implemented", which was only
  discoverable as untrue by grep.

### Fixed
- **`openai_compatible` returned `None` on reasoning models.** `Completion` is
  `prompt -> str`, but a model that spends its whole budget on `reasoning_content`
  answers with JSON `null` for `content` -- DeepSeek's reasoner and GLM's thinking
  modes both do. The `None` surfaced as
  `'NoneType' object has no attribute 'strip'` from inside `LLMAgent`, which the
  engine caught and **retried as a backend transient**, diagnosing a systematic
  model/parameter mismatch as a flaky endpoint. Doubly unfortunate: `LLMAgent`
  already carries the right diagnosis for a starved reasoning model, and never got
  to run it. Normalised to `""` so that warning fires instead.
- **HTTP errors discarded the provider's message.** The useful part -- "rate
  limit: retry in 12s", "context length exceeded", "insufficient quota" -- lives on
  `e.read()`, so re-raising bare collapsed every 4xx to `HTTP Error 429: Too Many
  Requests`, for the error class most likely to occur in a loop making thousands
  of calls. `_git` and `_CliAgent` both surface the underlying detail; this was the
  one provider path that did not. An HTTP 200 carrying `{"error": ...}` (some
  proxies) now names the endpoint and model instead of raising a bare `KeyError`.

### Added
- **`EvolutionResult.stop_reason`** -- `"target_reward"` / `"patience"` /
  `"rounds"` / `"max_seconds"` / `"max_iters"` / `"error"`. A run that converged
  and a run that ran out of budget both returned `error=None` with a populated
  `history`, and the only way to tell them apart was re-deriving `len(history)`
  against arguments whose meaning changes between the two paths. The `verbose`
  print lines always knew; now a non-interactive caller does too.
- **`evolve(shuffle=, seed=)`** (and on `async_evolve`). The train/held-out split
  is positional -- the last `held_out_frac` of `tasks`, in the order given -- which
  is right for a pre-split `Dataset` and wrong for grouped data. On a 20-task set
  whose first 12 are one class, the default holds out **0/8 of that class**;
  `shuffle=True` gives 5/3. Every gate in the run is measured on that set. Off by
  default so `Dataset.val_frac` keeps its meaning and seeded runs stay
  reproducible.
- **`openai_compatible(**create_kwargs)`** -- `temperature=0` and provider-specific
  fields now reach the request body, matching `claude()`.

### Changed
- `evolve(asynchronous=True)` warns about the two parameters it silently
  *redefined* rather than ignored: `max_seconds=None` becomes 20 seconds (it means
  "unbounded" on the synchronous path, so flipping one boolean could truncate a
  run into something that looked converged), and `rounds` becomes a
  `rounds x n_workers` rollout budget with `RoundInfo.round` as a sweep index. The
  three it *ignores* already warned.
- A held-out set smaller than 4 tasks warns: at 1 item `final_reward` is 0.0 or
  1.0 and nothing in between, yet it still gates every acceptance decision.

### Fixed
- **The oracle audit could never fire below `blast_radius=0.5`.** `force_oracle`
  gates on `blast_radius >= 0.5 or trust < 0.75`, and `update_trust` -- the only
  writer of trust -- sat *inside* that branch. The condition gated the one thing
  that could change it, so for any artifact under 0.5 it was unreachable: measured
  on the default `blast_radius=0.2`, `oracle_calls_used` stayed at **0** for a
  whole run and trust at its initial 1.0. An artifact at 0.4 -- which
  `governance.classify` calls L1, the *slow, conservative* layer -- received
  exactly as much scrutiny as an L2 skill: none. Cheap-vs-full agreement is now
  measured on every merge and costs nothing, since the Beta acceptance test
  already scores base and candidate on the full held-out set.
- **`evolve()` collapsed the three-layer verifier into one, so `oracle_budget`
  capped nothing.** It pinned `rule_subset=len(held_out)` with zero noise, on the
  reasoning that `eval_fn` is deterministic ground truth -- true of the synthetic
  router domain, and exactly backwards here, where `eval_fn` **runs the agent**.
  Rule, learned and oracle computed the identical number, so the aggregator bought
  a full held-out sweep for every candidate it merely wanted to *rank*, and the
  budget's documented fallback (`rule_eval`) returned the very value it was trying
  to avoid buying. New `evolve(cheap_eval_tasks=N)` / `async_evolve(...)` sizes the
  ranking sample; the acceptance test still scores the full set, so this trades
  ranking precision, never commit safety. Default `None` keeps exact scoring, so
  no existing run changes behaviour.
- **The cheap sample moved between calls.** `ThreeLayerVerifier._subset` drew a
  fresh `random.sample` every time -- harmless only while the "sample" was the
  whole set. The aggregator compares candidates *against each other* with it
  (`_resolve_conflicts` head to head, `_tournament` ranking all of them), so a
  moving sample scores candidate A on `{1,3,5}` against candidate B on `{2,4,6}`
  and calls the difference a winner. It also defeated the evaluation cache, which
  memoises per (artifact, task). Now drawn once per size.

### Changed
- `docs/concepts.md` §5 states plainly that `AuditScheduler`'s priority queue has
  no consumer: the audit that runs is the inline `force_oracle` gate, and the heap
  is a priority *model*, not work in flight.

### Fixed
- **The stable branch was never promoted, because `promote_after_k` counted
  commits instead of regression-free rounds.** The counter was bumped on the
  commit path, so it measured how many times an artifact had *changed* -- the
  opposite of what every description of it said ("survival rounds",
  "regression-free rounds", "EMA confirmation rounds"). The incentive was
  inverted: an artifact that converged stopped committing and could therefore
  never be promoted, while one that thrashed promoted every K commits. In
  `examples/run_demo` the artifact reached 1.000 held-out accuracy after two
  commits and `stable` then sat at **0.000 for all 40 rounds**, while
  `docs/usage.md` published a table showing it catching up at round 8 -- output
  this code could not produce. One round is now one `step()`; a commit restarts
  the clock (the new version has survived nothing yet) and so does an oracle
  rejection, while a `below-threshold` rejection counts as a round *survived*,
  because the gate turning a challenger away is the artifact winning. Promotion is
  idempotent per version (an unchanged head was being re-copied every K sweeps --
  52 times in one 6-second async run, each a handful of git operations under the
  lock every worker queues behind), and a clean run calls `finalize()` to publish
  its head, since `target_reward` fires on the very commit that reaches it and
  confirmation takes K rounds it will never get.

### Changed
- **The documented aggregator pipeline now matches the code.** All four copies --
  both `docs/architecture.md` diagrams, the `concepts.md` §4 list and
  `aggregator.py`'s module docstring -- listed the audit as stage 7, after the
  commit, drawn with a dotted "spot-check" arrow. It actually runs at stage **4**,
  before the Beta-posterior acceptance test, and returns `oracle-rejected`
  outright: a blocking gate on the accept path, not an advisory review. Three
  consequences the old ordering hid: the budgeted oracle sits on the critical path
  of every merge that trips `force_oracle`; `oracle-rejected` masks candidates that
  would also have failed the acceptance test, so `outcomes()` under-counts
  `below-threshold`; and `prob_improvement` runs 4000 Monte-Carlo draws before a
  gate that may discard the result unused.
- `docs/usage.md`'s `run_demo` output was refreshed against a real run (the fork
  baseline had drifted from 0.353 to 0.379).

### Added
- **An ungated dataset for EvoSkill — `--dataset finqa`.** OfficeQA is HF-gated, and
  the fallback was a bundled 12-row sample that splits into 5 train / 3 val / 2
  test — too small to measure anything, so every run reported **0.000** and read
  like a broken algorithm rather than a missing dataset. FinQA (`dreamerdeo/finqa`)
  is the same shape — a financial document plus a numeric answer to locate and
  compute — at 60 items with ~4 KB documents a non-tool model can actually read.
  Measured: val **0.487 → 0.573**, held-out **test 0.617**, one skill discovered.
- **`select_hard(items, score)`** — keep the items a baseline gets wrong, turning a
  saturated benchmark into one with headroom without swapping the dataset (which
  would break fidelity to the paper being ported). Wired into SkillOpt and ADAS as
  `--hard`. It refuses to return an unusable split: on a near-saturated benchmark
  the survivors can be a handful, and 3 items either measure nothing or crash the
  engine's train/held-out split, so it tops up to `min_items` and warns with the
  fraction of the pool that was already solved.

### Measured, after setting the difficulty
- With difficulty set, every port that has a gap now shows one. Full setups and
  before/after on the [results page](docs/results.md):

  | | before (default settings) | after |
  |---|---|---|
  | ACE, FiNER-139 | 1.000 → 1.000 at `--top-k 10` | `--top-k 120`: **0.844 → 0.889**, test 0.884 |
  | SkillOpt, SearchQA | 0.900 → 0.900, 0 edits accepted | `--hard`: **0.250 → 0.500**, test 0.450 |
  | EvoSkill | 0.000 → 0.000 (12-row gated fallback) | FinQA: **0.487 → 0.573**, test 0.617 |
  | GEPA, HotpotQA | — | **0.500 → 0.600**, test 0.700 |
  | DGM, surrogate | — | **0.000 → 0.300** |
- **`eval_concurrency=`** — how many held-out tasks a gate scores at once, the
  merge half of the run's parallelism and independent of `n_workers`. It existed
  only as a default on a private dataclass, which made it both unreachable and
  *unmeasurable*: setting the class attribute silently did nothing, because a
  dataclass bakes its defaults into the generated `__init__`. Measured on the same
  workload: **193.6 s at 1, 96.7 s at 4, 90.0 s at 8**, saturating once it reaches
  the size of the held-out set.

### Fixed
- **TensorParallel silently discarded 75-88% of every worker's proposals.**
  `plan()` sharded **task ids** through `section_of`, while `evolve()` enforced the
  section against the **artifact keys** the resulting diff wrote -- two unrelated
  key spaces, so a worker's legal tasks said nothing about its legal edits. With
  `SingleSlot` the key is a constant, so one section owned everything and the other
  workers could never commit at all; with `AppendRules` the key is a content hash,
  so legality was a coin flip. Nothing reported it: the rejections were appended to
  a list that was never read, so a TP run that threw away most of its work looked
  exactly like one whose reflector had nothing useful to say. Tasks are now sharded
  data-parallel and the section is a separate axis; the pairing is validated before
  the first rollout (a strategy with no declared key space, or more sections than
  keys, is refused with a message naming the fix); every rejection is counted as
  `section-violation` in `result.outcomes()`; and the new `TensorParallel(route=)`
  maps a task to the artifact key its failure will edit, so each worker is handed
  only tasks it may act on and TP delivers exactly what DP does.
- **`section_of` was a hash bucket, not a partition.** On four keys and four
  sections it put two keys in one section and left another owning nothing, so the
  worker holding it could never commit. `assign_key_sections` partitions a declared
  key space evenly and deterministically instead.
- **`parallel=PipelineParallel(...)` was accepted and quietly ignored.**
  `WorkUnit.stage` -- the only thing distinguishing PP's units, since it hands every
  worker the whole task list -- was never read by any driver, so PP degraded to
  n_workers redundantly rolling out the same tasks: strictly worse than the default,
  with no signal. Measured on 24 tasks and 3 workers, it covered 14 distinct tasks
  against DP's 22, with three workers on the same task in one round. `evolve()` now
  raises and points at `PipelineChain`, which is where PP's stage ordering and blame
  attribution actually live.
- **A personal `~/.gitconfig` could stop `evolve()` before it ran a single task.**
  The ledger shelled out to plain `git`, so `commit.gpgsign = true` -- a common
  setting, and the default in several corporate onboarding scripts -- failed the
  genesis commit and raised `GitError` out of `Ledger.__init__`, from a call whose
  signature mentions git nowhere. A global `core.hooksPath` ran the user's
  `pre-commit` hook against a temp directory it knew nothing about. These are the
  ledger's own bookkeeping commits in a scratch repo the caller never sees, so git
  now runs with an isolated config (`GIT_CONFIG_NOSYSTEM`, signing and hooks off)
  plus `GIT_TERMINAL_PROMPT=0` and a timeout, so a credential prompt or a stalled
  filesystem surfaces as an error instead of wedging every worker behind the
  ledger lock. A missing `git` binary now says so by name instead of raising a
  bare `OSError`.
- **A ledger failure mid-run escaped as an exception, discarding the artifact.**
  `EvolutionResult` documents that "a run that died still returns a (partial)
  result rather than raising", and every rollout, reward and merge call site was
  wrapped -- but the ledger's five call sites were not. The worst case: `ledger.log()`
  was fetched *inside the `return` expression*, purely to fill the cosmetic
  `ledger_log`, so a git failure there threw away a run that had already completed
  every round and computed its final reward. A ledger failure is now its own
  category alongside a caller-contract violation (raises) and a backend blip
  (absorbed): it ends the run, names itself in `error`, and still hands back what
  was learned. `ledger_log` degrades to `[]`.
- **Scratch ledgers were reclaimed only at interpreter exit.** `atexit` does not
  run on SIGKILL or an OOM kill, so each such death leaked a git repo into
  `$TMPDIR` -- 115 directories totalling 19 MB accumulated on one machine in a
  single day. Worse, inside a notebook or a parameter sweep `atexit` fires only
  when the *process* ends, so every run in the process held a live repo. `evolve()`
  and `async_evolve()` now remove their own scratch ledger on the way out (a
  caller-supplied `repo_path` is never touched -- it is how a run is resumed), and
  each run first collects orphans older than a day.
- **A transient network error outside the engine discarded a whole run.** The
  engine retries its own evaluations, but an example's *final* held-out scoring is
  a plain `completion(...)` call with no cover — measured, one
  `RemoteDisconnected` there threw away a complete EvoSkill run. `claude()` and
  `openai_compatible()` now retry (`retries=3`, `0` opts out), which covers every
  caller rather than each call site.
- **ACE's difficulty default demonstrated nothing.** `--top-k` is the difficulty
  knob and defaulted to 10, where `deepseek-v4-flash` scores **1.000** and there is
  nothing to learn. Raised to **120**, the first value that actually demonstrates
  the algorithm: at 40 there is headroom (0.850) but no bullet beats the baseline,
  so the gate rejects everything; at 120 two bullets survive and val goes
  **0.844 → 0.889**.
- **The merge gate was serial, and it dominated the run.** `EvolvingArtifact.score`
  summed a generator, so every held-out evaluation ran its tasks one at a time --
  and the aggregator calls it once per candidate, so a round paid N x held-out
  rollouts sequentially while the workers that produced those candidates ran in
  parallel. Measured on HotpotQA with a reasoning model: **~25 min per round ->
  ~5 min**. Evaluation is memoised and lock-guarded, so this is only a matter of
  fanning it out (`_Runtime.eval_concurrency`, default 8; set 1 for the old
  behaviour).
- **`last_number` read the gold as a bare number and silently scored everything
  zero.** A dataset's answer column is often the whole worked solution — GSM8K's
  ends `#### 72` — and parsing that as a number fails, so *every* item scored 0.
  The failure is invisible: it reads as a hopeless model, not a scorer mismatch.
  Measured on real GSM8K with `deepseek-v4-flash`, this was the difference between
  a reported **0/7** and the true **7/7**. The gold is now read the same way as the
  output, and a gold containing no number at all raises instead of scoring zero.
- **A transient during *merge decisions* ended a synchronous run.** The third
  unprotected backend call site, after the round and final scoring: the aggregator
  runs the agent for its own accept/reject comparisons (`cheap_eval`,
  `eval_counts`, `oracle_eval`), and a blip there propagated out of the round.
  Rather than guard each site, the single memoised evaluation every one of them
  funnels through now retries — so a retry re-runs only the task that failed, and
  the round scoring no longer loses a whole measurement to one unlucky task (on a
  30-task held-out set at a 1% per-call failure rate, ~26% of rounds measured
  nothing). Contract violations are not retried.
- **An abandoned straggler kept the process alive.** `round_timeout` documents that
  a slow worker is abandoned and the run continues, and it was — but the round ran
  on a `ThreadPoolExecutor`, which registers an atexit hook that *joins* its
  workers. So `shutdown(wait=False)` bounded the round and not the program:
  measured, a rollout wedged for 600 s printed its result and then held the
  interpreter open indefinitely. Rounds now use daemon threads with a semaphore
  preserving `max_concurrency`; the same case exits in **4.5 s**. A `ContractError`
  raised in a worker is carried back to the main thread by hand, since an exception
  in a plain thread goes to the excepthook rather than propagating.
- **A single transient ended a whole *synchronous* run** — the default path, and
  the one every shipped example uses. A worker's exception propagated out of its
  future and broke the round loop, with no retry or tolerance anywhere: measured,
  one 429 on call 5 turned a 20-round run into **0 rounds**. A failing worker now
  costs its own evidence and nothing more, and the round merges what the others
  gathered; the give-up rule is the same global signal used on the async path,
  counting consecutive rounds in which *every* worker failed. A genuinely dead
  backend still ends the run in under a second. Contract violations still
  propagate.
- **A failing per-round held-out score raised out of `evolve()`.** It sat outside
  the round's error handling although it runs the agent like any other backend
  call, so a blip discarded every commit the run had already made. It is now
  treated as a failed round, carrying the last known reward forward.
- **A flaky backend killed the whole async run.** Measured against a real endpoint
  refusing 1 call in 3 (~56% per rollout — an ordinary 429 storm): the run ended
  after **22 s with 0 sweeps and nothing learned**, while two thirds of calls were
  succeeding. Two causes, both now fixed.

    *Workers* retired on 3 consecutive failures regardless of context. Retirement
    now keys on a **global** signal — if no worker has *ever* succeeded the backend
    is misconfigured, so fail fast; once any has, the backend demonstrably works, so
    nobody retires and they back off instead. Shedding workers could never have
    helped anyway: they all share one backend. The signal is global because keyed
    per-worker, an intermittent backend retires whoever loses its first few rolls
    (~30% of them at a 2-in-3 failure rate). `max_worker_errors=` is now a
    parameter, and `result.retired_workers` reports the count — a run can finish
    *cleanly* at a fraction of its requested concurrency with `error` still `None`.

    The *merger* had a single try/except around its whole loop, and it calls the
    backend every sweep to score held-out — so one transient took it out
    permanently and the run reported 0 sweeps while every worker was healthy. It now
    retries with a short backoff and never ends the run itself, since a dead backend
    already retires the workers. A `ContractError` raised there (a broken custom
    aggregator) used to be absorbed and reported as a provider outage; it now
    propagates, as documented. After the fix the same 1-in-3 run reaches **1.000
    held-out** and learns the rule, with 24 of 70 calls still failing.
- **`error` conflated "the run died" with "the final measurement failed".** A
  transient on the last held-out scoring of an otherwise healthy run was reported
  as a run-ending failure. Scoring is now retried (it is memoised per task, so a
  retry re-runs only what failed), and if it still cannot be made the message says
  so and that `final_reward` fell back to the last measured round.
- **`SingleSlot`'s docstring did not compile.** It advertised
  `SingleSlot(initial=...)` when the field is `initial_value`, and described a
  `keep_longest` parameter that never existed. A test now walks every dataclass in
  the module and fails when a docstring's constructor example names a field that
  is not there.

### Added
- **`evolve_skill()` — a dataset to an evolved skill in one call.** Evolving a
  skill needs three things that are genuinely yours: your data, how to score an
  answer, and which model. Everything else was boilerplate everyone rewrote
  identically — wrapping rows as `Task`s, the lambda that puts the skill in front
  of the question, the same last-number regex, and a dozen knobs a first-time user
  has no basis to choose. The same real program goes from **21 lines to 11**, and
  from ten decisions to three. It is a thin wrapper: same engine, same
  `EvolutionResult`, every default overridable, and any extra argument passes
  straight through to `evolve()`. Measured end to end on 40 real HotpotQA items
  with `deepseek-v4-flash`: held-out exact match **0.167 -> 0.583** in four rounds,
  learning *"Respond with only the requested answer, omitting any extra
  explanation or restatement."* -- exactly the failure it was shown.
- **`agentdescent.rewards`** — `last_number`, `exact_match`, `contains`,
  `numeric_close`. Ready-made scorers that get right the details that are easy to
  get wrong: thousands separators, a trailing period, a model that answers in a
  sentence.
- **`tasks_from(rows, prompt=, gold=)`** — the six lines everyone writes after
  loading a dataset, including the `enumerate` for ids and the `meta` dict that
  both the scorers and the reflector read.
- **A fault-injection harness in the suite (`tests/faults.py`).** Every resilience
  bug so far surfaced only under a real fault — a dead socket, a wedged thread, a
  process that would not exit — each found by a throwaway script that then
  disappeared. The faults are now reusable (`never_works`, `flaky`, `dies_after`,
  `recovers_after`, `wedged`, `slow`) and a matrix runs each against **both**
  engines, asserting outcomes rather than exceptions: a run never hangs, never ends
  silently, survives anything recoverable, and gives up fast on anything not. It
  found the merge-decision gap above on its first run.
- **`result.outcomes()` — why the run went as it did.** A run that committed
  nothing reported `rejected: 3` and no more, though the aggregator had computed
  the reason and thrown it away. Merge outcomes are now tallied by a stable
  category on `RoundInfo.reasons` and across the run via `outcomes()`, on both
  the sync and async paths, and they survive `save()`/`load()`. The distinction
  matters because the fixes are opposite: `below-threshold` means proposals
  reached the acceptance gate and lost (the reflector is the problem),
  `all-stale` means they never reached it (the lag budget is). `MergeReport`
  gains `category` alongside `reason`, which interpolates measured values and so
  makes a useless tally key.

### Fixed
- **The settled-evidence pool grew without bound.** Every discarded card — stale,
  oversized, CAS-conflicted — is `settle()`d, and **nothing in the library reads
  the pool back**, so it was a pure accumulator. Worse, the oversized-diff path
  settles precisely the payloads the trust region exists to reject: 500 diffs from
  a reflector echoing its input retained **250 MB** unreachable by any code path
  (now 2 MB). It is bounded to `SETTLED_MAX_CARDS=256` / `SETTLED_MAX_CHARS=2M`,
  newest kept. The docs claimed discarded evidence "settles back into the pool for
  reuse"; they now say plainly that reuse is not implemented and point at the
  SkillOpt example as the worked version of it.
- **`evolve(asynchronous=True)` no longer drops knobs in silence.** `patience=`
  was accepted and never forwarded, so an async run ignored it entirely; it is
  now implemented in the async runtime, counting **merge sweeps** since there are
  no round barriers. `parallel=`, `max_concurrency=` and `round_timeout=` genuinely
  have no meaning there (the runtime shards round-robin across `n_workers`, and
  there is no barrier to bound) and now raise a `RuntimeWarning` naming the ignored
  argument — previously `parallel=TensorParallel(4)` looked honoured while the run
  was plain DP. A test now walks `evolve`'s whole signature and fails if any future
  argument is neither forwarded nor warned about.

### Added
- **The reflector can see `Task.meta`.** It previously received the score and
  nothing else — told *that* it was wrong, never what right looks like. That made
  any **convention** it could not guess (an output unit, a format, a required
  field) permanently unlearnable, no matter how many rounds it ran. `meta` is
  free-form and caller-owned, and every shipped port already puts the expected
  answer there. Rendered truncated (`meta_chars=600`) so a document in `meta`
  cannot blow up the prompt; the template asks for a *general* rule so the
  reflector does not simply restate this task's answer; `show_meta=False` opts
  out. Custom `propose_template`s that lack the new field keep working.
  Verified on a real two-step `deepseek-v4-flash` agent over 12 money word
  problems scored in integer cents — a convention stated nowhere in the prompt:
  the initial prompt gets **3/12**, a reflector blind to `meta` plateaus at 0.500
  over 8 rounds, and one reading `meta` reaches **12/12 in a single round**. It
  generalised rather than memorised, writing *"Express all monetary amounts as
  integers representing cents, without dollar signs or decimal points."*
- **`SingleSlot`** — the artifact *is* one value (a system prompt, an instruction)
  and each accepted proposal replaces it. The most common thing anyone evolves, and
  until now every caller wrote it themselves: three of the six shipped ports each
  rolled their own variant, and the docs offered it as copy-paste.
- **Early stopping — `evolve(target_reward=..., patience=...)`.** A run spent all
  `rounds` regardless of whether the artifact was still changing. Measured on a
  workload that converges in two rounds: 20 rounds cost 141 model calls for a
  result reached at 69, so **51% of the budget bought nothing**. `target_reward`
  stops at a held-out score, `patience` after N rounds without improvement; both
  default to off, so existing runs are unchanged.
- **`reflector(completion)`** — turn any model into the thing that looks at a
  failure and says what to change, so you can keep your own agent as `run=`.
  Evolving an agent you already have is now three lines: adapt it, pick a
  reflector, say what evolves — and switching parallel↔async is one argument.

### Fixed
- **The default `max_tokens` silently halved reflection with a reasoning model.**
  Such a model spends its budget reasoning before emitting anything, so too small
  a cap returns an *empty* completion, which the engine reads as "nothing worth
  changing" — a run then looks incapable of learning while the reflector never
  spoke. Measured on `deepseek-v4-flash`: at the old default of 1024, **4 of 8**
  reflection prompts came back empty; at 3000, none did. Both adapters now default
  to 4096 (billing is per token generated, not per cap), and an empty reflection
  emits a `RuntimeWarning` naming the likely cause. Found by evolving a real
  multi-step agent end to end, where it presented as "evolution does not work".
- **A custom aggregator's mistakes surfaced as cryptic crashes.** `aggregator_factory`
  is the main extension point and all six shipped ports use it, yet a class missing
  `ingest` failed with an `AttributeError` mid-run, `step()` returning `None` gave
  `'NoneType' object is not iterable`, and a wrong element type gave
  `'str' object has no attribute 'committed_version'` — none naming the aggregator.
  The protocol is now checked at construction and `step()`'s return is validated.
- **Caller mistakes were reported like provider outages.** `RewardContractError`,
  `ProposalContractError` and the new `AggregatorContractError` now share a
  `ContractError` base that both engines let propagate, while backend failures stay
  absorbed into `result.error`. A broken contract makes the run meaningless, so
  hiding it just spends the budget.
- **The trust region bounded op *count* but not op *size*.** A runaway proposal —
  a reflector echoing its input, say — committed a 500 KB value that then rendered
  into every later prompt, exploding cost and context silently. `AggregatorConfig`
  gains `trust_region_chars` (default 32k, ~12x the largest real op in the shipped
  ports).
- **A non-text proposal failed as `'int' object has no attribute 'strip'`**, deep
  inside a strategy, and was reported as a backend failure. `propose` returning
  anything but text or `None` now raises `ProposalContractError` naming the task
  and the contract, on both engine paths.

## [0.2.0] — 2026-07-30

A correctness and honesty pass over 0.1.0. Most of what changed was not a crash
but a **silent** wrong: flags the engine accepted and ignored, budgets that
counted without capping, a probability reported as a reward, seeded runs that were
not reproducible, and several mechanisms the documentation described as working
that no code path actually reached. Where a claim could be made true it was made
true; where it could not, the docs now say so.

Also: one contract for every backend (API model, CLI agent, OpenHands), real cost
accounting, and the advertised examples now run.

### Added
- **One contract for every backend.** The framework had two unrelated agent
  interfaces: `Completion` (`prompt -> text`) for API models and
  `AgentBackend.answer(question, document, skills)` for tool-using ones — a
  signature with three *domain* concepts baked into what should be the general
  interface. Now every backend is a `Completion`:
  `cli_agent(command)` runs **any** command-line agent (prompt on argv or stdin,
  stdout is the answer), with `claude_code()` and `codex()` as presets and
  `openhands()` as the SDK equivalent. Failures raise `AgentError` carrying the
  agent's own stderr, and each takes a `timeout`.
- **`WorkspaceAgent`** — the one optional capability an *acting* agent needs:
  `agent.in_workspace(path)` returns a completion bound to that directory. Plain
  API models deliberately do not implement it, so consumers feature-detect.
- **`backends.document_agent(completion)`** — the OfficeQA shape is now an explicit
  *domain adapter* over the general contract, and it adapts to what it is given: a
  workspace agent gets a scratch directory with the document staged (so it can
  really grep a 1 MB table), a plain completion gets it inline and truncated. The
  same example therefore runs on OpenHands, Claude Code, Codex, or a bare API
  model — `evoskill --backend claude-code|codex` are now available.
- **Acceptance decisions in a round were correlated by a shared Monte-Carlo seed.**
  `prob_improvement` is an MC estimate (sd ~0.003 at 4000 samples) and was seeded
  from the artifact version alone, so every candidate in a round drew the *same*
  stream. On a knife-edge case (measured: true P = 0.748 against a 0.750 threshold,
  where 26 of 60 seeds accept) that meant one draw accepted every marginal diff and
  another rejected all of them, instead of deciding them independently. The seed now
  includes the candidate's diff id via `stable_hash`, so draws decorrelate while
  runs stay reproducible across processes.
- **`document_agent` no longer truncates in silence.** Validating the inline path
  on real OfficeQA documents (266–390 KB) gave 1/3 correct with two *empty*
  answers: at the default `inline_chars` about half of each document never reached
  the model, and the figure was sometimes in the dropped half — indistinguishable
  from a model failure. Truncation now emits a `RuntimeWarning` naming the sizes
  and pointing at the fix (pass a workspace agent, which reads the file itself).
- **The reward contract is enforced.** `reward` must return `[0, 1]`; the engine
  treats `>= 0.999` as solved, so a scorer on a 0-100 scale silently made *every*
  task look solved — `propose()` was never called, nothing was learned, and
  `final_reward` came back as a healthy-looking `85.0`. Out-of-range or
  non-numeric returns now raise `RewardContractError` naming the offending task
  and what to do, on both engine paths, and it propagates rather than being
  reported as a backend failure (it is a caller bug, so the run is meaningless).
- **`evolve(round_timeout=...)`** — cap how long a round waits for its concurrent
  workers. The aggregator *is* the barrier, so one hung rollout previously stalled
  the run indefinitely; stragglers are now abandoned (their work continues in the
  background, since Python cannot cancel a thread) while genuine backend errors
  still surface. Verified: a 30s hang no longer blocks a run that finishes in 2.5s.
- **Task samplers (`agentdescent.sampling`)** — `evolve(task_sampler=...)`. A rollout
  is the expensive unit of work, and spending it on a task the agent already solves
  teaches nothing. `DifficultyWeighted` prefers tasks whose pass rate sits away from
  the all-pass / all-fail extremes (the zero-advantage filter), landing ~1.6-2.2x
  more rollouts on informative tasks than the `RoundRobin` default in measurement.
  That is a *targeting* result: on a real ACE/FiNER run the sampler reached a
  lesson sooner but scored lower than round-robin, so it ships opt-in with the
  caveat documented rather than as a default.
- **`Usage` cost accounting (`agentdescent.agents`)** — `claude(usage=...)` and
  `openai_compatible(usage=...)` now keep the **real** token counts the API
  returns (they were discarded at the `prompt -> text` boundary), plus calls,
  failures and model wall-clock; `metered()` covers any other completion.
  Thread-safe, and `estimated_cost()` takes the prices so no stale price table
  ships with the library.
- `evolve(on_round=...)` / `async_evolve(on_round=...)` — a progress callback per
  round (per merger sweep when async). A long LLM run previously reported nothing
  until it returned; a callback that raises is warned about, never fatal.
- `EvolutionResult.save(path)` / `.load(path)` — persist the evolved artifact and
  its run summary as JSON instead of hand-rolling the same serialisation.
- `agentdescent.dataloader` / `agentdescent.backends` are now importable from the
  package namespace (`Dataset` and `split_dataset` are re-exported).
- Full parameter documentation for `evolve()` (25) and `async_evolve()` (23) —
  previously 9 and 5 were described, including knobs that silently change cost
  (`self_verify`) or bound the run (`max_seconds`, `max_iters`). A test keeps the
  docstrings in step with the signatures.
- `tests` CI workflow — runs the offline suite on push/PR across Python 3.9 / 3.11 / 3.12.
- Test coverage for the async SGD path: `SgdSkillAggregator` keep/rollback,
  `eval_at_end`, batch-level propose, and the sync frontier gate.
- PyPI Trusted-Publishing workflow (`publish.yml`): OIDC release, no stored token.
- `EvolutionResult.error` — `None` on a clean run, otherwise the backend failure
  that ended it early, so callers can tell "converged" from "died".

### Performance
- **Ledger reads no longer fork a `git checkout` per call.** Every `snapshot()` /
  `head_version()` switched branches unconditionally — ~19 ms each, serialised on
  the ledger lock, capping the pipeline at ~50 ledger ops/sec regardless of
  `n_workers`. The current branch is now tracked, so a read on the branch already
  checked out costs 0.02 ms (900x), and `run_demo` runs end-to-end in 2.2 s
  instead of 4.7 s with byte-identical results.

### Fixed
- **The reference runtime had no error handling at all.** `async_runtime.py` and
  `orchestrator.py` contained zero `except` clauses, so a failing backend printed
  tracebacks from dead worker threads, the run span out its **entire**
  `max_seconds` with no producers, and it returned `rollouts=0, accuracy=0.000` —
  a normal-looking result with no way for the caller to tell. It now retries
  transient failures, retires a worker after 3 consecutive ones, ends the run once
  every worker has retired (20s budget → returns in 6s), guards the aggregator
  thread the same way, and reports the cause through the new `AsyncStats.error`.
  This is the same treatment the general engine got; the reference stack still
  drives the `run_async`, `efficiency` and `duration_scheduling` examples.
- **A wall-clock-dependent test was flaky in CI.**
  `test_guarded_discards_more_than_reflective` asserted an absolute accuracy
  (`>= 0.95`) from a run bounded by 12 seconds, so how far it converged depended on
  how many rollouts the machine fitted into that budget — it passed locally and on
  the 3.11/3.12 runners and failed on the slower 3.9 one at 0.83. It now asserts the
  relationship it is named for (Guarded discards more, Reflective wastes fewer
  rollouts and is never behind), which holds at any machine speed.
- **`result.history` means different things on the two paths.** Synchronous
  `evolve(rounds=5)` yields exactly 5 entries; `async_evolve` appends one per
  non-empty merger *sweep* — 221 in a 3-second run — and the count is bounded by no
  argument. The docs described only the former. Now stated in both the guide and
  the docstring, with a test pinning each.
- **`TensorParallel` was not tensor parallelism.** `evolve()` read only
  `WorkUnit.keys` and `WorkUnit.worker` and ignored `WorkUnit.section`, so TP's
  defining guarantee — each worker owns a disjoint section, which is what makes the
  merge a conflict-free union — was never enforced: with four workers all proposing
  an edit to the same hot key, **all four landed**. A worker's diff is now rejected
  if it touches a key outside its assigned section, so only the section owner can
  edit it. Pipeline parallelism remains unenforced (`evolve()` evolves a single
  artifact, so there is no chain for stages to walk) and the docs now say which of
  the three paradigms the engine actually honours — and that `async_evolve` shards
  round-robin itself, ignoring `parallel=` entirely.
- **`pip install agentdescent` could not run any documented example.** README and
  docs contain ~30 `python -m examples.…` commands, but `examples/` ships with the
  repository, not the wheel (a top-level `examples` package would squat the name).
  Verified by installing the built wheel into a fresh venv outside the repo: the
  library and dataloader work fine there, the examples are simply absent. The
  install instructions now say a checkout is needed, right where they say `pip
  install`, and tests pin both the packaging decision and the caveat's placement.
- **The README Quickstart did not run.** It used undefined names (`tasks`,
  `reward`), so copy-pasting it — the first thing a new user does — raised
  `NameError` immediately, and it also required API credentials. It is now
  complete, runnable with no key and no dependencies, and a test executes both it
  and the usage guide's entry-point example so they cannot rot again.
- **The usage guide's "Programmatic use" section showed only the reference stack**
  (`AgentDescent` / `AsyncAgentDescent`), not `evolve()` — the documented entry
  point every algorithm port actually uses. It now leads with `evolve()` and labels
  the reference stack as the experiment-reproduction runtime it is.
- **A governance violation was slow and misreported on the async path.** Only the
  reference aggregator's per-merge guard caught an L0-frozen target, so nothing was
  ever mutated — but `async_evolve` burned its whole `max_seconds` budget first and
  then reported the violation through the *backend failure* channel. Both paths now
  check governance before the first rollout and raise `GovernanceError` at once.
- **The quoted efficiency numbers were stale.** The tables cited a specific older
  run (7.92x at 8 workers, 2.53x async); after the ledger read optimisation the
  measured figures are ~8.1x and 2.57–2.93x. Updated, and the pages now say to read
  scaling efficiency as "≈1.0 within noise" rather than as a precise constant —
  values slightly above 1.0 come from the single-worker baseline absorbing the same
  fixed start-up inside its timed window, not from a superlinear effect.
- **`--provider glm` was misleading** — it means "any OpenAI-compatible endpoint",
  and every real run in these docs used it to reach *DeepSeek*. Examples now accept
  `--provider openai` with `glm` kept as an alias, and the help text says so.
- **The commit stage was described as "CAS / 2PC".** `commit_atomic` exists and is
  tested, but the reference aggregator buckets per artifact and no engine path
  calls it, so 2PC is an available Ledger capability rather than pipeline
  behaviour. Corrected in four places.
- **Two more documented-but-absent features.** The "tail canary set" inside
  held-out eval and the L1 staged rollout ("counterfactual replay → canary →
  full") do not exist; the docs now say so. Dual-branch promotion was described as
  "after *K* regression-free rounds" when it fires every *K* accepted **commits**
  and has no separate regression check (a round that merges nothing does not
  count) — verified working end to end, only the description was wrong.
- **The async staleness gate ignored `agg_config`.** It hardcoded `alpha = 5/1`
  while the aggregator behind it read `alpha_head`/`alpha_tail`, so a tightened
  staleness tolerance was honoured in one place and not the other.
- **Conflict detection was described as "syntactic and semantic"** but only
  semantic contradiction gates resolution — correctly so, since two diffs
  proposing the *same* value for a key are duplicates rather than a conflict. The
  docs now say that, and `diffs_conflict` documents itself as an unused primitive
  for custom aggregators.
- **`ResumeQueue` / "partial rollout" was documented as implemented but is not.**
  The README, concepts page and analogy table promised "turn-level checkpoint +
  `ResumeQueue`, resumed against the latest ledger (a free cross-version A/B
  signal)". In fact the rollout is never interrupted (the flag is recorded *after*
  it returns), the queued item carries no continuation state (`turn=0`,
  `conversation=[]`), and **nothing pops the queue**. The docs now describe what
  the code does — straggler *detection and accounting* — and say plainly that
  resume needs a rollout contract exposing turns, which `run(rendered, task)`
  does not. The `duration_scheduling` example no longer claims to checkpoint.
- **Git failures were opaque.** `capture_output=True` swallowed stderr, so any
  git problem read only "returned non-zero exit status 128"; a new `GitError`
  carries git's own message (missing repo, held index lock, ...).
- **Trust-region rejects vanished.** Over-large diffs were filtered out before
  `considered` was computed and were never settled back into the evidence pool,
  so a diff dropped for size left no trace in the report or the pool.
- **Two documented locks were not locks.** `L1SerialGate.try_acquire` — described
  as "a global L1 lock" — was a check-then-act on a plain dict, so under
  contention several threads could each believe they held it (a 16-thread race
  now confirms exactly one winner); `ResumeQueue.push/pop` was unguarded while
  every async worker pushes into it.
- **Resuming a run silently discarded `initial_state`.** Re-using `repo_path`
  continues an interrupted run (the ledger is a real git repo) — useful when a
  multi-hour run dies — but a supplied `initial_state` was dropped without a
  word. It now warns, and resume is documented and tested rather than being an
  undocumented side effect.
- **Async shutdown overshot the budget in proportion to `n_workers`** — each
  worker was joined for 2s and the merger for 10s, so a 1s budget could return
  16s later. The joins now share one bounded `shutdown_grace`, and abandoning an
  in-flight rollout is reported rather than silent. `max_seconds` is documented
  precisely: it bounds the production phase, after which one held-out scoring
  pass still runs (memoised, so free when the head was already scored).
- **`Task` was unhashable** despite being a frozen dataclass — the generated
  `__hash__` hit the mutable `meta` dict, so `set(tasks)` and `{task: ...}` raised
  `TypeError`. It now hashes and compares on `id`, which the engine already
  requires to be unique.
- **Conflict resolution could leave contradicting diffs in the accepted set,
  silently disabling fusion.** Resolution stopped at the first conflict, so a diff
  that displaced one survivor was never re-checked against the rest; two
  contradicting cards then survived together, which made the tournament's
  "no contradictions" guard false and skipped building the fused candidate —
  losing the model-soup benefit the aggregator exists for. It now resolves to a
  fixed point.
- **`oracle_budget` capped nothing.** The budget was decremented but the oracle
  evaluation ran regardless, so a cost-control knob controlled no cost — on an LLM
  workload each call is a full held-out sweep. It now falls back to the cheap
  verifier layer once exhausted.
- **The audit queue was unbounded and quadratic.** `submit()` re-sorted the whole
  list every call and nothing drains the queue; 28k submits now take 0.1s instead
  of growing without limit, the queue is capped, and it is lock-guarded because
  worker threads submit into it.
- **`AsyncAgentDescent`'s threads were non-daemon**, so a run that overran
  `max_seconds` blocked interpreter exit until the rollouts finished.
- **A typo in the caller's `run`/`propose` produced a clean-looking empty result.**
  The round body's catch-all treated programming errors as backend failures, so a
  signature mistake returned `final_reward=0.0` with zero rounds and no output at
  the default `verbose=False`. Actor signatures are now bound-checked before the
  first rollout, and any run that ends early emits a `RuntimeWarning`.
- **A dead backend could still raise out of synchronous `evolve()`** during the
  final held-out scoring, discarding everything already committed.
- **`Ledger` CAS could be bypassed.** `base_version.get(aid, head)` defaulted a
  missing entry to the current head, so a writer that declared no base version
  (or an unrelated one) always committed — the exact lost update the
  compare-and-swap exists to prevent. Both `commit` and `commit_atomic` now
  require the vector to declare every artifact they write.
- **`async_evolve` reported a probability as the held-out reward.** A caching
  optimisation put `MergeReport.prob_improve` (a Beta-posterior P(delta>0)) into
  `RoundInfo.held_out_reward`, so `history` was fiction and `target_reward` could
  fire on a probability. Re-scoring is memoised, so the optimisation saved nothing.
- **Seeded runs were not reproducible across processes.** Builtin `hash()` of
  `str` is randomised per process, and it seeded worker RNGs, assigned tensor
  sections, bucketed clusters and staggered refreshes — so `seed=` was meaningless
  even in the deterministic synchronous orchestrator. Added `stable_hash`.
- **Async backend failures were silent and fatal.** One exception in one worker
  called `stop.set()` and ended the whole run with no message, and the final
  held-out scoring (plus the merger loop) could raise straight out of the driver,
  discarding committed work. Failures are now retried with backoff, a worker
  retires only after 3 consecutive errors, the run ends when all workers retire,
  and the cause is reported via the new `EvolutionResult.error`.
- **`self_verify=False` was silently ignored by synchronous `evolve()`** — the
  extra verification rollout ran anyway, quietly doubling the LLM cost of every
  proposal. It is now honoured on both paths.
- **`max_seconds=` was silently ignored by synchronous `evolve()`** — a sync run
  had no wall-clock bound at all. It is now enforced; the default is `None`
  (unbounded) so existing runs are unaffected, and the async default is unchanged.
- **Scratch ledgers leaked.** Every `evolve()` call without `repo_path` created a
  temp git repo that was never reclaimed (133 had accumulated in `$TMPDIR` during
  development); they are now cleaned up at exit.
- Input validation: `n_workers=0` raised `ZeroDivisionError` deep in the async
  sharding, duplicate task ids silently collapsed tasks, and out-of-range
  `held_out_frac` / `blast_radius` or an `artifact_id` containing a path separator
  were accepted and failed later. All now raise `ValueError` immediately.
- Bare `pytest` (fresh clone / CI) failed with `ModuleNotFoundError` because the
  repo root was not on `sys.path`; set `pythonpath = ["."]` in the pytest config.

## [0.1.0] — 2026-07-26

First public release on PyPI as **`agentdescent`**.

### Changed
- **Renamed the project Concordia → AgentDescent** — package `concordia/` →
  `agentdescent/`, classes `Concordia`/`AsyncConcordia` →
  `AgentDescent`/`AsyncAgentDescent`, and all docs, URLs, and branding.

### Added
- **EvoSkill**: batch-level failure-driven induction (one skill per batch, shared
  across workers), a sync per-candidate frontier (`TopKFrontierAggregator`) and an
  async SGD-style optimizer (`SgdSkillAggregator`: apply → validate every
  `val_every` steps → roll back on no held-out gain), plus an `eval_at_end` mode.
- Async engine: `self_verify` flag (skip the per-trajectory re-run for held-out-only
  ports) and a cold-start pending-intake throttle so the lag budget bounds
  un-merged work before the first commit.
- Faithful, offline-tested ports of ACE, GEPA, EvoSkill, SkillOpt, ADAS, and DGM,
  each loading its benchmark through the shared `agentdescent.dataloader`.
- The general `evolve()` / `async_evolve()` engine, git-backed `Ledger`, the
  discrete-space `Aggregator`, staleness policies, DP/TP/PP parallelism, layered
  governance, and the provider-agnostic `agentdescent.agents` completion layer.

[Unreleased]: https://github.com/Birfy/agentdescent/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/Birfy/agentdescent/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Birfy/agentdescent/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Birfy/agentdescent/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Birfy/agentdescent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Birfy/agentdescent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Birfy/agentdescent/releases/tag/v0.1.0
