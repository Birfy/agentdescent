# Changelog

All notable changes to AgentDescent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Task samplers (`agentdescent.sampling`)** — `evolve(task_sampler=...)`. A rollout
  is the expensive unit of work, and spending it on a task the agent already solves
  teaches nothing. `DifficultyWeighted` prefers tasks whose pass rate sits away from
  the all-pass / all-fail extremes (the zero-advantage filter), landing ~1.6-2.2x
  more rollouts on informative tasks than the `RoundRobin` default in measurement.
- `EvolutionResult.save(path)` / `.load(path)` — persist the evolved artifact and
  its run summary as JSON instead of hand-rolling the same serialisation.
- `agentdescent.dataloader` / `agentdescent.backends` are now importable from the
  package namespace (`Dataset` and `split_dataset` are re-exported).
- `tests` CI workflow — runs the offline suite on push/PR across Python 3.9 / 3.11 / 3.12.
- Test coverage for the async SGD path: `SgdSkillAggregator` keep/rollback,
  `eval_at_end`, batch-level propose, and the sync frontier gate.
- PyPI Trusted-Publishing workflow (`publish.yml`): OIDC release, no stored token.
- `EvolutionResult.error` — `None` on a clean run, otherwise the backend failure
  that ended it early, so callers can tell "converged" from "died".

### Fixed
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

[Unreleased]: https://github.com/Birfy/agentdescent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Birfy/agentdescent/releases/tag/v0.1.0
