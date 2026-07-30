# Changelog

All notable changes to AgentDescent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/Birfy/agentdescent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Birfy/agentdescent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Birfy/agentdescent/releases/tag/v0.1.0
