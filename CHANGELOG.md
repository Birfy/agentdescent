# Changelog

All notable changes to AgentDescent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
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

[Unreleased]: https://github.com/Birfy/agentdescent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Birfy/agentdescent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Birfy/agentdescent/releases/tag/v0.1.0
