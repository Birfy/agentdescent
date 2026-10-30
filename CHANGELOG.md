# Changelog

All notable changes to AgentDescent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.8.0] — 2026-10-30

### Added

- `agentdescent install grok-bot` / `uninstall grok-bot` scaffold a skill +
  `HOST.md` + `mcp.example.json` under `~/.agentdescent/plugins/grok-bot/`
  (#165). Grok Bot has no public one-click plugin API for local skill+stdio
  MCP yet, so `doctor` reports `mcp n/a` when the scaffold is current rather
  than pretending a host config was wired. Docs: `plugins.md`,
  `plugin-quickstart.md`.

- `agentdescent install opencode` writes `/agentdescent/evolve` under
  `~/.config/opencode/command/agentdescent/evolve.md`, and `install codex` writes
  `/prompts:evolve` as `~/.codex/prompts/evolve.md` — the same prompt body as
  Claude Code's `/agentdescent:evolve`. `uninstall` removes each only while still
  stock (#165).

- `agentdescent install` writes an `/algorithms` slash entry beside `/evolve` on
  Claude Code (`commands/algorithms.md` → `/agentdescent:algorithms`), OpenCode
  (`command/agentdescent/algorithms.md` → `/agentdescent/algorithms`), and Codex
  (`prompts/algorithms.md` → `/prompts:algorithms`). The body lists, shows, or
  runs a named method through the same cards as `agentdescent algorithms
  list|show` and MCP `list_algorithms` / `describe_algorithm`. Grok Bot's
  `HOST.md` names the CLI/MCP discovery path; `uninstall` removes each slash
  file only while still stock (#384).

- `rsi_report` / `agentdescent rsi` / MCP `rsi`: a read-only longitudinal view of
  the consolidation archive. Per decision slot it reports the first and best
  adopted reward, the adoption that set the best, the gain, adoptions since that
  best, and the exam counts, and returns an overall `improving` / `stagnant` /
  `mixed` / `insufficient` verdict with the current portfolio reallocation.

- `run_tournament` / `agentdescent tournament`: run the same spec under several
  registered algorithms (and seeds) with the same evolve budget, then report the
  measured ranking and winner. The winner is aggregated per algorithm over seeds
  (mean reward, fewer mean calls on a tie); a tie is reported as a tie. An errored seed counts as
  reward 0 and is reported under `errors`, so a crash cannot win by shrinking the
  denominator; `--dry-run` prices every arm before a run. `default` is the spec's own recipe; each named arm
  gets a deep copy, so the spec file is untouched. Ordinary
  `EvolveSpec -> compose() -> evolve()` per arm, not a second engine or a claim to
  reproduce the ports' published numbers.

- `run_campaign` / `agentdescent campaign`: run several evolution sessions on one
  spec, each starting from the previous session's artifact. The campaign works on
  a copy in a temporary workspace, applies only improving sessions, stops after
  `patience` non-improving rounds, and reports the best artifact, workspace and
  `stopped_reason`. Session 0 is the baseline (not a claimed improvement), the
  split seed is fixed across sessions so rewards are comparable, and `final_reward`
  is the last session while `best_reward` is the artifact kept. `--dry-run` prices
  the sessions first; `policy_slot` and inline `text` targets are refused because
  there is no file to carry, and `--workdir` must be empty.

- MCP tools `safety`, `portfolio` and `audit_trace`: a host agent can now ask
  whether an archive is safe to seed from, how its layers rank for the next
  budget window, and where one adoption's evidence came from, without finding the
  CLI verbs in the source. They return the same structure as the CLI's `--json`
  (which now also carries `archive` for `safety`), write nothing, and treat a
  missing archive as an error. The installed `SKILL.md` says when to use them, and
  the `doctor` tool description now lists `opencode` (#505).

- `LeakageGuard`, a pre-evaluation screen and acceptance backstop that rejects a
  candidate introducing a protected gold answer, task id or corpus string. It
  checks diff keys and values before the held-out sweep and reports the existing
  `critic-rejected` outcome, so a leak never crashes the aggregator or buys an
  inflated score. The acceptance form only refuses newly introduced matches, so a
  protected string already in the seed does not freeze the run. Matching is
  case-insensitive and whitespace-collapsed, and short strings are ignored.

- `ApprovalGate` + `ApprovalStore`: high-risk self-modifications (blast radius or
  diff size above a threshold) are not committed; a request is appended to a JSONL
  store, and a human approval lets the same **content** through while a rejection
  stays sticky. The key fingerprints target, ops and base version instead of the
  producer's `diff_id`, `needs-approval` / `approval-rejected` are real
  `MergeOutcome` members, requests carry a reviewable summary, and malformed store
  lines are skipped. Low-risk candidates delegate to the inner acceptance policy.

- `MarginalGainStop`, a stateful `stop_when` policy: it watches the recent
  reward/call slope and stops after `patience` measurements whose per-call gain
  is below `min_gain_per_call`. The per-call bar decides on its own under the
  defaults, so a run that improves too slowly to pay for its calls now stops;
  `min_gain` is an optional absolute-gain floor combined by OR. The `reason`
  names the measured gain and call delta, a round with no new calls never counts
  as low, and `reset()` makes the object reusable across runs. This is an
  economic stop for long self-improvement runs, not a safety verdict.

- `VarianceSampler`, a `TaskSampler` that prefers tasks whose recent reward
  variance is high (flip-flopping tasks) over a lifetime pass-rate average, with a
  UCB exploration bonus and round-index tie rotation. It is a drop-in
  `task_sampler=` argument, thread-safe like `DifficultyWeighted`.

- `diagnose_payload` / `agentdescent diagnose` / MCP `diagnose`: turns `show`'s
  samples, rejected candidates and outcome histogram into one actionable verdict
  (`solved`, `partial-improvement`, `running`, `cancelled`, `run-failed`,
  `reflector-error`, `scoring-mismatch`, `gate-rejected`, `agent-failures`,
  `no-improvement`, `unknown`) with bounded evidence and the next step. A
  committed-but-partial run is no longer called "no candidate adopted", gate
  rejections win over the worst-sample rule, truncated samples do not claim a
  scoring mismatch, and running/cancelled states are named. It reuses the same
  read model as the CLI/MCP without building a discarded diff.

- `GuardPipeline`, one ordered acceptance gate that composes the checks a caller
  supplies and names the ring that refused a candidate. It keeps the inner
  guard's merge category (falling back to `below-threshold`) instead of inventing
  a category the aggregator cannot decode, reports the first refusal even with
  `collect_all`, and forwards `bind`/`configure` plus checkpoint, restore and
  screen-reject hooks to every guard. It imports no specific guard, so third-party
  checks are first-class.

- `DecisionRecord` + `DecisionLog`: an append-only JSONL of immutable adoption and
  session decisions with the guard category, evidence, cost and parent link.
  Malformed lines (including non-object JSON) are skipped, a torn last line is
  closed before the next append, duplicate record ids are refused, non-finite
  floats become null, and `children()` exposes the decision tree a dashboard or
  audit export needs. Writers in one process share a per-path lock.

- `bootstrap_ci` / `is_significant` / `paired_gain` and `contamination_overlap`:
  seeded paired-task bootstrap confidence intervals with an effect-size test, and
  a contamination report between train and validation text. Paired keys are
  sorted so the same seed is reproducible across processes, non-finite scores are
  refused, normalized exact duplicates and CJK/short text are screened, and
  "nothing to screen" is `None` rather than a clean-looking `0.0`. Pure
  standard-library functions for adoption gates, reports and external audits.

- `KillSwitch`, `canary` and `rollback_plan`: a file-based global stop, a paired
  exact sign test that recommends promote/hold/rollback from incumbent vs candidate
  per-task scores, and a pure state diff for rollback. Missing candidate tasks
  block promotion and are reported, significant losses use a separate
  `max_regression` bar, a delete-only rollback is not a no-op, and the switch
  treats a directory at its path as engaged and writes atomically. These are
  deployment operations separate from the acceptance gate, so a committed change
  can still be canaried and reverted deliberately.

- `run_guarded`, one entry point for a session that cannot skip the kill switch or
  the decision record. The injected acceptance is chained **before** the spec's
  own acceptance policy or the shipped default statistical gate, so guards can
  only tighten adoption; stateful checkpoint/restore/screen-reject hooks are
  forwarded. The decision record says whether a candidate actually committed and
  carries the error separately, a failing log does not hide the result, and an
  `aggregator_factory` algorithm is called out in `notes`. The kill switch is an
  explicit pre-flight check.

- `RunManifest` / `build_manifest` / `verify_manifest` / `diff_manifest`: a
  content-addressed manifest of a run's spec, data files or directory tree, model
  configuration (secret keys redacted), seeds, algorithm and code version. The
  manifest id is stable for identical inputs, and verification names the fields
  that changed.

- `RiskPolicy` / `RiskRule` / `RiskContext`: ordered, declarative rules that map a
  change's tier, blast radius, diff size, capability domain and tenant to
  `allow`/`approve`/`canary`/`rollback`/`deny`. First match wins, no match is
  fail-closed (`deny` by default), and invalid actions or duplicate rule names are
  refused at construction.

- `evaluate` / `EvaluationReport`: a deterministic, budgeted verifier-vs-oracle
  pass. The verifier scores every task, the oracle only a seeded sample capped by
  `budget`, and the report carries means, signed/absolute bias, coverage, oracle
  calls and per-task rows. Zero coverage is explicitly "no truth evidence", not a
  pass.

- `run_controller` / `ControllerConfig` / `SessionOutcome`: a generic multi-session
  loop with max sessions, call budget, kill switch, patience/stagnation and an
  optional decision record per session. The caller supplies `run_session(index,
  seed)`, so a guarded run, campaign step, population generation or remote job can
  sit behind the same scheduling and stop policy.

- `metrics_from_records` / `metrics_from_log`, `to_prometheus` and `alerts`: RSI
  aggregates from decision records -- acceptance rate, calls/tokens/wall clock,
  calls per accepted decision and category frequency -- plus a deterministic
  Prometheus text export and pure threshold alerts. The export gives the
  per-category counter its own family, escapes label values and rejects malformed
  records instead of silently coercing them. No model calls and no second source
  of truth.

- `docs/algorithm-authoring.md` plus `agentdescent/algorithms/_TEMPLATE.py`: the short contributor path to register one named Algorithm card in a single PR (`algorithms list` / `show` / `plan`), with links from the generated catalog, plugins docs, and the porting checklists (#539).

- The MCP `apply` tool now accepts optional `from` + `to` (mutually exclusive with
  `run_id`, same rules as CLI `apply --from`: `from` requires `to`, symlinks
  refused, dry-run / backup via `apply_from_payload`), so an MCP host can install
  a campaign `final_target` without shelling out (#540).

- `QualityDiversityArchive`, a MAP-Elites-style archive that keeps one elite per
  behaviour cell and reports coverage and QD-score instead of collapsing the
  search to a single best artifact.

- `leave_one_out_credit` and `shapley_credit` attribute a candidate's gain to
  individual edits, with a shared subset cache and an explicit interaction term so
  a reflector can learn which change actually paid.

- `ConformalGate`, a split-conformal lower prediction bound for one new task's
  paired improvement. It uses the finite-sample floor rank, returns `-inf` (and
  `insufficient`) when the calibration set cannot certify `alpha`, and states the
  exchangeability/per-task assumption instead of claiming a candidate-level
  false-acceptance rate.

- `agentdescent mcp --transport streamable-http [--host 127.0.0.1] [--port 8000]`
  serves the MCP tools over HTTP at `/mcp`, so URL-only hosts such as Grok Bot can
  connect through an HTTPS tunnel; stdio stays the default. A non-loopback
  `--host` is refused unless `--allow-remote` is passed. The Grok Bot `HOST.md`
  and plugin docs now give the copy-paste commands (#560).

- `EProcess` / `sequential_test`, an anytime-valid hedged-betting e-process for
  paired improvement deltas: a dashboard can watch the same evidence continuously
  without inflating the false-positive rate.

- `detect_goodhart` and `GoodhartMonitor` measure proxy-oracle inflation, rank
  tracking, and concrete shortcut candidates, so a self-improving loop can alarm
  when the reward it optimises stops tracking the reward it cares about.

- `MetamorphicSuite`, `MetamorphicRelation` and `invariance` check output
  relations under input transforms, so a candidate that scores higher by learning
  surface form can be caught by a broken invariance. The suite is an acceptance
  probe, not yet wired into the evolution loop; `accept(min_rate=...)` supports a
  bounded failure rate.

- `SelfModel`, a deterministic **logit-space Newton/IRLS** Platt calibrator with
  ECE/Brier reporting and calibrated answer/abstain routing. It refuses to predict
  before `fit`, reports anti-correlation instead of hiding it, marks tiny samples
  `insufficient`, and converges on saturated 0/1 confidences where fixed-step
  gradient descent diverges.

- `dedupe_texts` with Unicode-aware SimHash and MinHash fingerprints, clustering
  near-duplicate artifacts deterministically. CJK runs contribute character
  trigrams and punctuation-only text falls back to characters, so distinct rules
  do not collapse; it is an archive-hygiene utility, not yet wired into the
  population/QD path.

- `LineageGraph` and `lineage_report` for artifact provenance: ancestry,
  descendants, shortest derivation path, impact size, roots/leaves, topological
  order, external-parent accounting, and cycle detection. Walks are iterative and
  child adjacency is indexed, so long self-improvement lineages stay usable.

- `fit_power_law` / `plan_budget` fit a deterministic saturating compute curve
  (`a - b * budget^-c`) with grid + local refinement, report R² and RMSE, and
  invert it to the smallest budget that reaches a target. Unreliable fits are
  refused by default and targets already observed are not extrapolated.

- `RSIController`, a duck-typed control plane that turns each session result into
  `accept / reject / defer / rollback / continue / stop` using target, cost,
  patience, risk, pluggable gates and a budget policy. It wraps the existing
  guarded/controller/portfolio pieces instead of replacing them.

- `TaskFoundry`, a failure-seeded task assembly line: generate, validate,
  deduplicate (exact + optional near-duplicate), filter contamination and
  unlearnable tasks, and split accepted tasks deterministically into train and
  holdout. `failure_seeds` mines low-reward trajectory rows as seeds.

- `ArtifactGraph`, a Unicode-aware TF-IDF retrieval graph over prompts, skills and
  harness fragments with kind/tag filters, trust weighting, parent/child edges,
  and budgeted composition into a task prompt.

- `VerifierEnsemble`, a cost-aware verifier cascade with weighted/majority/unanimous
  strategies, early stopping on confident agreement, a hard cost cap, fail-closed
  incomplete panels, explicit-fail vetoes, and disagreement/error reporting.

- `ReplayLab`, a deterministic counterfactual replay layer over a session trace:
  remove/replace/disable interventions, leave-one-out effects, and Shapley
  attribution through `credit`. The cache is value/metadata-based (or caller-keyed,
  or disabled for unhashable payloads), and multi-step results carry the real
  intervention tuple.

- `DeferralPolicy`, a learned confidence threshold for `auto-approve / defer /
  deny`: it fits from past reviewer outcomes under a target auto-approve risk,
  honours a hard-risk deny ceiling, refuses small calibration sets, and makes the
  defer-all fallback genuinely defer everything (including confidence 1.0).

- `SocietyRuntime`, a role-based blackboard loop for multi-agent RSI:
  proposer/critic/verifier/historian roles, per-role budgets, consensus or target
  stopping, cost accounting, participation-based role credit, critic vetoes, and
  optional Shapley attribution through `credit`.

- `SafeSelfModifier`, a capability gate plus canary/rollback promotion path for
  self-modifying code: allow-listed imports, forbidden name/attribute references
  (not only calls), inferred files/network/process capabilities, risk tiers, and a
  canary measurement before promotion. Static screening is defense-in-depth; a
  sandbox is still required for execution.

- `RSITelemetry`, an OTel-inspired span/event tree for RSI runs
  (`run → round → rollout → merge → gate → commit`) with unique IDs, attributes,
  events, exception/cancellation status, wall vs span time, escaped Prometheus
  samples, and OTel-shaped dictionaries (not OTLP JSON).

- `bench.rsi_bench`, a long-horizon RSI benchmark: hash-split train/holdout with a
  generation-outer loop, transfer, mean/max/forgetting-fraction, proxy-vs-oracle
  **gain** gap, error counts, cost cap, generations-to-target, family breakdown,
  and `improving / flat / regressing / no-transfer / forgetting / hacking /
  insufficient` verdicts.

- `DurableRunService` + `SqliteRunStore`: an event-sourced, reopenable run log
  with leases/heartbeats, idempotent event keys, retries, exactly-once completion,
  cancel, and expired-lease recovery for long RSI sessions.

- `WorkspaceRegistry`: tenant/workspace principals, role-based permissions
  (viewer/operator/approver/admin/owner), token/cost/concurrency quotas with
  reservations, and append-only audit export as JSONL or dicts.

- `PolicyBundle` + `AdmissionController` + signed `ArtifactManifest`: deny-overrides
  policy evaluation, approval obligations, canonical manifest hashes, HMAC-SHA256
  detached signatures (demo signer; swap for cosign/Sigstore), and fail-closed
  admission when the signature is missing or invalid.

- `EvalOps`: versioned dataset registry with content hashes, eval specs, paired
  bootstrap results, CI regression gates, drift monitoring, scheduled reruns, and
  a serialisable operations snapshot.

- `OperationsConsole`, a read-model API for runs, approval queues, QD coverage,
  lineage, and cost: filtered/paginated run lists, risk-sorted pending approvals,
  terminal decisions, and a serialisable dashboard snapshot.

- `DataFlywheel`: turn production traces into a content-hashed dataset candidate
  with reward/error filtering, deterministic sampling, recursive secret/PII
  redaction, exact/near-duplicate clustering and a per-cluster cap.

- `ModelRouter`: a cost-aware Thompson-sampling router over model quality and
  price, with context/tag/budget filters, epsilon exploration, posterior updates
  and a per-model routing report.

- `PreferenceModel` + `PreferenceBuffer`: a Bradley-Terry reward model over
  pairwise preferences with tie targets, deterministic IRLS, accuracy/log-loss,
  and an uncertainty/distance-based next-query selector for active feedback.

- `ContinualMetaLearner`: online softmax meta-policy updates with a bounded replay
  buffer, validation-gated promotion, rollback on regression, and a feature-drift
  signal for when the validation context stops matching the replay context.

- `open_ended_coevolve`: novelty-and-difficulty task/solver co-evolution with a
  bounded archive, a target-difficulty frontier, deterministic generation and a
  best-task/coverage report.

- `ProcessRewardModel`: a calibrated step-level logistic reward model with IRLS,
  accuracy/log-loss/ECE/Brier reporting, and candidate reranking for PRM-guided
  credit and search.

- `MCTSPlanner`: UCT planning over a caller-supplied simulator and value function,
  with a hard simulation budget, deterministic seed, terminal transitions,
  cost penalty, and a PlanResult carrying the chosen action/value/visits.

- `ToolSynthesizer`: generate/validate/select tool candidates with a recorded
  rejection trail, capability metadata, custom selector, and an `EnvironmentSpec`
  for grouping validated tools.

- `Deliberation`: structured multi-agent debate with evidence-bearing arguments,
  confidence/weighted judge votes, agreement tracking, and a consensus/round-budget
  stopping rule.

- `ProofRunner` + `ProofCertificate`: execute an invariant suite, record
  warning/error obligations, hash the artifact, and emit/verify a serialisable
  proof-carrying certificate for manifests, lineage and policy admission.

- `LoopKernel`: a resumable RSI loop over `durable_runs` that composes the
  existing seams in one order — `step -> RSIController -> apply/rollback` — and
  replays completed generations after a crash. Each generation carries its
  observation, control action, gain, cost and optional decision-log/telemetry
  ids; the final generation is durable instead of being lost in the completion
  payload.

- `promote()` + `PromotionRules`: one fail-closed promotion decision over
  evaluation, contamination, conformal/sequential evidence, canary, proof,
  admission, risk and human approval. Missing evidence returns `defer`, a failed
  gate returns `reject`, a canary regression returns `rollback`, and only a fully
  green bundle returns `promote` with a stable evidence hash for manifest/lineage.

- `ComputeController` + `ComputeOption`: an action-level Thompson-sampling
  allocator over direct / PRM-rerank / MCTS / verifier-ensemble strategies. It
  composes `model_router` for the concrete model, respects difficulty bands,
  context/tag/cost constraints and an optional call budget, and records the
  chosen option, model, expected quality/cost and exploration flag for the
  decision log.

- `CapabilityFoundry`: the admission loop for a synthesized tool. It binds
  validation score, debate/review agreement, `safe_selfmod` capability/canary
  checks, proof certificate verification, holdout gain and artifact-graph
  registration into one fail-closed outcome (`promoted` / `deferred` / `rejected`
  / `rolled_back`). Missing evidence defers; failed gates reject; a measured
  canary regression rolls back.

- `ExperienceStore` + `ExperienceEngine`: append-only cross-run experience
  memory with nearest-neighbour recall and a continual meta-policy update guarded
  by a frozen anchor slice. The engine warms its learner replay from an existing
  store on restart; a candidate that improves validation but regresses the
  anchors is rolled back, and `ContinualMetaLearner.rollback()` restores the
  weights while keeping the attempt in history as a rejected update.

- `examples/rsi_loop`: one offline run that composes the loop kernel, adaptive
  compute, trusted promotion, capability foundry and continual experience, with a
  test asserting each layer's decision. It is the end-to-end demonstration of the
  five integration features rather than another primitive.

### Changed

- `agentdescent doctor` prints `skill` and `mcp` on each installed host line
  (for example `partial (skill ok, mcp missing; …)`), so stale skill vs missing
  MCP is visible without reading the problems section (#165).

- CI's test matrix runs on `ubuntu-24.04` instead of `ubuntu-latest`: the
  Ubuntu 26.04 runner image has no Python 3.9 for `setup-python`, so the required
  `test (3.9)` check could never pass, and the sandboxed AlgoTune timing test was
  failing there too.

- `LoopKernel`, `promote` / `PromotionRules` and `ComputeController` are now
  re-exported from the top-level `agentdescent` package, listed in the generated
  API reference and documented with the loop, promotion and compute guides. The
  modules stay usable through their submodules as before. An explicit
  `approved=False` now rejects even when `require_approval` is off, matching the
  documented fail-closed rule.

- `CapabilityFoundry`, `ExperienceEngine` / `ExperienceStore` and the
  `ContinualMetaLearner` types are now re-exported from top-level
  `agentdescent`, listed in the generated API reference, and documented with the
  capability-foundry and continual-experience guides. The modules remain usable
  through their submodules as before.

### Fixed

- `agentdescent init` with an `opencode` worker and both `--model` and
  `--base-url` now writes the OpenCode Coding recipe into `agent` (`-m
  <provider>/<model>` plus `env.OPENCODE_CONFIG` naming a provider-only
  `<spec>.opencode-providers.json` beside the spec, key as `{env:VAR}`, never
  inline) instead of a bare `{"ref": "opencode"}` that ignored both flags.
  Without both flags the bare ref and its warning are unchanged (#485).

- `agentdescent portfolio`, `exam` and `audit` no longer create a missing archive
  directory and report it as empty: like `safety`, they now fail with
  `error: archive '<path>' does not exist` (exit 2). `audit` likewise rejects a
  `--meta-audit` / `--inner-audit` file that does not exist (#502).

- `gepa`, `skillopt` and `dgm` specs no longer get the default reflective
  merge pair their own aggregator ignores, so `evolve` stops emitting a
  `RuntimeWarning` about `conflict`/`fusion` policies nobody wrote; a spec that
  names those policies explicitly gets the warning from `plan` instead (#504).

- `python -m agentdescent` now runs the CLI, the same as the `agentdescent`
  console script, instead of failing with "No module named
  agentdescent.__main__" (#510).

- Missing-archive errors from `safety` / `portfolio` / `audit` (CLI and MCP) now
  name `continuous_meta_run` and say that `evolve` / plugin runs do not write an
  archive, so a host agent does not invent a nonexistent "consolidation" step.
  Those read-only verbs also leave an existing empty archive untouched (no
  `runs/` directory as a side effect) (#512).

- `--json` now works after the subcommand as well as before it, for every verb:
  `agentdescent show --json <id>`, `doctor --json`, `status --json`,
  `plan --json <spec>`, `portfolio --json` and `audit <id> --json` used to exit 2
  with "unrecognized arguments", even though the docs write them that way.

- `agentdescent install opencode` no longer leaves OpenCode without its MCP
  server when `~/.config/opencode/opencode.jsonc` has comments: the commented file
  is still never rewritten, and the `mcp.agentdescent` entry goes into
  `opencode.json` beside it, which OpenCode merges. If neither file can be parsed,
  install prints a `FAILED:` line and exits 1, and `install` / `uninstall` now
  honour `--json`. `doctor` no longer reports such a host as `ok` when the
  commented file does not even name `agentdescent` (a commented-out mention no longer counts), and `uninstall opencode`
  cleans up the `opencode.json` entry too.

- `agentdescent campaign` results can now be installed: `agentdescent apply --from
  <final_target> --to <target> [--dry-run]` applies an on-disk artifact with the
  same `.bak-N` backup and no-op check as `apply <run_id>`, and the campaign's
  output ends with the ready-to-copy diff / dry-run / apply / cleanup commands
  (`next` in `--json`). `docs/campaign.md` no longer points at an `apply --to`
  form that failed with "has no result to apply yet".

  Symlinks inside a ``--from`` directory (or a ``--from`` path that is itself a
  symlink) are refused rather than dereferenced into real files under ``--to``,
  matching ``apply <run_id>`` which never materialises links.

- `campaign` now stops as soon as a session reaches the spec's `target_reward`
  and reports `stopped_reason: "target_reward"`, instead of spending another
  session and then reporting `patience` (#534).

- `agentdescent doctor` no longer labels a host CLI it found on `PATH` as
  `missing`: an installed but unwired CLI now reads `not installed (cli ...)`
  with the `agentdescent install` command, a host without a CLI reads `no cli
  on PATH`, and the closing line says `ok: no blocking problems`, naming how
  many hosts can be wired. The `--json` report is unchanged (#535).

- A bare `agentdescent algorithms` (and `agentdescent algorithms --json`) now lists
  the algorithms like `algorithms list` instead of exiting with an argparse
  "arguments are required: action" error (#546).

- `agentdescent algorithms --kind K` now filters like `algorithms list --kind K`
  (with or without `--json`) instead of exiting 2 with a misleading
  `invalid choice` error, and `--kind` with `algorithms show` is refused with a
  clear message rather than ignored (#549).

- `agentdescent plan` without `--json` no longer prints a `{"ok": false, ...}`
  JSON object when the spec fails to compile (bad `--algorithm` / `--param`, bad
  data rows): it prints `error: <message>` on stderr and exits 2, like an
  unreadable spec file. `--json` output is unchanged (#551).

- `agentdescent plan` without `--json` no longer dumps the `{"ok": false, ...}`
  JSON object when the spec compiles but fails a hard check (too few tasks,
  missing CLI binary, unfilled placeholder): it prints `error: <message>` on
  stderr and exits 2, like a spec that fails to compile. `--json` output is
  unchanged (#553).

- `agentdescent install` no longer reports `updated` for files whose contents are
  already identical; it prints `kept <path> (unchanged)` and leaves the file
  untouched (#555).

- `agentdescent init --data` now reads the cases file's first row and, when it has
  no `prompt`/`gold` column, maps a single common alias (`input`/`question`/`query`/`problem`,
  `expected`/`answer`/`output`/`target`) and says so, or warns with the real column
  names. Spec loading now names the `data.prompt` field in a missing-column error,
  and `plan`/`evolve`/MCP refuse a gold-comparing scorer (`contains`, `exact`,
  `last_number`, `numeric_close`) when no row has the `data.gold` column instead of
  silently scoring every answer 0.

- `agentdescent mcp --transport streamable-http --allow-remote` now serves
  non-loopback clients on mcp 1.x too: the SDK pins allowed Host headers to
  loopback when the server is built, so a remote bind answered every request 421;
  the pin is now dropped for a non-loopback host (#587).

- `agentdescent mcp --transport streamable-http` on mcp 1.2–1.7 now exits with a
  one-line "needs mcp>=1.8" message instead of the SDK's opaque Unknown transport
  error; CLI host/port defaults stay locked to `mcp.DEFAULT_HTTP_*`, and the
  `.mcp` import stays lazy so a shadowed cwd still prints one line (#589).

- `agent_society._is_literal` no longer references deprecated `ast.NameConstant`
  (removed in Python 3.14); `True`/`False`/`None` are already `ast.Constant` on
  the supported 3.9+ range (#591).

- `agent_society._is_literal` now accepts unary `+` / `-` on numeric constants
  (`-1`, `+2`, nested in dict/list), matching `ast.literal_eval`, so the society
  gate no longer reports a signed number as "computed code" (#593).

- The `evoskill` algorithm card now accepts `kind: skill_dir` and `plugin` as well
  as `agent_dir`, so `agentdescent init <skill> --kind skill_dir --algorithm
  evoskill` no longer exits 2; the module docstring, `docs/algo-evoskill.md` and
  the registry now agree (#604). Its `SkillLibraryTree` is now built over the
  kind's own `FileTree` (new `Overrides.strategy_from` hook), so a plugin keeps its
  host's frozen paths and its `.ts`/`.js` code, and a `skill_dir` keeps its
  SKILL.md gate; on a `skill_dir` the induced rule goes to `references/<slug>.md`,
  linked from a managed index block in the root `SKILL.md`, instead of a nested
  skill the host would not load.

- `agentdescent doctor` accepts `--home <dir>` with the same probe semantics as
  `install` / `uninstall` (#605): host skill/MCP status is read under that tree,
  and an explicit `--home` wins over `CODEX_HOME` / `DSH_HOME` /
  `XDG_CONFIG_HOME`. The MCP `doctor` tool takes the same optional `home`
  argument.

- `diagnose` and the guarded-run decision log treat named-algorithm adopt
  categories (`gepa-select`, `skillopt-accept`, `dgm-select`) and per-round
  `committed` totals as adoptions, so a successful GEPA/SkillOpt/DGM run is
  no longer reported as "no candidate was adopted" (#617).

## [0.7.0] — 2026-10-23

### Added

- A persistent-island population run now returns `EvolutionResult.lineages`
  (#150, semantics C): a read-only row per lineage (`dev` plus each island) with
  its candidate count, best archived version/score, selection count and lineage
  tip. It is read from the archive the gate already scored, survives `finalize`,
  and is `None` on a single-head run; the official artifact is unchanged.

- ACE's grow-and-refine playbook is now a named algorithm (`"algorithm": "ace"`,
  `kind: text`) in the in-package registry (#384). `ACEPlaybook`, its
  near-duplicate de-dup and the `GrowAndRefine` Curator moved from `examples/ace`
  into `agentdescent/algorithms/ace.py`; the example keeps its FiNER domain,
  prompts and CLI and imports the one implementation.

- SkillOpt's skill-document training is now a named algorithm
  (`"algorithm": "skillopt"`, `kind: text`) in the in-package registry (#384).
  The bounded edit patch, strict held-out gate, rejected-edit buffer and annealed
  textual learning rate moved from `examples/skillopt` into
  `agentdescent/algorithms/skillopt.py`; the example keeps its SearchQA domain,
  prompts and CLI and imports the one implementation.

- EvoSkill's bounded skill-library frontier is now a named algorithm
  (`"algorithm": "evoskill"`, `kind: agent_dir`) in the in-package
  registry (#384). The two-role Proposer/Generator induction, the top-K frontier
  and the best-member parent rule moved from `examples/evoskill` into
  `agentdescent/algorithms/evoskill.py`; the example keeps its OfficeQA domain,
  prompts and CLI and imports the one implementation.

- Third-party packages can register a named algorithm through the
  `agentdescent.algorithms` entry-point group (`Algorithm` or a zero-argument
  factory returning one). A plugin that fails to load is recorded in
  `agentdescent.algorithms.external_errors()` and does not hide the shipped
  registry; a name that duplicates a built-in is an error, not a shadow.

- DGM's keep-all capability archive is now a named algorithm
  (`"algorithm": "dgm"`, `kind: text`) in the in-package registry (#384). The
  sigmoid×novelty parent draw, the keep-all archive and the one-capability
  proposal moved into `agentdescent/algorithms/dgm.py`; the example keeps its
  coding-agent/Docker domain and imports the mechanism.

- `recursive_self_improve`: the self-improvement loop now carries a
  `ReallocationPolicy`, and each outer round evolves that rule with
  `portfolio_evolve` on the archive the loop just wrote. The first round uses the same fixed rule as `self_improve` (config floor/min_samples),
  each later round searches from and validates against the incumbent, and a rejected
  policy round is reported with the previous rule kept; a cold archive is a reported rejection,
  not a failure. `SelfImproveConfig.min_exam` also applies the verified-seeding bar
  before an archived rule is used as a prior.

- `CapabilityGuard`, an acceptance policy that protects capabilities outside the
  task being evolved: it scores the base artifact and the candidate on named task
  suites and rejects a self-modification that misses a suite's absolute floor or
  falls below the base by more than its tolerance, then delegates to the inner
  acceptance policy. This is the anti-catastrophic-forgetting gate for recursive
  self-improvement rather than a second verifier.

- `VerifierCurriculum` and `VerifierAgreement`: a curriculum that selects the
  tasks where a verifier disagrees most with the oracle (abs / overestimate /
  underestimate), feeds the selected oracle labels back into the verifier, and
  reports the gap before/after (one oracle label per task per round); plus an
  acceptance policy that audits a sample of commits and refuses a verifier-only
  gain or an oracle regression (`oracle-disagrees`). This targets the
  Goodhart failure mode of recursive self-improvement without adding a second
  verifier.

- `JEVModel`, a dependency-free distilled Judge–Evaluator–Verifier student: a
  linear logistic scorer over caller-supplied `(artifact, task)` features that
  learns oracle labels online (`observe`), in passes (`fit`), predicts in `[0, 1]`
  where a verifier is expected, and checkpoints as JSON. Intended to distil oracle labels from a disagreement-selected task loop, not to
  replace the oracle; features are standardised online and the JSON checkpoint
  carries the scaling state.

- `safety_report` / `agentdescent safety`: a read-only pre-flight check for an
  archive that a recursive loop is about to seed from. It combines exam coverage,
  whether each slot's best eligible rule is trusted, and the worst oracle-vs-
  verifier gain bias from an optional `MetaAuditStore` on the *seedable* set, returning `safe` /
  `caution` / `unsafe` with the named checks and reasons. `--min-exam` mirrors
  `verified_rules`; CLI exit codes are 0 safe, 1 caution, 3 unsafe, 2 usage. No
  model calls, no writes.

- `agentdescent uninstall <dsh|claude-code|codex|opencode>` (`--dry-run`, `--home`)
  reverses `install` (#165): it removes the skill files and the plugin directory,
  the marker-delimited `cordis.patch.yml` block, the `[mcp_servers.agentdescent]`
  table and the pinned `mcp.agentdescent` entry, and nothing else. A Codex table
  is removed even when Codex has appended other tables after it. Anything that is
  no longer exactly what `install` writes (an edited launcher, a bare
  `opencode mcp add` entry, a hand-written table) is kept and reported with exit
  status 1, a commented `opencode.jsonc` is never rewritten, and the run store is
  untouched. New `integrations.uninstall()`.

- Genesis (`examples/genesis`): a blind mode in which the acceptance suite lives
  outside the repository and the agents see the requirement and the assertion's
  name rather than its source, `own_review` / `own_test_failures` so a child's own
  tests are run and implementation code with no test under its node is sent back
  (a test file is `test_*.py` or `*_test.py`; package markers, `conftest.py` and
  generated modules owe none), the `fly` domain (only `REQUIREMENTS.md` is frozen),
  a containerised session sandbox that forwards credentials by name rather than by
  value, and the `claude-cli` provider with `--max-tokens` / `--timeout` in
  `examples/_common.py`. `agentdescent/` itself is untouched (#188).

- Multi-artifact evolution — `extra_artifacts=` on `evolve()` and `async_evolve()`
  (#197): register any number of artifacts beside the primary; the worker loop
  proposes against each (PP stages name them; `PipelineParallel` lifts once
  extras exist), the aggregator merges by `diff.target`, and a contract-breaking
  diff lands atomically via `Ledger.commit_atomic` with exact dependent-adapter
  coverage (`missing-adapters` when incomplete) and CAS retry that refuses a
  stale adapter. `Contract.depends_on` drives dependent cache eviction
  (`MemoryCache` / `FileCache.invalidate`). Sync/async rollout records attribute
  extras to the extra's artifact id. Default path without extras is unchanged.

- RRSI wired into `evolve()` (`examples/rrsi/rrsi_policies.py`, #254 / #221):
  `build_rrsi_bundle()` assembles the three regularizers -- `RRsiCritic`
  (pre-evaluation screen: annealed edit-budget gate + deterministic leakage
  screening + optional model reviewer), `RRsiProposalPolicy` (injects budget,
  measured per-edit history, pruning targets and stall exploration into the
  reflection), and `RRsiAcceptance` (the pinned Algorithm-2 floor + both
  cost-rule branches + novelty, with monotone `S*`) -- into one `Policies`
  bundle sharing a checkpointable `RRsiState`. New engine seam
  `Policies(screen=...)` (`ScreenPolicy`) runs **before measurement**, so a
  leaking or over-budget card is settled and reported as `critic-rejected`
  without a held-out sweep; new outcomes `noise-floor` / `cost-rule`; new
  counters `critic_checked` / `critic_rejected`; the aggregator's checkpoint
  now carries an acceptance policy's optional state, so a resumed run does not
  lose `S*`. The selection core and schedule fixtures are reused, not
  rewritten; declared deviations (one-candidate-per-round judging, per-merge
  anneal on the barrier-free path, configured `noise_band`) are recorded in
  `docs/algo-rrsi.md`. Default behaviour unchanged -- a run without the bundle
  reports zeros.

- Verifier bootstrap flywheel (`agentdescent/audit/boost.py`, #273):
  `VerifierBoost` wraps any verifier and learns a monotone calibration curve
  (`isotonic_fit`, numpy-only PAV) from the audit's **IMPROVEMENT** pool, then
  activates it **only if it provably reduces residual error on a held-out slice
  of that pool** (the bootstrap gate). When active it corrects the cheap and
  learned layers the aggregator uses to rank candidates and the audit uses for
  its trust signal; `eval_counts` / `full_eval` are delegated **raw**, so the
  commit gate keeps the oracle's authority over what commits. It trains on the
  IMPROVEMENT pool only (never the CALIBRATION pool that measures it), filters
  to the current verifier version, and declares itself as a new verifier version
  (`version_tag`) so `attach(boost=...)` / `Audit.recompute()` refresh the audit
  fingerprint and `VerifierWatch` withdraws the old calibration. Default
  behaviour unchanged -- no boost means no fitting and no correction.

- `examples/textgrad/`: a `mechanism_microport` of TextGrad
  (zou-group/textgrad@75e912e2) as a reflection / propose seam. A proposal is a
  textual-gradient call (the base-case backward prompt) followed by a TGD rewrite
  read from `<IMPROVED_VARIABLE>` tags, applied to one prompt slot whose diff id
  hashes the text (an unchanged rewrite is a counted no-op). Acceptance is the
  default held-out gate, so no core contract changes; it is stricter than
  upstream's `run_validation_revert` (a tie is kept upstream, rejected by the
  Beta-posterior gate), which is a declared departure. Registered in
  `bench.candidate_methods`, with `docs/algo-textgrad.md` and an offline
  `tests/test_textgrad_example.py`; `--dry-run` needs no network and no API key.
  The reflection family no longer lists TextGrad as unimplemented (#290).

- `examples/gskill`, a `mechanism_microport` of GEPA's gskill: `optimize_anything`
  learning a repository-specific `SKILL.md` for a coding agent. The artifact is a
  `FileTree` skill directory behind the `skill_dir` structural validator, scored
  by a binary fail-to-pass / pass-to-pass fitness over a vendored twelve-bug
  fixture repository (no SWE-smith, no Docker), accepted under a strict held-out
  gate, and exported as a valid `.claude/skills/<name>/` tree with
  `--export-skill`. `python -m examples.gskill --dry-run` needs no network,
  Docker or API key. Real gskill (Jinja, Bleve) numbers are not claimed (#291).

- `examples/dynamic_cheatsheet/`: a `mechanism_microport` of Dynamic Cheatsheet
  DC-Cu (arXiv:2504.07952, `suzgunmirac/dynamic-cheatsheet@5cfe3c3`), the
  test-time Gen + label-free Cur memory baseline ACE improves on, on a vendored
  offline Game-of-24 slice. The curator never receives a gold answer or the
  reward; the default is online (every rewrite kept) and `--gated` installs the
  held-out gate. `MethodPolicy` gains `blast_radius` (default 0.6, so every other
  port is unchanged) because above 0.3 the runtime audits every commit in front of
  the acceptance policy; the cheatsheet declares 0.2. Registered in
  `bench.candidate_methods`; page `docs/algo-dynamic-cheatsheet.md` (#292).

- LearnableFusionPolicy (#298): a `FusionPolicy` that learns when merging wins.
  An online logistic over diff-only features (candidate count, key overlap, diff
  sizes) is trained on the tournament's own `FusionTrial`s, and once confident
  that the union is the right answer for a merge shape it returns the union
  **without buying the cheap ranking sweep**; uncertain merges still run the
  tournament and observe a fresh label. The rollback guard scores the **pre-
  update** prediction -- the rule actually in force -- and retires skipping
  permanently after a confident-and-wrong counterfactual (default
  `max_wrong_skips=1`). The budget-governor's `tournament=False` degrades to the
  plain union and keeps recording unranked trials on the list `fusion_stats`
  reads. With no trials the policy is `DefaultFusion(tournament=True)` verbatim.
  Commits are unaffected: a skip returns the same union the non-tournament
  default would have committed, and the acceptance gate still scores it on the
  full held-out set.

- Solver-challenger self-play (`agentdescent/selfplay.py`): `curriculum_evolve()`
  co-evolves the solver and its task distribution. A `Challenger` owns a
  task-generation spec; each round it generates a validated frontier batch, the
  `FrontierSampler` mixes those into the solver's train pool, and the challenger
  tunes its spec on the measured learnability (GoDoors-style `r0` = a frontier
  task's last solver score minus its first). The engine runs once -- one
  aggregator, one ledger, held-out fixed from the seed tasks -- so the challenger
  never touches the reported evaluation. A validity gate is mandatory
  (`Challenger.validate` raises on malformed tasks; failures are counted), the
  frontier is self-regulating (tasks the solver already passes leave instantly,
  tasks it never learns are dropped as stuck), and the sampler may return a Task
  object directly, which the engine resolves without a pool lookup (both the
  synchronous and barrier-free paths). `EvolutionResult.curriculum` carries the
  run's `r0_mean` / learned / stuck / invalid / generated stats. Offline demo:
  `python -m examples.selfplay.selfplay_demo --dry-run`.

- Cross-run consolidation (`agentdescent/consolidation.py`): what a run learned
  survives it. `ConsolidationStore` archives a run's product — the artifact
  state, rendered skill, measured reward, rounds and configuration — one JSON
  record per run under `<path>/runs/<run_id>.json`, atomic and idempotent, with
  a durable trajectory-store path linked rather than copied. `seed_from_archive`
  retrieves a candidate for the next run (same configuration fingerprint first,
  then measured reward, same strategy only) and hands its state to
  `evolve(initial_state=...)` **only if a mini-eval on a sample of the new tasks
  shows it does not regress beyond `min_reward` against the empty baseline** —
  a harmful cross-domain seed is refused and reported, never installed.
  `ConsolidationRecord` / `SeedReport` carry what was decided and why;
  non-finite rewards are refused. Offline demo:
  `python -m examples.consolidation.consolidation_demo` (seeded run reaches the
  quality bar ~200x sooner than the empty seed on the measured domain).

- Persistent-island population engine: `evolve(population_islands=N)` (and
  `async_evolve`) widens the population layer into a *persistent* search. With a
  declared selection policy (e.g. `Policies(selection=CladeThompson())`), `N`
  head-islands survive across steps and each records its own lineage
  (`_branch_parent`), so tree-shaped policies score clades -- a lineage that
  accumulates, which is what HGM's metaproductivity means. `PopulationAggregator`
  creates/reuses the island forks instead of re-forking from `dev` every step,
  checkpoints them (`forks`, `island_count`, `branch_parent`, the archive), and
  reaps them at `finalize`. `island_count=0` keeps the old per-step-fork
  behaviour exactly; a value above `0` without a declared selection policy is
  refused ("honour or refuse"). `bench/population_engine.py` A/Bs single-head /
  CladeThompson / islands+CladeThompson at equal budget and reports time-to-
  quality, final reward and committed counts honestly (on a smooth monotone
  landscape single-head is competitive; the population engine targets rugged
  landscapes).

- `ConsolidationStore` / `ConsolidationRecord` / `SeedReport` /
  `config_fingerprint` / `seed_from_archive` and `Challenger` /
  `ChallengerConfig` / `FrontierSampler` / `curriculum_evolve` are listed in
  `agentdescent.__all__` and rendered in `docs/api.md`, matching the
  consolidation and selfplay guides (#341).

- Meta-evolution product surface: `meta_run()` evolves one decision slot of the
  inner search (`meta_evolve` on the train problems), validates the evolved value
  on problems the outer loop never saw (`meta_validate`, with `val_seeds`
  disjoint from the training seeds), and returns a `MetaRunReport` — the rendered
  value (`report.value`; compile with `spec.compile(...)`), the per-problem
  before/after/gain, the transfer ratio (mean val gain / mean train gain), and an
  `adopt` / `reject` verdict with its reason. A value is adopted only when its
  mean gain on unseen problems clears `min_gain` and the transfer is not below
  `transfer_floor`; otherwise it is rejected and the reason says whether that is
  "no gain" or "a fit that does not transfer". The report rides the run result
  (`EvolutionResult.meta`), so a CLI or plugin shows the decision without
  recomputing it. Governance: the caller must keep validation problems and seeds
  disjoint from the training set (`meta_run` raises on overlap), so a verdict is
  never measured on the problems that shaped the value — the same final-evaluation
  isolation every other development set gets. Offline demo:
  `python -m examples.meta.meta_evolve_demo`.

- `agentdescent doctor` (CLI and the MCP `doctor` tool) now reports whether each
  host is wired up and whether it is still current (#382). New read-only
  `integrations.host_status(host)` compares the installed `SKILL.md` with the one
  this package would write (`ok` / `stale` / `missing`) and checks the MCP entry,
  following the same `DSH_HOME` / `CODEX_HOME` / `XDG_CONFIG_HOME` paths as
  `install`. `doctor_report()` gains `hosts` and `hints`: a stale or half-written
  integration is a problem (the skill a `pip install -U` left behind teaches the
  host a procedure the package no longer has), while a host CLI on `PATH` with no
  integration is only a hint. Nothing is written.

- Algorithm registry (#384, slice 1): `EvolveSpec.algorithm` (`"rrsi"` or
  `{"name": ..., "params": {...}}`) picks a named recipe whose `Policies` slots,
  `agg_config` and `evolve` defaults the spec can still override; `agentdescent
  algorithms list [--kind K] [--json]` and `algorithms show <name>` describe each
  one (kinds, typed parameters with defaults, extra cost, fidelity label) from the
  wheel. Unknown names, unknown or mistyped parameters and a kind the algorithm
  does not apply to fail in `plan` naming the field and the valid choices, and
  `plan` echoes the resolved algorithm and which of its defaults the spec
  overrides. Two entries ship: `rrsi` (text; the RRSI screen / proposal /
  acceptance policies) and `archive_selection` (DGM-style archive parent sampling,
  any kind). The RRSI mechanisms moved from `examples/rrsi/` into
  `agentdescent/algorithms/rrsi/` so the wheel carries them; the `examples.rrsi.*`
  modules remain as aliases.
  Because four of those modules are adapted from google-research/rrsi
  (Apache-2.0), the wheel now declares `MIT AND Apache-2.0` (PEP 639; build
  requires setuptools>=77) and ships `LICENSE`, `NOTICE` and
  `LICENSES/Apache-2.0.txt` in its `.dist-info/licenses/`. `plan` warns when
  `rrsi` runs with an empty `protected` list (the leakage screen is then
  vacuous), `archive_selection`'s `temperature` now declares its real lower
  bound, and a failed built-in registry load re-raises the original error.

- `agentdescent --version` (and `-V`) prints `agentdescent <version>` and exits 0.
  `doctor` — the CLI text and `--json` output and the MCP `doctor` tool — now
  reports `version` (`agentdescent.__version__`) and `executable`
  (`sys.executable`), so a host-side `doctor` and a shell-side `doctor` can be
  compared directly to tell whether the host's MCP server runs the same install.
  The SKILL's `doctor` step, `docs/plugin-quickstart.md` and
  `docs/testing-the-plugins.md` say to compare them when results differ (#416).

- Continuous meta-learning (`agentdescent/continuous_meta_run`): the archive
  drives the next session. `ConsolidationRecord` gains an optional `problem`
  descriptor (backwards compatible); `archive_rule` writes an adopted slot rule
  back to the archive with its problem descriptor; `problem_from_archive`
  rebuilds a runnable inner `Problem` from an archived descriptor (the caller
  supplies the tasks via `tasks_for`, the archive supplies the slot and inner
  settings -- the "link, don't copy" rule applied to a problem instead of a
  trajectory); and `continuous_meta_run` closes the loop between sessions --
  train and validation problems come from the archive, the search starts from
  the best archived rule for the slot/spec/config, the gain is measured both
  against that prior and against the fresh seed (the honest cross-session
  number), and an adopted rule is archived so the next session starts from a
  better one. An empty archive is an ordinary `meta_run`. A record archived by a
  different spec is refused at rebuild, never misread.

- Meta-audit — the trust layer for the loop that evolves the rules
  (`agentdescent/meta_audit.py`). `audited_meta_validate` scores every
  `(problem, seed)` validation pair under the meta reward for the decision, and
  with probability `sample_rate` scores the pair again under an independent
  oracle and persists it to a `MetaAuditStore` — a probability sample (every
  record carries its `inclusion_prob`) with a durable, append-only JSONL trail
  that reconciles by last occurrence, so truth arriving later from another
  process can be attached without a rewrite. `meta_run_audited` is `meta_run`
  with this validation: the verdict is persisted beside the sampled evidence,
  and the report carries `audited_n` / `oracle_gain` / `gain_bias`. Over the
  resolved sample, `gain_bias` is the Hajek estimate of `verifier_gain -
  oracle_gain` — the number that says "your meta reward is 0.3 optimistic"
  instead of letting that optimism adopt rules in silence. The decision path
  never sees the oracle (auditing can be turned on mid-fleet without changing
  which rules are adopted), and the same `_require_disjoint` isolation `meta_run`
  enforces applies.

- Meta-population (`agentdescent/meta_population.py`): the archive seeds a
  population, the population promotes. `meta_population_run` ranks the archive's
  rule records for a slot/spec/config as `continuous_meta_run` does, seeds the
  top `islands` distinct rules each into its own session, and every island trains
  on the same problems and validates the same `val_problems` at `val_seeds`
  disjoint from training — so the islands differ only in where they started. Each
  island runs in its own fresh ledger repo (`repo_dir=` names the base directory;
  a shared `repo_path` would resume the ledger and silently ignore the distinct
  `initial_state`), the best island whose verdict is `adopt` is written back via
  `archive_rule`, and `MetaPopulationReport` / `PopulationIsland` report every
  starting rule's outcome, the promotion and the archive count. An island that
  errors is reported and never promoted without stopping the others; a population
  needs at least one archived rule to diversify from (an empty archive is
  refused, with a pointer to run `continuous_meta_run` once first). Offline demo:
  `python -m examples.meta.population_meta_demo`.

- Meta-curriculum (`agentdescent/meta_curriculum.py`): the meta loop generates
  the problems it is judged on. `curriculum_meta_run` is `continuous_meta_run`
  plus a `MetaProblemChallenger` — the solver-challenger of `curriculum_evolve`
  (#329) applied one level up, where the challenger generates frontier *problems*
  instead of frontier *tasks*. Between outer rounds the challenger draws a
  candidate problem configuration, runs it through a mandatory validity gate
  (`validate(config) -> Problem`; a payload that does not build a runnable inner
  problem is counted, never crashed over), and installs the validated problems
  into a `FrontierSampler` that mixes them with the seed problems. Learnability
  is two-phase and measured online (`record` sees the meta reward; a problem the
  rule already solves is dropped at once, one it never learns is dropped as
  stuck), and the challenger's `propose(descriptor, report)` pushes the
  generation spec outward. The challenger never touches the final evaluation —
  the validation set is fixed from `val_problems` and the held-out split is
  computed from the seed tasks alone. The archive still drives continuity: the
  search starts from the best archived rule in a fresh ledger repo, and on
  `adopt` the rule is written back with its problem descriptor. The report
  carries the decision fields plus the frontier statistics (`generated_total`,
  `invalid_total`, `learned`, `stuck`, `r0_mean`). Offline demo:
  `python -m examples.meta.curriculum_meta_demo`.

- Harness composition (`agentdescent/harness.py`): combine algorithm ports, learn
  which wins where. A `Harness` is a named `(tasks, repo) -> EvolutionResult`
  callable; the consolidation archive becomes the leaderboard — one
  `ConsolidationRecord` per `(config, harness)` under `artifact_id="harness:<name>"`
  with the domain's `config` and the measured `final_reward`, written atomically
  and idempotently. `record_harness` / `harness_score` / `harness_leaderboard`
  read and write that evidence; `select_harness` routes a new domain to the
  harness the recorded rewards say wins there (same-config fingerprint first, best
  reward), falling back to round-robin or a caller default when no evidence exists
  — a harness with no runs on a domain can never beat one that was measured there.
  `compose_run` runs the selected harness (`mode="best"`) or every harness
  (`mode="all"`, the leaderboard's training mode), each on a fresh ledger repo,
  records the outcomes, and returns a `HarnessReport` (per-harness reward /
  time-to-quality, winner, selection reason). Every algorithm port binds into a
  `Harness` unchanged; the archive's shape and `config_fingerprint` domain
  separation are reused as-is.

- Self-exam (`agentdescent/meta_exam.py`): adoptions are provisional until the
  oracle says so. `exam_archive` walks the archive's rule records, re-validates
  each against the problems its own `problem` descriptor describes (rebuilt by
  `problem_from_archive`) using an independent oracle, and stamps the outcome on
  the record via `ConsolidationStore.mark_exam`: `passed` when the oracle gain
  clears `min_gain` (flags reset), `flagged` when it does not (flags + 1), and
  `demoted` when a rule fails truth `max_flags` times in a row. A demoted rule is
  excluded from future seeding — `_archived_rules` skips it, so
  `continuous_meta_run` and population selection never start a search from a rule
  truth has retired. `ConsolidationRecord` gains optional `exam` / `exam_flags` /
  `exam_note` fields (backwards compatible: an archive written before the exam
  read back as `None` and stays seedable until examined); `mark_exam` rewrites
  atomically (tmp + rename). Adoptions are thus provisional, and the archive is
  self-correcting: what the loop believed is checked with evidence, and what
  fails the check is retired.

- Portfolio governance (`agentdescent/portfolio.py`): watch every layer's
  measured return, shift the budget to what pays. A `LayerBudget` is one spending
  line (a spend limit, what it spent, the measured return that spend bought, and
  how many measurements it rests on). `portfolio_report` ranks layers by
  return-per-cost and reallocates the next window deterministically and
  conservatively: a layer at or above `floor` with at least `min_samples`
  measurements grows (`budget * (1 + grow)`, capped), one below the floor or
  flagged `diminishing` shrinks (`budget * (1 - shrink)`), one with too few
  measurements is held (no budget moves on a sample of one), and no layer is ever
  zeroed. `layers_from_archive` turns the consolidation archive's rule records
  into one layer per decision slot (return = mean measured reward, spend = mean
  rounds per rule, budget = total rounds, samples = rule count) — the honest view
  of "which decision's evolution
  pays" — and other layers (audit bias, harness leaderboard, population islands)
  are caller rows in the same table. Non-finite inputs are refused. The rule is
  the fixed, deployable baseline for the next rung: evolving the reallocation
  policy itself via `meta_run`.

- Chain audit (`agentdescent/chain_audit.py`): one adoption, traced through every
  evidence ring. `chain_audit` walks an archived rule's rings — governance
  (decision plane vs machinery), the recorded decision, the meta-gain oracle
  bias (from a `MetaAuditStore`, #426), the self-exam stamp (passed/flagged/
  demoted), the inner verifier's residual bias (from an `AuditStore`, #273), and
  harness routing evidence — and reports each ring as `verified` (evidence
  present and within `bias_threshold`), `flagged` (present and outside it), or
  `unverified` (the evidence simply is not there, reported rather than silently
  assumed). `ChainAuditReport.verified` is true only when no ring is flagged or
  unverified, and the reason names the rings that keep it false. An unknown run
  is refused with `KeyError`. The promise "no number in the loop is unverified"
  becomes inspectable end to end.

- The meta stack on the command line: `agentdescent portfolio`, `exam`, and
  `audit`. `portfolio [archive]` reads the consolidation archive (default
  `$AGENTDESCENT_HOME/archive`), builds one `LayerBudget` per decision slot via
  `layers_from_archive`, runs `portfolio_report`, and prints each layer's
  return-per-cost and reallocated budget with its reason (`--json` for
  machine-readable output; empty archive exits 1 with a pointer to run a meta
  session first). `exam [archive] --domain module:builder` resolves the domain
  through a `module:attribute` ref (a zero-arg builder returning `oracle`,
  `tasks_for`, `run`, `reward`, `propose`, `strategy`, and optionally `spec`),
  runs `exam_archive`, and reports passed / flagged / demoted per rule. `audit
  <run_id> --archive PATH [--meta-audit ...] [--inner-audit ...] [--harness
  module:callable] [--bias-threshold N]` traces one archived adoption through its
  evidence rings via `chain_audit` and prints each ring's verified / flagged /
  unverified status; an unknown run or a bad ref fails cleanly (non-zero exit),
  never with a traceback. One surface, two transports: the same verbs the MCP
  server can expose.

- Trajectory-driven curriculum (`agentdescent/trajectory_curriculum.py`): the
  problems come from the run's own failures. `failed_rollouts` reads a rule's
  linked `TrajectoryStore` (via `ConsolidationRecord.trajectory_store`, #334) and
  returns the rollouts whose reward fell below `failure_threshold` — the memory
  of #248 turned into evidence. `trajectory_challenger` is a
  `MetaProblemChallenger` whose generation spec is an archived run_id: `generate`
  mines that rule's recorded failures, `validate` builds a frontier problem over
  the failed tasks via `problem_from_tasks` (raising — and so counted invalid —
  when there is nothing to mine), and `propose` advances across the archive's
  rules that recorded trajectories. Plug it into `curriculum_meta_run` (#429) and
  the meta loop's frontier is generated from what its own runs got wrong, with
  the #429 guarantee inherited unchanged: the validation set stays fixed and the
  held-out split is computed from the seed tasks alone, so a mined failure can
  only ever appear in the training frontier.

- Async meta-population (`async_meta_population_run`): the islands of
  `meta_population_run` run concurrently. Each island starts from its own
  archived rule, trains on the shared problems, validates the shared isolated
  set, and runs in its own fresh ledger repo — so they are independent sessions
  run through a `ThreadPoolExecutor` with `workers` islands at a time (default
  all of them). The archive is only read while the islands run
  (`ConsolidationStore` is thread-safe) and written once at promotion, so the
  decision and the report are identical to the sequential call's, only faster to
  reach. An errored island is reported and never promoted, exactly as sequential;
  an empty archive is refused with the same pointer to run `continuous_meta_run`
  first.

- Evolved decision code that beats the shipped implementation on quality-per-cost
  (`agentdescent/evolved_programs.py`). `source_ops` counts the arithmetic/call
  nodes of a candidate rule's source — a deterministic, attributable cost proxy;
  `priority_grid_problem` scores a compiled `priority` rule's ordering agreement
  with the shipped seed's own ordering on a fixed grid (the seed scores 1.0 at
  the baseline cost); `cost_adjusted_reward` is `quality - lam * (cost / baseline)`,
  so a candidate must be cheaper at equal quality (or better at equal cost) to
  outscore the seed; and `evolve_fast_rule` evolves the `selection` priority rule
  on that reward and reports the evolved rule against the seed on **both axes**
  (`FastRuleReport.seed_ops` / `evolved_ops` / quality / the cost-adjusted gain).
  The AlphaEvolve / FunSearch idea applied to the decision plane: the search
  finds a rule that is as good *and cheaper* than the hand-written default, and
  the report measures it.

- Learn-to-think (`agentdescent/learn_to_think.py`): the framework evolves *when*
  to spend compute. The route rule is a gated source `Policy.route(difficulty,
  budget_left, reward)` returning `direct` / `search` / `reflect` (the gate
  refuses imports, dunder access, forbidden calls, a wrong signature, and a
  return outside the three routes; a smoke test runs the route on easy /
  hard-with-budget / hard-without-budget inputs). `think_problem` is a workload
  instance whose outcome depends on the route — each route has a cost model and a
  quality model, and the reward is quality-per-cost — and `learn_to_think` runs a
  bespoke `evolve()` over those problems (the artifact is the route source, the
  propose rewrites it) so the framework learns, from its own measured returns, to
  search the hard tasks and answer the easy ones directly, instead of paying a
  fixed price everywhere. `LearnToThinkReport` compares the evolved rule with the
  default on validation problems and shows the route each chose on an easy and a
  hard input. The Search-R1 / ZigZag "learn when to search" idea, applied to the
  decision plane.

- Self-rewarding meta-evolution (`agentdescent/self_rewarding.py`): the judge
  improves as the run goes. `outcome_features` turns a `MetaOutcome` into
  curve-shape features (mean best-so-far, final, sweeps, rollouts — never the
  answer); `LearnableMetaJudge` is a small pure-Python linear regressor from those
  features to oracle quality, fit by least squares (deterministic, no numpy),
  trained **only on labels the oracle actually produced**; and
  `self_rewarding_meta_run` alternates — evolve the slot with the plain `auc`,
  score `judge_sample` outcomes with the oracle and fit the judge, then re-evolve
  with the judge as the meta reward and refit on the newly oracle-labelled
  outcomes — so each round the search moves against a closer approximation of
  truth. The report carries the judge's measured agreement with the oracle on
  validation outcomes it never trained on (`SelfRewardingReport.judge_agreement`),
  how many labels it learned from, and the evolved value's oracle gain over the
  seed. The Self-Rewarding Language Models (Meta 2024) idea applied to the meta
  level.

- Rule tournament (`agentdescent/rule_tournament.py`): candidates play against
  each other, ratings decide promotion. `rule_tournament` runs a round-robin
  where every pair plays twice on each other's ground — rule A is scored by a
  critic on rule B's challenge problem (rebuilt from B's descriptor via
  `problem_from_archive`, or a caller-supplied `challenges` map) and rule B on
  A's — and `EloRating` (K-factor 32, rating mass conserved, draws split) converts
  the matches into ratings. A rule that wins on its rival's ground is a better
  rule, and `tournament_promote` re-archives the winner with its measured final
  reward (its mean critic score on rivals' ground), so the next session's
  `_archived_rules` ranks the tournament's choice first. The self-play /
  verifiable-reward idea (DeepSeek-R1, self-playing agents) applied to the
  archive; a tournament with fewer than two rules is reported as no tournament.

- Self-modifying framework (`agentdescent/self_modify.py`): the framework evolves
  patches to its own code, gated by its own tests. `patch_source` / `apply_patch` turn a candidate into a
  unified diff and back (a patch that cannot apply is refused); `gate_source` is
  the same AST walk every other self-edit in the framework runs (no imports, no
  dunder, no interpreter calls — a candidate compiles in a namespace of safe
  builtins only); `evaluate_source` runs the candidate against callable tests and
  a benchmark, where **every test must pass before any benchmark score counts** —
  a patch that breaks anything is a zero, never a regression with a number
  attached; and `improve_source` runs the ordinary `evolve()` over `(test, seed)`
  tasks, so the framework searches its own code under its own gate and reports
  the accepted patch, the tests it passed, and the measured gain over the seed
  (`SelfModifyReport`).

- World model (`agentdescent/world_model.py`): the framework learns an internal
  simulator and dreams before it acts. `LearnedSimulator` is a pure-Python
  nearest-neighbour model fit on observed `(state, reward)` pairs — deterministic,
  dependency-free, and unable to extrapolate beyond what it has seen, which is
  exactly the honesty a simulator of a measured domain should have;
  `simulator_problem` turns the simulator into a `Problem` (a cheap, model-free
  rollout); `simulator_score` measures the simulator against the real problem
  (mean abs error + rank agreement), so the framework knows when to trust the
  dream; and `imagine_then_evolve` spends most of its budget in the dream —
  evolving against the simulator via `meta_evolve` — then verifies the value on
  the real `val_problems`, reporting the dreamed vs verified rollout counts and
  the real gain over the seed (`ImagineReport`). The Dreamer loop applied to the
  framework: pay the real cost where it pays, dream where it is cheap.

- Agent society (`agentdescent/agent_society.py`): the framework evolves a
  cooperating topology, not just one artifact. `AgentNode` (name, instruction,
  budget) + `SocietyGraph` (nodes, delegation edges, root) are the container;
  `society_spec` is the evolvable artifact — a gated `society()` source compiled
  into a `SocietyGraph`, with `SOCIETY_SEED` the single-agent baseline;
  `society_problem` is a multi-step task routed through the graph, rewarded as
  correctness minus delegation cost; and `society_evolve` runs the ordinary
  `evolve()` over those tasks, so the search finds the topology that solves the
  task with the least hand-off. `SocietyReport` compares the evolved graph with
  the seed on validation tasks — the delegation dimension (Anthropic's
  multi-agent research system; AgentEvolve) measured, not assumed.

- Trajectory tree search (`agentdescent/trajectory_search.py`): the evolution
  loop becomes a learned MCTS over states. `SearchTree` is a state tree over
  rendered values with UCB selection (`ucb_c`), expand and back-propagation;
  `search_evolve` spends a fixed budget expanding the most promising branches —
  each leaf is branched by a callable, scored by a critic (`auc` by default, or
  an oracle / the self-rewarding judge) over seeds, and the value back-propagated
  — and returns the highest-mean-value state with a `SearchReport` carrying the
  budget, the node count, the search's own value estimate, and a **freshly
  verified** score of the best state side by side, so the search's reasoning is
  inspectable against final truth. The LATS / AlphaCodium idea applied to the
  state space, using the same UCB bandit the tree policies in `selection` run,
  one level up.

- Failure-weighted replay (`agentdescent/curious_replay.py`): the framework
  replays the past it is still failing. `ReplayExample` is one replayed past task
  with its rolling scores; `curiosity_weight` is high while the artifact is still
  failing it (falls as it improves, floor once mastered); `curious_replay_evolve`
  runs the ordinary `evolve()` over one task per replayed example, weighted by a
  **snapshot** of the weights taken before any candidate rolls out, with the
  mastered and still-curious cohorts interleaved so both land on both sides of
  the held-out cut; each outcome is recorded into the examples' rolling scores
  for the next call, and a `CuriousReplayReport` shows the weights and the gain
  on the still-curious vs mastered past. The Curious Replay (ICLR 2025) idea:
  spend the budget on the past not yet conquered, measured not assumed.

- Rationale distillation (`agentdescent/rationale_distill.py`): the artifact
  learns to carry its own reasoning. `extract_rationale` turns one rollout into a
  rule — a success says how it answered, a near-miss says what it needed
  (`need cap >= d+1`) — and `distill` dedupes the rollouts' rationales into the
  skill the artifact carries; `rationale_evolve` runs the ordinary `evolve()` and
  then distills the run's own successes back into the skill, returning a
  `RationaleReport` with the distilled skill, the rationale count and the success
  count. The STaR (Self-Taught Reasoner) idea: the artifact boots itself by
  carrying what its own runs taught it, near-misses included.

- Self red-teaming (`agentdescent/red_team.py`): the framework attacks its own
  value to find the OOD gap. `red_team` mutates the training tasks outward and
  scores the current value on each mutation, collecting the failures — the
  out-of-distribution region where the value does not generalise (deterministic,
  reproducible); and `red_team_evolve` runs the ordinary `evolve()`, attacks its
  own result, and retrains on the union of the problems and the failures it
  found, closing the OOD gap. `RedTeamReport` shows the failures found and the
  coverage (fraction of the attack's mutations solved) before and after the
  retraining, so "the framework found its own blind spot and fixed it" is
  measured, not claimed.

- `show` (CLI and MCP) now explains a low reward: a new `samples` key lists up to
  five of the lowest-reward rollouts of the final artifact, each with task id,
  the agent's output, the task's gold (both cut to 300 characters) and reward,
  under a fixed 4 KB cap; text mode prints a short block. The run store keeps the
  engine's rollout records under `ledger/trajectories/` in the run directory. A
  new spec field `record_rollouts` (default `true`) turns this off. First part of
  #452; rejected-candidate diffs follow separately.

- Verifiable-reward policy improvement (`agentdescent/verifiable_rl.py`): the
  framework trains on truth it can check. `verifiable_problem` wraps a
  `(value, task) -> output` problem so its outcome carries the output and the
  gold in `detail`; `verifiable_reward` scores exact match (1.0 / 0.0 — a rule
  that cannot be checked is not rewarded, the DeepSeek-R1 verifiability
  contract); and `policy_improve` evolves a decision slot with that reward via
  `meta_evolve`, reporting the per-round verifiable reward curve (the
  trajectory-level signal RL trains on) and the before/after on held-out
  verifiable facts (`VerifiableReport`).

- Joint multi-slot evolution (`agentdescent/joint_meta.py`): search the whole
  decision plane as one artifact. A `PoliciesBundle` holds several `SLOTS`
  together; each outer rollout installs **all** the slots on `Policies` and runs
  the inner search, so an effect that only appears when two rules change
  together is visible to the search (an effect two independently evolved rules
  never see). A proposal names one or more slots with `=== <slot> ===` blocks, so
  the aggregator's existing union-on-distinct-keys / resolve-on-same-key rules
  apply unchanged. `joint_meta_validate` attributes a joint change to each slot by
  its **marginal** held-out gain and, with `isolate=True`, rolls back a slot that
  regresses while adopting the rest -- so one over-eager rule cannot smuggle a
  regression into the archive. An adopted bundle archives one ordinary
  `meta-rule` record per slot, so `_archived_rules`, `exam_archive` and
  `continuous_meta_run` keep seeding and examining it unchanged.

  Based on #388 (P7 multi-slot joint evolution + isolating untrusted rules).

- Closed self-improvement loop (`agentdescent/self_improve.py`, with
  `agentdescent/evidence.py` and `agentdescent/admit.py`). `evidence.snapshot`
  reconciles the consolidation archive into one deterministic view -- per decision
  slot's return-per-cost and exam stamps (from `portfolio.layers_from_archive`),
  the harness leaderboard, and the archive's demoted/flagged counters -- and
  `rank_slots` turns it into the next targets (falling back to round-robin on a
  cold archive). `admit` composes an adoption's trust rings (exam / audit /
  tournament / red-team / chain) into one fail-closed `pass` / `flag` / `demote`
  verdict -- the generalisation of the joint rung's "isolating untrusted rules" to
  every adoption path. `self_improve` runs the loop: read evidence, evolve the
  paying slots with `meta_run` (adopt -> `archive_rule`), trust-gate each adoption (a
  `demote`/`flag` is stamped on the record with `mark_exam`, quarantining it in the
  archive's own paths), and move the next tick's budget with `portfolio_report` --
  for a bounded number of ticks. Ticks compound: each target is seeded from the
  archive's best rule for its slot, like `continuous_meta_run`. Targets are chosen
  by evidence with an exploration share and a per-tick rotation, so an unseen slot
  is tried instead of starved.

  Based on #465.

- End-to-end self-evolution bench (`bench/self_evolution.py`): one command runs
  the whole chain — seed the archive (`continuous_meta_run`), diversify
  (`meta_population_run`), re-judge with truth (`exam_archive`), reallocate spend
  (`layers_from_archive` + `portfolio_report`), compose competing harnesses
  (`compose_run` / `harness_leaderboard` / `select_harness`) and trace one
  adoption (`chain_audit`) — on a deterministic offline arithmetic domain, and
  reports every rung's measured number. Exits non-zero only when a rung fails to
  run or produces no measurement; a raising rung is recorded as a failed stage,
  not a crash. `--json` prints the structured report.

- Portfolio evolution (`agentdescent/portfolio_evolve.py`): the framework evolves
  the rule that governs its own spend. `ReallocationPolicy` is the reallocation
  rule as four numbers (`floor`, `grow`, `shrink`, `min_samples`); its `apply`
  delegates to the shipped `portfolio_report`, so the evolved rule is the fixed
  policy with different numbers (the default is the baseline, no second
  implementation to drift). `reallocation_spec` compiles the numbers with
  `ParamSlot`; a `LayerBudget` window is the problem, scored by the projected
  return the policy's reallocation would buy; `portfolio_evolve` runs the ordinary
  `evolve()` over the archive's (or the caller's) windows and reports the evolved
  policy's validation gain over the fixed baseline with an adopt/reject verdict.

- Verified seeding (`agentdescent/verified_seed.py`): trust becomes a gate, not a
  report. `verified_rules` starts from `_archived_rules` and returns only the rule
  records that clear the caller's bar -- `min_exam` (default `"passed"`, so an
  unexamined or once-flagged rule does not seed) and, when a `MetaAuditStore` is
  supplied, a gain bias within `bias_threshold` -- and explains, per excluded run,
  why it did not qualify. `continuous_meta_run`, `meta_population_run` and
  `async_meta_population_run` gain a `min_exam` knob that filters the seeding
  rules before the session starts; `None` keeps the existing behaviour unchanged.

- `show` (CLI and MCP) can now say what a run that learned nothing actually
  proposed. The run store writes `candidates.jsonl` beside `rounds.jsonl` with the
  last few diffs the merger measured and refused: touched files, a short diff, the
  rationale, the gated probability and the outcome. `MergeReport.candidate_diff`
  and `evolve(on_reports=...)` expose the record; `record_rollouts: false` still
  writes neither diagnostic file. `show` returns a bounded `rejected` block, and
  the shipped SKILL.md tells the host to read `samples`/`rejected` before
  suggesting more rounds.

- The named-algorithm registry is now reachable from the plugin surface, not just
  from a spec file: MCP gains `list_algorithms` / `describe_algorithm`, `init` and
  `plan` accept `--algorithm NAME --param KEY=VALUE`, and `docs/algorithms.md` is
  generated from the registry. The shared SKILL.md, the Claude Code `/evolve`
  command and `docs/plugins.md` explain when to offer a method (only when the user
  names one or the artifact clearly fits) and that the kind's default recipe is
  used otherwise.

- GEPA's Pareto illumination is now a named algorithm (`"algorithm": "gepa"`,
  `kind: text`) in the in-package registry (#384, slice 5a). The Pareto pool,
  per-instance scoring and win-frequency parent selection moved from
  `examples/gepa` into `agentdescent/algorithms/gepa.py`; the example keeps its
  HotpotQA domain, prompts and CLI and imports the one implementation, and the
  generated catalog lists its `seed` / `eval_concurrency` parameters.

### Changed

- Install docs no longer say "once published" / "before publication" for 0.6.0:
  the release is on PyPI. README installs it with `pip install "agentdescent>=0.6.0"`
  (core package; the offline demo needs no extra), `docs/plugins.md` and
  `docs/plugin-quickstart.md` use `pip install "agentdescent[mcp]>=0.6.0"` for the
  CLI and MCP tools, and `docs/install.md` answers "Can I install from PyPI?" with
  that command while keeping the installer / current-main flow first; a source
  checkout is the alternative (#165).

- RRSI wiring fidelity closure (`examples/rrsi/rrsi_policies.py`): the wired
  proposer/selector now consume the **pinned upstream fixtures**, not
  approximations. Component attribution goes through `components.normalize` /
  `classify_diff` (the upstream `K` / `K_STR` evidence rules — `skills/x/SKILL.md`
  is a `skill` edit, `ToolRegistry(...)` is `client_tool`, pure prose is
  `prompt`), replacing the first-path-segment heuristic. The per-edit ledger is
  `history.History` (the pinned durable JSONL) when `build_rrsi_bundle(
  history_path=...)` is given, with an in-memory twin for scratch runs pinned to
  the same summaries by a parity test. `RRsiProposalPolicy` reads
  `History.render` / `prune_set` (with the actionable accepted edits) /
  `stall_flag`; `RRsiAcceptance` scores novelty over
  `History.incumbent_component_counts`; the aggregator checkpoint now carries
  only the search state (round, S*, incumbent cost, progress), while the
  per-edit evidence is the History's own file — so a resumed run reads the same
  evidence the loop conditioned on. Six fidelity tests pin the vocabulary, the
  memory/file parity, durable resume and a genuinely novel structural component
  carrying a within-band candidate.

### Removed

- `integrations/dsh-agentdescent/node_modules` (210 tracked files of a pnpm peer
  install) is no longer in git (#165). The package's `files` list never shipped
  it, the tests build their own tmp `node_modules`, and the npm publish workflow
  uses only the declared `files`. A root `node_modules/` entry in `.gitignore`
  keeps it that way; `tests/test_repo_hygiene.py` asserts none are tracked.

### Fixed

- `agentdescent demo`'s next-step footer now shows `init ... --out spec.json`, so
  the `plan spec.json` line after it names a file that exists; bare `init` writes
  `.agentdescent/<name>.evolve.json` (#484).

- Dynamic Cheatsheet online mode now matches its claim (#292, follow-up to #317):
  `MethodPolicy` gains opt-in `cheap_eval_tasks` (default `None`, engine default)
  so a cheap/full held-out disagreement can no longer lower the audit's trust
  below 0.75 and veto a rewrite the accept-everything policy meant to keep, and
  opt-in `test_time_learning` (default `False`) so the runner scores the test
  split as a sequential Gen -> Cur stream. No other port changes. The
  self-evolution catalog counts (twenty-three ports, fifteen microports) are
  corrected across the docs, README and install guide.
  Standard baseline/final scores stay frozen so quality_gain is not
  mixed-protocol; streamed learning is under `framework.test_stream`.

- `tools.changelog_fragments` release no longer truncates `CHANGELOG.md` before
  the replacement is safely on disk (#297). The folded text is staged in a
  sibling temporary file, flushed and closed, then atomically replaced with
  `os.replace`; fragments are removed only after that replace succeeds. A
  write, flush, close or replace failure leaves the original history and every
  fragment untouched.

- `tools.changelog_fragments` release now syncs `CHANGELOG.md`'s parent
  directory after the atomic replace and before deleting any fragment, and
  keeps the existing file's POSIX permission bits instead of publishing the
  private `0600` staging file (#301). If the directory sync fails, the error
  says `CHANGELOG.md` may already have been replaced and every fragment is
  kept. Non-POSIX platforms skip the directory sync and mode copy explicitly.

- `LearnableFusionPolicy` no longer trains on an unranked union after its
  tournament flag is switched on at runtime (including the engine budget
  governor's re-enable) (#302). The label-producing path now always delegates
  to an inner policy that actually ranks the candidates, and a merge the inner
  did not rank is neither counted as a tournament run nor used as a training
  label. The budget-disabled union path and the default behaviour are unchanged.

- Multi-artifact contract dependencies now include the primary artifact (#303):
  the reverse dependency graph is built from every registered artifact's
  effective contract, so a primary whose `contract=` `depends_on` an extra is
  required to land an adapter with a contract-breaking change to that extra
  (`missing-adapters` otherwise) and has its cached evaluations invalidated on
  the branch that changed, in both `evolve()` and `async_evolve()`. Runs without
  primary dependencies are unchanged.

- `VerifierBoost` no longer flips between active and inactive on repeated
  `Audit.recompute()` calls with no new data, and a wrapped verifier that only
  defines the legacy `oracle_eval` no longer raises `AttributeError` (#304).
  Activation changes the audit's measurement version but not the scorer's:
  `AuditedReward.scorer_version` (stamped on records as an optional
  `AuditRecord.scorer_version`, omitted when equal to `verifier_version`) names
  the scorer alone, the boost trains on every label of that scorer, and keeps a
  validated curve when the unchanged scorer merely lacks labels — a different
  scorer still withdraws it. `VerifierBoost.full_eval` / `oracle_eval` delegate
  through `full_eval_of`, preserving the inner verifier's independence declaration.

- `prob_improvement` no longer hangs forever on a non-finite posterior (a damaged
  or hand-edited `checkpoint.json` restored via `Aggregator.restore()`) or
  crashes on negative counts / `samples < 1`. `BetaPosterior` refuses a
  non-finite or negative count or update weight, `observe_delta(nan)` is ignored,
  `prob_improvement` raises `ValueError` before sampling, `restore()` skips such
  a posterior like any malformed entry, and `DefaultAcceptance` rejects
  `accept_samples < 1` at construction / configure time (#305).

- `VerifierBoost.retrain()` no longer activates a calibration curve that
  only shifts the verifier's mean: the bootstrap gate must now also beat the best
  constant predictor (the train slice's mean oracle score) on the validation
  slice by `min_gain`, so a verifier with no discriminating power that merely
  reads systematically high or low is rejected ("no discrimination beyond a
  constant") instead of being "validated" and re-ranking candidates. A curve with
  fewer than two distinct fitted levels is rejected too, and `status()` and
  `BoostState` expose the new `error_const` baseline (#306). On rejection via the constant-baseline or bootstrap gate, the measured `error_const` / validation MSEs are retained on the rejected `BoostState` so `status()` can diagnose why the candidate failed.

- `audit.ppi.t_ppf` is now accurate for every `df > 0`. The Cornish-Fisher expansion
  was 11% too narrow at df = 1 (36% at the 99% level) and badly off below 1, and
  cluster-robust PPI intervals (`df = groups - 1`) routinely land there. The
  expansion is kept for `df >= 30`; df 1 and 2 use closed forms and every other
  `df < 30` inverts the exact Student-t CDF (regularized incomplete beta) with a
  bracket that grows instead of saturating (#313).

- `audit.ppi` refuses what it used to turn into a NaN or zero-width interval:
  `Stratum` raises `PPIError` for a non-finite `weight`, any non-finite score in
  `f_lab`/`y_lab`/`f_unlab`, non-finite `mean_unlab`/`var_unlab`, and a non-integral
  or non-finite `n_unlab`; `ppi_mean_stratified` raises `PPIError` unless
  `0 < alpha < 1` and `k_folds >= 2`, and its weight-sum check now fails on NaN.
  `Calibrator` rejects an invalid `alpha` with `ValueError` at construction, and
  `Calibrator.recompute` turns any unexpected estimator exception into a stale
  rectification, keeping `current()`'s "never raises" contract (#314).

- Audit records that carry a nonempty `scorer_version` (VerifierBoost identity) now persist `schema_version=2`. The previous release kept `SCHEMA_VERSION=1` while adding the field, so an older v0.6.0 reader hit `TypeError` on the unknown keyword, counted the line corrupt, and silently skipped pending work. The new reader still migrates v1 lines (unstamped and the mistaken early stamp); unsupported versions keep raising explicitly (#315). Scorer-identity behaviour from #304/#312 is unchanged.
  A v1 line that parses but cannot construct a record (missing key, invalid inclusion probability) is now one corrupt line rather than an unopenable store, and a record with an unknown key is counted corrupt instead of loading under default values.

- `audit.plan_audit` (`audit.sampler.plan`) now refuses an enabled `AuditPolicy` with `alpha` outside (0, 1), `calibration_fraction` outside [0, 1], a NaN `target_halfwidth`, negative or fractional label counts, and negative or non-finite stratum weights or `expected_units`, raising a `ValueError` that names the field. Previously these crashed with unrelated errors, returned NaN inclusion probabilities, or silently planned the wrong number of labels (#325). A disabled policy is unchanged.

- `audit.ppi.Stratum` now refuses a numeric string or a `bool` in `weight`,
  `n_unlab`, `mean_unlab` or `var_unlab` with a `PPIError` that names the field,
  and stores those fields as plain `float` (`int` for `n_unlab`). Previously
  `weight="0.5"` escaped as an unrelated `TypeError`, `mean_unlab="0.3"` was
  stored as a string and broke `ppi_mean_stratified` later, and `True` was
  silently read as 1 and reported back as `True` in `per_stratum`. numpy scalars
  and integral-float counts are still accepted (#327).

- `audit.sampler.plan` now refuses a stratum-weight total that is not finite and
  positive (two finite `1e308` weights used to overflow the sum to `inf` and raise
  `ZeroDivisionError`), and requires `AuditPolicy.target_halfwidth` to be finite
  and `> 0` (0 / negative used to silently fall back to `max_labels`; ±inf
  collapsed the Neyman total so only the floors decided) (#330).

- `audit.calibrator.Calibrator` now refuses `min_labels` / `min_per_stratum` that
  are not non-negative integers (`numbers.Integral`, not `bool`) at construction,
  matching `AuditPolicy` (#325). Floats, numeric strings and negatives used to be
  stored and later poison folding or the stale message as `TypeError` (#331).

- `audit.gate.RectifiedAcceptance` now requires `inflate_when_stale` to be a
  finite real number `>= 1.0` (`bool` refused). NaN and inf used to pass the old
  `< 1.0` check and yield `var_after=nan` (or zero evidence) on the stale path
  (#332).

- `FrontierSampler.install` keeps the measured history of a task that is redrawn
  (an equivalent task is deduplicated, a conflicting task under a known id is
  refused and counted invalid), so redrawing no longer erases learning or resets
  the stuck threshold; and the engine records only the designated solver's
  rollouts in the sampler, so an extra artifact's score on the same task is not
  read as the primary's first-to-last learning (#336).

- The RRSI example bundle's default in-memory history now rides the checkpoint, so
  a resumed bundle keeps accepted component counts, pruning and proposer evidence
  (an already-accepted skill is no longer "novel" again; a durable `history_path`
  still reads its own file). `upstream_components` now classifies each op through
  a real changed-line diff, and slot-keyed ops (`SingleSlot` / rule ids) as
  `prompt`, so prose such as "Do not mention ToolRegistry" no longer gets
  structural novelty, and comment- or string-only edits stay `prompt` (#337).

- `ConsolidationStore` filenames hash the complete `run_id`, so distinct ids
  such as `a/b` and `ab` no longer share `runs/ab.json` and overwrite each
  other's archived products; `find` still accepts a pre-#338 sanitized file
  only when the stored identity matches (#338).

- `PopulationAggregator.restore` drops island fork names whose ledger branches
  were already deleted by `finalize`, and `step` recreates a missing island
  head before committing to it — a clean-stop checkpoint no longer routes workers
  to nonexistent branches on resume (#339).

- `compose` / `plan` refuse `kind: "text"` when `agent` resolves to a
  `WorkspaceAgent` (`claude_code` / `codex` / `dsh` / `opencode` / `cli_agent`),
  with a `SpecError` naming `agent` and pointing at `openai_compatible` /
  `host_model` / `claude` instead of returning `ok: true` (#340).

- `--budget` / `budget_usd` without `--usd-per-call` / `usd_per_call` is rejected
  before a run directory is created, so detach and MCP `start` no longer report
  ok/started and then fail only in `log.txt`; `status` also surfaces the real
  `error:` line from the log when a child dies without writing a terminal state.
  The plugin quickstart now shows both flags (#342).

- `plan` / MCP `plan` returns `ok: false` when a spec has fewer than 4 tasks —
  the same train/held-out rule `evolve()` already raises — instead of a soft note
  that let agents start a run that then crashed. Soft "8–20 is a workable minimum"
  notes remain for 4 ≤ n < 8 (#343).

- `plan` / MCP `start` / CLI `evolve` fail closed when a named CLI agent
  (`claude_code` / `codex` / `dsh` / `opencode`) is missing from `PATH`, instead of
  returning `ok: true` and failing later with `AgentError`; the error names the
  binary and how to install or switch agents. Isolation/login notes stay soft
  warnings when the binary is present (#346).

- MCP `start` in nested mode (`AGENTDESCENT_NESTED=1`) still returns the
  documented refused stub when the proposed `agent` / `reflect` CLI is absent
  from `PATH`; the missing-CLI fail-closed gate runs only for non-nested starts
  (#351).

- `meta_run` no longer rejects distinct sequence-based train and validation
  problems (#356). The disjointness gate compared synthetic `p0`/`p1` labels
  assigned independently by `_named_problems`, so every pair of nonempty
  `Sequence[Problem]` inputs collided before `meta_evolve`. Isolation now
  checks explicit mapping keys when both sides are mappings, and problem
  object identity for genuine shared callables (including the sequence API);
  seed-overlap rejection is unchanged.

- `status` (CLI and MCP) now shows `pending_calls` and `phase` while an actor
  (`propose` / `run`) is in flight, so a multi-minute first openai_compatible
  propose no longer looks hung at `calls=0` (#358). In-flight state is paired
  by token (`Usage.begin` / `Usage.end`), so a provider's own retry records on
  a shared `Usage` do not end the actor call, `phase` reflects the calls still
  pending, and concurrent status rewrites are serialized per run.

- `agentdescent init` without `--agent` now writes the worker CLI that is on
  PATH (first of `claude_code`, `codex`, `dsh`, `opencode`, as `doctor` sees
  them) instead of always `claude_code`, so the quickstart `init` -> `plan` path
  works on an OpenCode-only machine. With no worker CLI on PATH it keeps the
  `claude_code` placeholder and prints a warning pointing at `doctor` /
  `--agent`; `--agent` still overrides, and only `claude_code` gets
  `--permission-mode acceptEdits` (#361).

- The `kind=skill_dir` / `agent_dir` "needs a WorkspaceAgent" error (and the
  `tree_runner` one) now lists `opencode` alongside `claude_code`, `codex`, `dsh`
  and `cli_agent`; the directory-kind and `kind=text` messages share one
  `WORKSPACE_AGENT_REFS` phrase so they cannot drift apart (#362).

- `plan`, `evolve` and MCP `start` now check a host plugin's `host` CLI on
  `PATH` separately from `agent` and `reflect`, so an OpenCode-only machine no
  longer gets `ok: true` (or a launched, doomed run) for a Claude-format plugin
  whose `claude` host is missing. `init` warns when the plugin's host CLI is
  absent. The nested MCP start stub is unchanged (#368).

- `resume` (CLI and MCP) now re-checks a stopped `kind="plugin"` run's stored
  `host` CLI on `PATH` before relaunching it (#370). A run started where `claude`
  was installed and resumed where it is not used to reach `runstore.launch`,
  write a fresh `running` status and fail only inside the detached child; it now
  refuses with an error naming the `host` role and the missing binary, leaving
  the stopped status, PID, spec and ledger untouched. Only the host is a hard
  check on resume: `agent` / `reflect` keep their soft warnings, an
  already-running run still gets its own rejection, and the host is never
  switched to whichever CLI happens to be installed.

- A worker CLI now runs in the rollout workspace even when it resolves its project
  directory from `$PWD` (#371). `worker_env()` set no `PWD`, so the caller's was
  passed through and OpenCode (Bun) worked in the directory `agentdescent` was
  launched from: it never saw the staged candidate, could read the dataset's gold
  labels and the run store, and with the default `build` agent could edit the
  user's real files. `worker_env(workspace)` now sets `PWD` to the workspace for
  both `isolate` values (`extra` can still override it), and `code_runner`'s
  trimmed child environment sets it too.

- A worker CLI no longer inherits the caller's stdin (#372). `cli_agent()` ran a
  worker with `input=None` unless `via_stdin` was set, and `runners._sh` started
  gate commands and plugin hosts with no `stdin=` at all, so under an open pipe
  that never reaches EOF (a CI step, `… | agentdescent evolve spec.json`, a
  notebook kernel, a tool runner) `opencode run "<prompt>"` waited on the pipe and
  every call burned the whole agent `timeout` before ending as `AgentError:
  ... exceeded timeout`. A worker that gets its prompt as an argument now gets
  `stdin=DEVNULL`, as do `runners._sh` commands; `via_stdin=True` still delivers
  the prompt on stdin.

- An isolated `opencode` worker that cannot see a custom provider now says so
  (#373). A custom provider (an OpenAI-compatible endpoint such as Volcengine Ark,
  a local vLLM, a gateway) lives in the user's `opencode.json`, which an isolated
  worker cannot read, so `-m ark/model` failed every call with OpenCode's opaque
  `UnknownError` while `plan` was `ok`. The `AgentError` of an isolated `opencode`
  worker that exits with `UnknownError` / `ProviderModelNotFound` now carries a
  hint naming the cause and the two fixes (`"isolate": false`, or
  `"env": {"OPENCODE_CONFIG": "/abs/providers.json"}` with a provider-only file);
  the hint is skipped when `OPENCODE_CONFIG` is already set in the worker's `env`
  and for every other CLI. The `plan` warning for an isolated `opencode` that pins
  `-m/--model provider/model` now names custom providers and the
  `OPENCODE_CONFIG` recipe. `docs/plugins.md`, `docs/agents.md` and the skill no
  longer say `extra_args` pins an OpenCode model "without giving up isolation";
  they document the recipe, and that an `OPENCODE_CONFIG` exported in the
  launching shell is dropped for an isolated worker, so it belongs in `env`.
  Providers are not copied out of the user's `opencode.json` automatically: that
  file also holds inline keys, MCP servers and plugins that isolation keeps out.

- Candidate code in a container sandbox now sees `PWD=/work`, matching its actual
  working directory (#377). `docker`/`podman exec -w /work` moves the process but
  leaves `$PWD` as whatever the image or the `run` set, so a CLI that finds its
  project from `$PWD` could look past the staged candidate. `exec_prefix()` now sets
  `PWD` to the container workdir for Docker and Podman, never the host workspace
  path; the engine CLI still gets the host environment and the
  candidate environment allowlist is unchanged.

- `status` (CLI and MCP) now reports `pending=1 phase=merge` while the merge step
  runs (#380). Conflict resolution, the fusion tournament and the held-out gate
  were invisible to `pending_calls` / `phase` (#358 only covered the `propose` and
  `run` actors), so a reflective merge blocked on one model call for minutes read
  as `calls` standing still with nothing in flight, and a host polling `status`
  could not tell it from a wedged run. `Meter.timed("merge_gate_seconds")` now marks
  its block in flight on both `evolve()` and `async_evolve()`. The shipped
  `SKILL.md` tells the host what `phase` means, to judge liveness from `status`
  rather than `ps`, and to keep shell waits under a minute.

- `plan` no longer reports `ok: true` for a spec whose `evolve` block names an
  object-valued `evolve()` parameter as a string (`strategy`, `aggregator_factory`,
  `run`, `propose`, `staleness_policy`, …), which `evolve` then crashed on in round
  setup with `'str' object has no attribute 'initial'`; an unknown `evolve` key is
  refused the same way, naming the valid knobs. A spec holds JSON, so objects come
  through `policies` / `agent` / `reflect`, not `evolve` (#384).

- `plan` now also refuses `evolve.allocator` (it is annotated `Any` but is an
  object) and `evolve.extra_artifacts` (values must be `EvolvingArtifact`
  instances), and `kind: policy_slot` converts `evolve.agg_config` through
  `build_agg_config` the same way every other kind does -- without that, `plan`
  said ok and the run died on `'dict' object has no attribute
  'audit_drain_per_step'` (#385 review / #384).

- `plan` again accepts `evolve.parallel: null` (the supported engine default) after #385's object-knob guard rejected every value for that key; `null` on an object-typed evolve knob (including `agg_config`) means "use the engine/compose default" and no longer clears objects `compose` assembled (`#392`).

- `agentdescent install` no longer writes an MCP launcher that breaks when the host
  is started in a directory containing an `agentdescent/` folder (#394). `python -m`
  puts the host's cwd first on `sys.path`, so that folder shadowed the installed
  package and the server died on import, which the host shows only as `Connection
  closed`. The pinned launcher is now `<python> -P -m agentdescent.cli mcp` on
  Python >= 3.11 and the absolute path of the `agentdescent` console script on 3.9
  and 3.10 (only a script that actually runs this interpreter; plain `-m` otherwise, and `install` then prints a WARNING that the launcher still honours the host's cwd); `-I` is not used because it would
  hide `pip install --user` packages. `install` recognises the old, `-P` and
  console-script shapes as its own, so reinstalling upgrades an existing config in
  place without duplicating it, and `uninstall` removes any of them. When the server
  cannot be imported, `agentdescent mcp` now prints one line (naming the directory
  that shadows the package, when that is the cause) and exits 3, instead of a second
  traceback from `mcp_unavailable()`.

- A reflector that raises on every call no longer ends a run as `done` with nothing
  recorded (#395). `propose` exceptions used to be folded into the worker failures
  after that worker's rollout had already counted as a success, so `rounds.jsonl`
  showed only `considered: 0` and the cause appeared nowhere. Each round now carries
  `propose_errors` (calls that raised) and `propose_error` (the first, as
  `Type: message`), `status` / `watch` / `show` and the MCP `status` / `show`
  payloads show them, and `evolve()` warns when any occurred. When every proposal
  raises in two consecutive rounds (`evolution.PROPOSE_FAIL_ROUNDS`) the run stops
  as `state: failed` with `error: reflector failed on every call in rounds N-M:
  <first exception>`, and `agentdescent evolve` exits non-zero; a reflector that
  fails only some of the time still finishes, and rounds with no `propose` call
  neither count nor reset the streak. `agentdescent init` now warns when the
  starter `reflect` is `openai_compatible` and `OPENAI_API_KEY` is unset, and the
  `plan` warning says what will happen. The barrier-free `asynchronous=True` path is unchanged.

- `agentdescent apply` (CLI and MCP `apply`, both through `cli.apply_payload`) no
  longer reports every file as written, and no longer leaves a `.bak-N`, for a run
  that changed nothing (#396). The evolved tree is compared with the target file by
  file: `written` lists only files whose content differs (new, changed, or a script
  that lost its exec bit), identical ones are in the new `unchanged` list, and the
  backup is made only when something is written or pruned. When nothing would
  change the result is `{"written": [], "unchanged": [...], "backup": [], "noop":
  true}` and the CLI prints `nothing to apply: <dest> already matches run <id>` on
  stderr (stdout stays the JSON document), exit 0. A single-file (`text`) target
  gets the same treatment. `show`'s `apply would:` and `apply --dry-run` use the
  same judgement (`EvolutionResult.write_to` now returns `unchanged` and `noop`
  too, and only rewrites the files that changed). The SKILL.md step and the MCP
  `apply` description say what `noop` means.
  Files are compared and written verbatim (CRLF included), a script counts as
  unchanged only with both user and group exec bits, the `text` dry-run result also
  carries `written`, and an artifact with no files is a no-op that creates nothing.

- `--store <relative path>` now works for `resume` and every other detached launch
  (#400). The launcher runs with the run directory as its cwd and was handed the
  still-relative `log.txt` path, so `agentdescent --store ./store resume <id>` died
  with `could not start the run: ... FileNotFoundError: './store/<id>/log.txt'`
  while an absolute store worked. `RunDir` now always holds an absolute path.

- A literal `~` in `--store` or `$AGENTDESCENT_HOME` is now expanded everywhere
  (#402). It arrives unexpanded from an MCP host config's `args`/`env` or a quoted
  `--store '~/st'`; `create` already landed the run under `$HOME` (via `RunDir`,
  #400) but `status`/`list_runs` looked in `./~/st`, said "no runs", and left a
  stray `~` directory in the cwd, and `AGENTDESCENT_HOME=~/x` put every run under
  `./~/x`. The store base is now expanded and made absolute once
  (`runstore.store_path`), so the CLI, MCP `status`'s `store`, and `doctor`'s
  `run_store` all report the same absolute path.

- Codex scrubs the environment of an MCP server, so the AgentDescent server it
  started saw no provider key and no `AGENTDESCENT_HOME`. `install codex` now
  writes an `env_vars` line (the provider keys, `ARK_API_KEY` and
  `AGENTDESCENT_HOME`) into `[mcp_servers.agentdescent]`, and the plugin's
  `.mcp.json` carries the same key for the `codex plugin add` route. Re-running
  `install codex` repairs a block written before this, `uninstall codex` still
  removes both forms, and `doctor` reads the old block as `stale` (#404).

- A spec-driven run (`evolve`, `evolve --detach`, MCP `start`) now counts the
  tokens its spec-built model adapters report: the run's `Usage` reaches every
  `openai_compatible` / `claude` (or any factory with a `usage=` parameter) named
  in `agent`, `reflect` or `policies`, as a token-only view
  (`Usage.token_sink()`), so `status` / `rounds.jsonl` show real `tokens` and
  `evolve.max_tokens` stops the run (`stop_reason="max_tokens"`) instead of being
  accepted and never firing. `calls` still counts one per actor invocation, so
  `max_calls` and `usd_per_call` budgets are unchanged. The Python API
  (`openai_compatible(usage=u)` with `evolve(usage=u)`) behaves as before (#405).

- `plan`, `evolve` and MCP `start` now check the variable an `openai_compatible`
  reflector/agent will actually read -- the block's own `api_key_env` (and name
  `base_url_env` in the hint) instead of a hard-coded `OPENAI_API_KEY`. A spec
  with `"api_key_env": "ARK_API_KEY"` no longer gets a false "needs
  OPENAI_API_KEY" alarm when `ARK_API_KEY` is set, and a missing `ARK_API_KEY` is
  no longer hidden by an unrelated `OPENAI_API_KEY`. The `claude` branch does the
  same, and `claude()` now really accepts `api_key_env` (it used to forward it to
  `messages.create` and fail on the first call). String refs and default names
  warn exactly as before (#406).

- `install`, `uninstall` and `host_status` with an explicit `--home` no longer let
  an exported `CODEX_HOME`, `DSH_HOME` or `XDG_CONFIG_HOME` redirect them to the
  real configuration: `--home` is now the host's whole world (a probe cannot edit,
  or remove, your real install). Without `--home` the variables apply as before
  (#410).

- `install`, `uninstall` and `host_status` refuse an explicitly empty `--home`
  (`--home ""`, which is what `--home "$UNSET_VAR"` expands to) with a clear error
  and exit code 2 instead of falling back to the real home and its
  `CODEX_HOME` / `DSH_HOME` / `XDG_CONFIG_HOME`, so a probe with a typo can no
  longer edit or remove your real install. Omitting `--home` is unchanged (#412).

- `plan` / MCP `plan` now notes the condition `evolve()` warns about — a held-out
  set under 4 tasks — from the same computation (`held_out_size`, `MIN_HELD_OUT`
  in `agentdescent.evolution`) instead of a separate `4 <= tasks < 8` rule. 8–11
  tasks at the default `held_out_frac=0.3` hold out 2–3 and used to plan clean;
  the note names the held-out count, both fixes (more tasks, or a higher
  `evolve.held_out_frac`) and the smallest task count that works (12 at 0.3). A
  detached / MCP `start` run records the note in `status`'s `notes` from the first
  poll (and `start` / `evolve --detach` return it), `agentdescent status` prints
  it, and a foreground `agentdescent evolve` shows one plain `warning:` line
  instead of a source-line `RuntimeWarning`. The SKILL and quickstart now say
  "12 to 20 cases" rather than 8 (#414).

- `agentdescent install dsh` and the native `dsh-agentdescent` plugin now forward
  the same environment list as Codex (`FORWARDED_ENV`: `DEEPSEEK_API_KEY`,
  `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
  `ARK_API_KEY`, `AGENTDESCENT_HOME`) instead of four keys, so `ARK_API_KEY`, which
  dsh scrubs unless it is declared, reaches the MCP server. There is one tuple
  for every host (`DSH_FORWARDED_KEYS` is gone). Re-running `install dsh` rewrites
  an older four-key block, `doctor` / `host_status("dsh")` report it as `stale`,
  and `install dsh` prints the new list. `docs/plugins.md` lists the names and
  notes that a custom `api_key_env` must be added by hand (#415).

- `agentdescent --help` no longer prints `run  ==SUPPRESS==`, and neither the help
  nor the usage line shown on every argparse error lists the internal `run`
  worker (what `evolve --detach` spawns) next to `evolve`; argparse does not
  honour `help=SUPPRESS` on a sub-parser. `agentdescent run --run-dir ...` still
  works, and `run --help` now says it is internal (#420).

- `openai_compatible` and `anthropic_compatible` no longer put a connection-shaped
  keyword (`base_url`, `api_base`, `api_url`, `base`, `url`, `endpoint`, `api_key`,
  `apikey`, `key`) into the request body. A spec that wrote `"base_url": "https://..."`
  was accepted, `plan` said `ok`, and every call went to `api.openai.com` carrying the
  other provider's key. It is now refused by `plan`, `evolve` and MCP `start`, naming
  the spelling that exists (`default_base_url` / `base_url_env` for the endpoint,
  `api_key_env` for the key); other body extras such as `temperature` are unaffected.
  `plan` also warns when `openai_compatible` would send a non-OpenAI `api_key_env` to
  `api.openai.com` because no endpoint is configured, and `docs/plugins.md` documents
  the three connection parameters (#422).

- `doctor` (CLI and MCP) now reports every provider key the installers forward --
  the key names of the shared `FORWARDED_ENV` list, `ARK_API_KEY` included -- in
  `provider_keys_present`, and raises "no provider key in the environment" only when
  none of them is set. Before, it checked three hard-coded names, so a machine with
  only `ARK_API_KEY` exited 1 and told the host there was no key. When the only key
  present is not OpenAI's or Anthropic's, `doctor` adds a hint to use it through
  `openai_compatible` with `api_key_env` and `default_base_url` / `base_url_env`
  (endpoint and model from the user). The shipped `SKILL.md` gains an "Other
  providers" paragraph saying the same, so a host never has to read the package
  source (#423).

- `agentdescent init` no longer guesses a model: the `openai_compatible` reflector
  (and the `kind: text` / `policy_slot` agent) is written with the key that is set
  here (`"api_key_env": "ARK_API_KEY"` when only a non-OpenAI key is present) and a
  `<your-model>` / `<your-endpoint-url>` placeholder, or the values from the new
  `--model`, `--base-url` and `--key-env` flags. Its warning names that key, `plan`
  and `evolve` refuse a spec that still holds a placeholder, and `plan`'s
  missing-OPENAI_API_KEY warning lists the provider keys that are set (#451).

- `agentdescent demo` now writes a valid Agent Skill: the generated
  `csv-total/SKILL.md` carries `name` / `description` frontmatter, so a real run on
  the demo skill keeps the SKILL.md structural gate on (no more `no-frontmatter`
  note from `plan` / `evolve --detach` / `status`) and hosts such as OpenCode,
  Claude Code and Codex actually load the skill. The offline agent and reflector
  are unchanged.

- `init` and `doctor` now name the OpenCode Coding worker recipe when only a
  gateway key such as `ARK_API_KEY` is set. `init` used to write a silent
  `{"ref": "opencode"}` (every isolated call then failed on OpenCode's built-in
  Volcengine default) and `doctor` only mentioned the `openai_compatible` reflect
  route; both now point at `docs/plugins.md` (`-m` plus `env.OPENCODE_CONFIG`).
  The Ark-specific wording (Volcengine default, `ark/deepseek-v4-flash`) appears only
  when `ARK_API_KEY` is set; any other gateway key gets a generic
  `<provider>/<model>` recipe, and `init` and `doctor` share one predicate, so
  they also agree when `OPENAI_API_KEY` is set alongside a gateway key.
  The `plan` half of this landed earlier in the same issue.

- `plan` / `evolve` no longer call a provider key "untested" for an isolated
  OpenCode worker. With no `-m` and only a gateway key such as `ARK_API_KEY` set,
  the warning now says OpenCode would fall back to its own default model for that
  provider (measured: an unactivated Volcengine default, every call failed) and
  gives the documented fix, `-m ark/deepseek-v4-flash` plus
  `env.OPENCODE_CONFIG` pointing at a provider-only file. That documented recipe
  itself (a pinned `provider/model` and `OPENCODE_CONFIG` or
  `OPENCODE_CONFIG_CONTENT` in the agent's `env`) no longer gets the "a login does
  not carry over" warning at all.

- `agentdescent init --agent <ref>` is no longer silently discarded for
  `kind: text` and `kind: policy_slot`: an explicit `--agent host_model` (or any
  other plain model ref) is written as given instead of being replaced by
  `openai_compatible` + `deepseek-v4-flash`. When the ref cannot work for the kind,
  `init` now refuses with exit 2 and writes nothing, using the same rules `plan`
  enforces: `kind: text` refuses file-editing agents (`claude_code`, `codex`,
  `dsh`, `opencode`, `cli_agent`), and `skill_dir` / `agent_dir` refuse plain model
  calls (`host_model`, `openai_compatible`, `claude`, `echo`). Omitting `--agent`
  keeps the old defaults.

- `agentdescent show` (CLI and MCP `show`) on a run without a result -- `failed`,
  `cancelled` or still `running` -- no longer returns only `note: no result yet`.
  It now also reports what is on disk: the round count and a short per-round
  `history` (held-out reward, committed/rejected, `propose_error`), the run's
  `best_reward` and `propose_errors`/`propose_error`, and the lowest-reward
  recorded rollouts as `samples` (output vs gold, same 4 KB cap). The `note` is
  kept and there is still no `diff`/`apply_plan`. A run with nothing recorded
  is unchanged.

- A reflector call through `openai_compatible` or `anthropic_compatible` that runs
  out of `timeout` no longer surfaces as a bare `TimeoutError: The read operation
  timed out`. It now raises a `RuntimeError` naming the endpoint, the model and the
  seconds, and how to fix it (`"stream": true`, a higher `"timeout"`, or turning the
  provider's thinking off), so `status` / `propose_error` say what to change. Read
  and connect timeouts are both covered; `with_retries` still retries them. The
  `openai_compatible` docs and the host skill now say reasoning models need
  `"stream": true`.

- `agentdescent audit` on an unknown run id now prints an unquoted
  `error: no archived run ... to audit` and exits 2, like the other unknown-run
  errors, instead of a quoted `KeyError` repr with exit 1 (#481).

## [0.6.0] — 2026-10-16

### Added

- **`Ledger.fork` / `live_heads` / `discard_head` — multiple live heads (#75).**
  The ledger held one `dev` branch; `PopulationAggregator` serialised candidates
  by rewriting it, so no two workers could expand different candidates at once.
  `fork("head/0")` creates an independent branch with its own version vector;
  `live_heads()` lists `dev` plus all `head/` forks; `discard_head` cleans up.
  `Candidate.branch` and `EvidenceCard.branch` carry the fork name.
  `PopulationAggregator.step(n_workers)` forks a `head/<slot>` per selected
  candidate and `head_for_worker(wid)` distributes workers across them.
  `_commit_with_retry` commits to `EvidenceCard.branch`; the sync and async
  worker loops snapshot from `head_for_worker` instead of hardcoded `dev`.
  `_decide` passes the candidate's own branch through to the commit (and the
  CAS-rebase path rebases against that branch, not `dev`), so a fork candidate
  updates only its fork. `finalize` discards all forks and commits the best
  candidate to `dev`.
  Default: `SingleHead` + no forks = old behaviour exactly. 12 tests.

- **Value-directed budget allocation — `evolve(allocator=...)` / `async_evolve(allocator=...)`.** `BudgetGovernor` degrades optional spend (fusion tournaments, the self-verify rollout) against global floors: once `spent / max_tokens` crosses a threshold, *every* candidate stops getting the expensive part. The EvoAlloc line of work measures the cost of that fixed allocation when evaluation is the scarce resource — this replaces it with a **per-candidate** decision learned from the run's own merge history.

  **`ValueBudgetAllocator`** (`agentdescent/allocator.py`) is a learner with two halves. **Value-directed allocation**: once the token budget is tight, the self-verify rollout is spent only on candidates the value model expects to commit — a `LearningValueModel`, a tiny online logistic fit over the **pre-spend features the worker actually decides on** (the rollout's group-relative advantage, the diff's size, the artifact's blast radius), with a logit clip and L2 so one outlier cannot blow the weights. The model is trained on exactly that vector — post-outcome signals (delta, P(improve), stable distance) are kept for the audit trail but never enter `AllocatorContext.features` — so it cannot fit the labels with signals the decision never sees. **Counterfactual exploration**: a candidate the model judged unworthy is skipped — except with epsilon probability, when it is evaluated anyway; the outcome is then ground truth the model was wrong about, and it learns from it. Without this term a value-based allocator only ever sees the candidates it already believed in, and a biased prior is never corrected — the selection-bias collapse this term exists to prevent.

  Wired end to end: the worker asks the governor `candidate_worth(ctx)` before the self-verify rollout; the aggregator reports **every** measured decision exactly once (`observe(ctx, committed)`), with oracle rejection and acceptance rejection as negative labels — a model fit on commits alone never leaves warm-up. **Every commit is attributed to a worker's actual pre-spend vector**: a fused candidate keeps its first contributor's context, and a tournament-selected single proposal is matched to its own winning card — never a losing first entrant, never a fused union size. The feature vector is **scale-normalised**: the raw-op diff size (1..100+) would swamp the advantage (a z-score) and blast radius in a raw logistic fit, so `size` is mapped through `log1p` to `[0, 1]` — verified by a regression where a high-advantage, large-diff candidate still predicts commit. `allocator=None` (the default) is the old behaviour byte for byte — the clock thresholds remain the *ceiling* the allocator may never exceed. 25 tests in `tests/test_allocator.py` (incl. negative-outcome, sync/async feature-parity, end-to-end fused + single-winner regressions, the A/B arm wiring, and the size-scale regression). **`bench/ab_allocator.py`** is the equal-budget A/B (off vs on, real datasets HotpotQA/finer, per-seed quality + measured tokens + a fired check that distinguishes "the allocator helped" from "it never ran"); exported from the package root; `docs/api.md` regenerated.

- **`async_evolve(resume_queue=...)` — the L-traj resume half, the straggler
  re-run.** Detection is **after-return**: `run(rendered, task) -> output` is
  opaque, so a rollout that overruns its predicted cost is noticed once it has
  finished (it is not cancelled mid-flight) and counted
  (`result.stragglers`). With a shared
  :class:`~agentdescent.scheduler.ResumeQueue` passed in, it now *records* the
  detected task and the version it measured instead of dropping it; the next
  idle worker whose shard owns that task re-runs it against the **current**
  head. A straggler measured version N, its re-run measures version N+k, so the
  pair is a free cross-version A/B signal — the design's L-traj note made
  behaviour.

  `ResumeQueue.pop_for(task_ids)` hands a queued resume only to a worker whose
  shard owns the task, so no two workers re-run the same one. `result.resumed`
  reports the re-runs. `resume_queue=None` (the default) is the old behaviour
  exactly: stragglers are counted and dropped.

  **Re-queue is bounded** (`ResumeQueue(max_attempts=...)`, default 1): the
  queue counts how many times each task has been resumed, and `push` refuses a
  task past the cap — so a chronically slow task gets exactly `max_attempts`
  re-runs instead of an infinite resume loop starving the worker's fresh tasks.

- `CladeThompson` selection policy (#276): HGM's expansion rule, Thompson
  sampling on clade metaproductivity (node plus descendants via
  `Candidate.parent`), with an opt-in budget-driven cool-down and
  seeded draws; `clade_counts()` exposes the `(S, F)` evidence. Ports the
  paper's formula without upstream's in-place evidence accumulation.
- Add the RRSI component vocabulary (`examples/rrsi/components.py`) and per-edit
  history `L_t` (`examples/rrsi/history.py`: yield, prune set, stall
  exploration) adapted from the pinned Google release, with defensive handling
  of non-finite measurements and torn JSONL lines, including tails cut
  inside a multibyte UTF-8 character (#252). Still not an
  `evolve()` integration; see `docs/algo-rrsi.md`.
- `kind="skill_dir"` now gates candidates structurally before any rollout
  (#246): a diff that deletes `SKILL.md`, drops its frontmatter, or breaks the
  Agent Skills `name` / `description` / `compatibility` rules (or renames the
  skill) is rejected and counted in `FileTree.structural_rejections` by reason.
  New `agentdescent.skillspec` (`validate_skill_tree`, `SkillTreeValidator`,
  dependency-free) and a general `FileTree(validators=...)` hook. Skipped with
  a note when the initial skill does not pass; opt out with
  `validate_skill: false`. The gate also covers model-written merges:
  `ReflectiveFusion` asks the strategy's new `FileTree.check_candidate` before
  returning a synthesised union, and a refused one falls back to ranking. Header
  values a YAML loader would refuse or read as non-text (unterminated quotes,
  invalid `\` escapes, text after a closing quote, `[...]` / `{...}`, unquoted
  `: `, reserved first characters, and plain `null` / `true` / number / date
  values) are `frontmatter-malformed`; a BOM and inline `# comments` (even ones
  containing a colon) are tolerated.
- `Diff.rationale` carries the FileTree reflector's reason into the ledger
  commit message and the ReplayAwareProposal rejected-evidence note (#247).
  `parse_proposal` returns edits plus rationale; `parse_edits` stays a thin
  wrapper. Fused/synth diffs concatenate parent rationales (deduped, capped).
  Rationales never enter `diff_id`. Control characters (NUL included) and
  lone surrogates are stripped from rationales so `git commit` cannot crash on
  them, and `parse_proposal` is exported from the package root.

- Add the RRSI cosine-annealed L0 edit-budget schedule (`examples/rrsi/schedule.py`)
  adapted from the pinned Google release, with endpoint / float-guard / bound
  regressions (#238). Still not an `evolve()` integration; see `docs/algo-rrsi.md`.

- Add an offline RRSI round-selection mechanism fixture adapted from Google's
  pinned release, with decision regressions and explicit validation boundaries.
  This is not yet an `evolve()` integration or full RRSI port; see
  `docs/algo-rrsi.md`.

- **Rollout trajectory ledger.** `evolve(record_trajectories=True)` (and
  `async_evolve`, and any spec through the `evolve` dict) now persists every
  rollout, self-verify re-run and gate evaluation as a versioned `Trajectory`
  under `<repo_path>/trajectories/` -- the exact rendered artifact text the run
  provider saw, the task, the output, the reward, and the artifact version and
  environment it all happened against. Records are content-addressed on the
  same identity the evaluation cache keys on (`rendered`, `task`, environment),
  so re-runs dedupe instead of piling up; the exists-check + write pair is
  locked, so a single process's racing writers count exactly one record and
  the rest as deduped. The boundary record (rendered, output, reward) is always
  written when recording is on; event *detail* text is metadata unless
  `AGENTDESCENT_RECORD_TEXT=1` or a store built with `record_text=True`. A
  failed write is counted in the meter and never fails a rollout.
  The store is the durable base the learning-type increments (process
  supervision, offline re-weighting, audit replay) read; worker-produced
  `EvidenceCard`s now carry the `trace_id`s of the records that justify them.
  New counters: `trajectories_recorded`, `trajectories_dedup`,
  `trajectory_write_seconds`, `trajectory_write_failures`; new
  `EvolutionResult.trajectories_path`. Off by default -- a run without a store
  reports zeros and behaves exactly as before.
- **Governance invariants.** `evolve(contract=Contract(invariants=(...)))` (and
  `async_evolve`) declares machine-checkable rules every diff to the artifact
  must satisfy -- ``mutable_keys``, per-commit op/char caps, a non-decreasing
  ``blast_radius`` (an artifact may not relax its own governance layer), a
  no-self-dependency rule, or a custom predicate resolved as a ``workspec.Ref``.
  The aggregator enforces them **before** a candidate is measured, so a
  policy-violating diff is rejected with a ``violates-invariant`` outcome
  instead of buying a held-out sweep; a rule that cannot run (an unknown kind,
  a broken predicate) is a violation, never a silent skip. The contract rides
  in the ledger payload so invariants survive the round-trip (an artifact whose
  rules evaporate in storage is a governance rule that silently stopped
  running); registration-time dependency-graph checks refuse self/cyclic
  ``depends_on`` (duck-typed, so they light up with multi-artifact deps). New
  meter counters ``invariant_checked`` / ``invariant_rejections`` and a
  ``violates-invariant`` outcome; an artifact that declares nothing reports
  zeros and is the untouched path exactly.

- Changelog fragments (#295): a PR now adds one file `changelog.d/<number>-<slug>.md`
  instead of editing `CHANGELOG.md`, so open PRs stop conflicting on the same
  lines. `python -m tools.changelog_fragments check` validates them (and runs in
  `pytest -q`); `release [--version X.Y.Z --date YYYY-MM-DD]` folds them into
  `[Unreleased]` by category and optionally cuts the release section. See
  `changelog.d/README.md`.

### Fixed

- `audit.AuditStore` keeps its crash-safety contract (#274). A torn tail left
  by a killed process gets a newline before the next append, so the record
  written after the crash is no longer glued onto the fragment and lost; the
  file is read as bytes and decoded per line, so a tear inside a multi-byte
  character, a line that parses to a non-object, or a side-record missing its
  keys costs that one line (counted in `corrupt`) instead of making the store
  unopenable. Lines are encoded with `allow_nan=False` before the file is
  touched and memory is updated only after the write returns, so a failed
  `append` / `resolve` / `reopen` / `remember_*` leaves the index agreeing with
  a fresh reload. `observe_unlabelled` and `resolve` refuse non-finite scores
  with `ValueError`, and `load` counts a line holding a non-finite number (the
  `NaN` / `Infinity` tokens older builds wrote, or an overflowing literal) as
  corrupt instead of loading it. `resolve_from_mapping` refuses a batch with a
  non-numeric or non-finite answer before anything lands, naming the offending
  ids, instead of reporting it as already resolved. An unknown
  `schema_version` still raises.
- `FlatPuct` / `CostEfficient` / `MCTS` / `Beam` no longer crash or silently
  poison `select()` on bad knobs (#293). `CostEfficient.cost_factors` and
  `FlatPuct._priors` used to raise `OverflowError` on a steep
  `cost_exponent` / `prior_exponent` against a wide ratio; both now use a
  safe power that returns `+inf` instead of raising, so an expensive
  candidate loses exploration rather than aborting the round. Non-finite
  `c_puct` / `prior_exponent` / `exploration` / `cost_exponent` are refused
  with `ValueError` at construction (a negative `cost_exponent` still clamps
  to 0 = ignore cost). `Beam(k=...)` now requires an integer `>= 1`, so
  `nan` / `1.5` / `True` no longer construct and then `TypeError` on the
  first slice.

- `sigmoid_novelty_weights` / `Archive(sampling='sigmoid_novelty')` no longer
  raise `OverflowError` on large negative scores, and a non-finite score no
  longer poisons the weight total (#286). The sigmoid is evaluated in a
  numerically stable form (large negatives contribute ~0, matching `-inf`);
  non-finite scores contribute weight 0, matching this mode's unscored
  convention. Scores in the normal `[0, 1]` band stay bit-identical to the
  previous formula so DGM port equivalence is unchanged.

- `runstore` rounds/status no longer break on a non-finite reward or a torn
  `rounds.jsonl` line (#271). `append_round` encodes with `allow_nan=False` and
  writes non-finite floats as `null`; `rounds()` skips malformed lines instead
  of raising; `on_round` tracks only finite rewards for `best_reward` /
  `last_reward` so `max(best)` cannot become order-dependent on `NaN`; the
  loopback panel dumps with the same sanitize + `allow_nan=False` contract.
  `append_round` writes a newline first when the file ends in a torn fragment,
  so the round written after a crash is not glued onto it; `rounds()` decodes
  per line so a tear inside a multi-byte character costs one line; text
  `status <run_id>` / `watch` print a `null` reward as `n/a` instead of raising.

- HuggingFace row-cache filenames sanitize split names and bound their
  components (#261), keeping slash-containing and traversal-style splits in
  one portable cache entry. Lossy conversions include a digest to distinguish
  split names (a bounded config is digested from the raw config, so configs
  that slug identically stay distinct), and a versioned suffix prevents old raw-split cache entries
  from aliasing the new names. Existing normal filenames and datasets-server
  requests are unchanged.

- `TrajectoryStore.record()` no longer raises when a record cannot be encoded
  (a NaN inside a structured output, non-string dict keys, a circular value, a
  `__str__` that raises, a non-string event detail) (#259). It is counted in
  `trajectory_write_failures` like a full disk and nothing is written, so
  `eval_one` no longer mistakes it for a backend transient and re-runs the
  already-paid rollout `ATTEMPTS` times before failing the gate.

- `FileCache` write failures no longer leave orphan `*.tmp` siblings, and
  encode errors (`TypeError`/`ValueError`) are swallowed like `OSError` so a
  paid evaluation is never failed by the cache; non-finite scores are returned
  but not written (`allow_nan=False`). Non-JSON-native keys (`Path`, `UUID`)
  still persist via `default=str`, matching how `_path()` hashes them (#263).
- A failed `Ledger` write no longer corrupts the ledger (#265). Payloads are
  encoded before any file is touched, each file is replaced via a temp
  sibling, and an operation that fails part-way restores the files it already
  wrote, so a serializer refusal or a full disk can no longer leave a torn
  `artifacts/<id>.json` (or half of a `commit_atomic`) for the next commit's
  `git add -A` to sweep in and break every later `snapshot()`.
- `runstore._write_json` encodes before creating a sibling tmp and always
  removes that sibling in `finally`, so a serializer refusal or a failed
  replace can no longer leave an orphan `status.json.<uuid>.tmp` /
  `spec.json.<uuid>.tmp` in a run directory (#267).

- `meta.cached_completion` is best effort and never costs the caller a paid call: a cache-write failure (`OSError`, unencodable or non-`str` response) returns the response uncached and leaves no `*.partial` sibling, the entry is encoded before the disk is touched, and a corrupt or non-`str` entry (including a JSON list, which used to raise `TypeError`) is counted once, as a miss, instead of as both a hit and a miss (#269).

- `TrajectoryStore.get` / `path_for` round-trip `trace_id`s whose `artifact_id` contains a colon by parsing with `rsplit(":", 1)` (#257). The key is always a hex digest; a leading split treated a namespaced id as part of the key and returned ``None`` for a record that was on disk.

- `dataloader` cache entries are written atomically (temp file + `os.replace`),
  read back as bytes instead of locale-decoded text, and refetched when they do
  not parse; a fresh non-JSON response is parsed before caching so it can no
  longer poison `hf_rows` / `hf_rows_modelscope` (#255).

- Container sandbox allowlisted env vars are forwarded as `--env NAME` so
  credential values no longer appear on the `docker`/`podman` argv (`ps`)
  (#244). Non-allowlisted names and `--env-file` remain forbidden. Allowlist
  entries ending in `*` are rejected before forwarding: Podman expands bare
  `--env PREFIX*` to every matching host variable, which would leak
  non-allowlisted credentials.

- `CallBudget` default floor is capped at `base`, so a configured ceiling below
  256 can no longer be silently raised above what the caller agreed to (#234).
  Explicit `floor > base` and non-positive `base`/`floor` raise `ValueError`.

- `hf_rows_modelscope(cache=False)` now bypasses an existing on-disk cache and
  does not write the fresh response back (#229). `fetch_bytes` / `fetch_text`
  accept the same `cache` flag so the raw-URL helpers share the `hf_rows`
  bypass contract.

- `AdaptiveTrustRegion` widens when a window is dominated by `OVERSIZED`
  refusals (the size cap was binding) instead of treating them like gate
  refusals and leaving the region stuck (#226). A batch that mixes oversized
  diffs with stale or in-range ones now also counts as `oversized` for the
  region unless it committed or was reversed; `MergeReport.oversized` records
  how many diffs the size cap refused.

- Allocator A/B comparisons now use the same fully scored seed pairs for
  quality and token summaries, retain spend for excluded runs, and seed each
  treatment allocator from its run seed. The runner rejects invalid or
  duplicate seed lists before setup and scopes allocator diagnostics to the
  pairs included in the comparison.

- The `OVERSIZED` merge report now quotes the trust region that actually
  rejected the diff (the adaptive policy's current region) instead of the
  config constants (#224).

- `BudgetGovernor`'s diminishing-returns stop skips a non-finite (`NaN`/`inf`)
  round reward like `None` instead of letting it poison the peak efficiency, and
  no longer misfires or crashes when `min_efficiency_samples < 2` (#242).

- `PopulationAggregator` records `Candidate.parent` and backpropagates
  subtree `selected` counts under `evolve()` (#275). Selecting a candidate
  remembers its version as the branch tip; the next new state admitted from
  that branch is filed as its child, and `selected` increments on the node
  and every ancestor (cycle-guarded), so `FlatPuct` / `CostEfficient` /
  `MCTS` see a real tree instead of a flat bag. Lineage survives
  checkpoint/restore; pre-change checkpoints restore as roots. Archive ids
  are unique even when two forks commit different states at the same
  ledger version, so one fork's selection never bumps or parents the other;
  restoring an older checkpoint that already holds such a collision gives
  the later row a fresh id (saved parent references keep the first).

- `CladeThompson.tau()` reads `budget_remaining` like `CostEfficient` (#280):
  `+inf` (no ceiling) and NaN mean a full budget (`tau = 1`) instead of an
  exhausted one (`max_tau`); the power is taken in log space and capped, so a
  small fraction with a steep `cool_down_beta` no longer raises
  `OverflowError` mid-`select()`. Non-finite `cool_down_beta` / `max_tau` are
  rejected with `ValueError`.

- `AdaptiveTrustRegion` moves each cap by at least one unit per widen or
  tighten (#282). With plain `round(cap * factor)`, the default `widen=1.25`
  mapped an op cap of 2 to `round(2.5) == 2` (and 1 to 1), so two reversals
  from the default 6 left the op cap stuck for the rest of the run while
  `chars` and `history` kept moving. A tighten no longer clamps a cap that sits
  below `minimum` *up* to it, so a rollback can never loosen the region.
  `maximum` stays a hard ceiling. `window < 1` (which raised
  `ZeroDivisionError` on the first `observe`), `widen < 1`, `tighten` outside
  `(0, 1]`, non-finite factors, `accept_rate_to_widen` outside `[0, 1]`, and
  non-positive bounds are rejected with `ValueError` at construction.

- `selection.Archive` no longer crashes at small temperatures (#284).
  `performance` / `novelty` computed `exp(score / temperature)` raw, so below
  about `temperature = 1/709` `select()` raised `OverflowError`. Just above
  that, a few hundred near-top candidates summed to `inf` and
  `random.choices` refused the total. The raw weights are still used,
  bit-for-bit, whenever they are representable, so seeded draws do not move.
  When they overflow, or their total is not a positive finite number, the
  max-shifted softmax (the same distribution) is used instead. A `NaN`,
  infinite or non-numeric `temperature` is now rejected with `ValueError` at
  construction rather than failing at the first `select()`.

- `rewards` reads numbers as they are written (#288). `last_number`,
  `numeric_close` and `command_scorer` used to take `1.5e-3` as `-3`,
  `6.02e23` as `23`, `.5` as `5`, `pages 3-4` as `-4` and `1,2,3` as `123`;
  the parser now understands exponents and leading decimal points, groups
  commas only in threes, and treats a minus glued to a preceding digit or
  letter as subtraction. An overflowing figure (`1e999`) is no longer read as
  `inf`, so `command_scorer` raises `GraderError` instead of clamping it to
  `1.0`.

## [0.5.1] — 2026-10-10

### Release summary

- Agent-first editorial website and plugin quickstart, modular-framework
  explanations, unified documentation styling, and an illustrative in-browser
  workflow that makes no model calls or file changes. The optional one-command
  setup installs current main and connects supported agent CLIs on PATH; it
  is not a pinned-release installer.
- Token budgets, cost-aware selection, graceful budget degradation and optional
  diminishing-returns stopping; checkpointed search state survives a restart.
- Sparse verifier auditing and calibration, opt-in priority-ordered audit drain,
  and replay-aware sampling and proposals that can use settled evidence.
- Policy-slot evolution, the EvoX Genesis port with skill extraction and read-only
  investigation, and the porous-molecule tree-search example. Research examples
  remain in the source repository rather than the installed wheel.
- More reliable first runs: actionable spec/dataset errors, policy-slot scaffolds,
  fail-fast host installation and a clean installed-wheel CLI/MCP smoke gate.

### Upgrade notes

- Python >= 3.9 remains supported; the MCP extra requires Python >= 3.10.
  NumPy >= 1.20 is now an installed dependency for prediction-powered auditing.
- Use `ThreeLayerVerifier.full_eval` and `full_eval_matches_counts`. The old
  `oracle_eval` and `oracle_shares_full_set` aliases remain available; using
  `oracle_eval` emits a `DeprecationWarning`. Removal is planned for 0.7.
- Checkpointing, replay weighting and audit draining remain opt-in. Token-budget
  enforcement depends on usage reported by the model adapter; it is not a
  provider-side hard spending limit.

The detailed changes below cover the merged commits since 0.5.0. They do not
include the open multi-head, multi-artifact, asynchronous trajectory-resume,
value-directed evaluation or Genesis blind-mode proposals.

### Fixed

- Interactive demo restarts no longer interleave with an older approval run,
  and the displayed artifact stays tied to the run that produced it.
- Keep the interactive demo reachable from the documentation navigation drawer
  on mobile, where the desktop header links are hidden.
- Prevent Material admonition styles from exposing the mobile menu on desktop
  or overriding the editorial disclosures; keep the homepage install command
  readable without splitting its URL. Add rendered responsive-layout CI checks.

- Machine-local host installs now pin MCP and session hooks to the installing
  Python, so they work when a later agent process lacks the pip scripts directory
  on PATH. Stock legacy and previously pinned Codex/OpenCode entries are repaired
  on reinstall, including after a Python environment moves; custom entries and
  distributable plugin templates stay intact.

- Spec and dataset file failures now report a concise `SpecError` instead of a
  CLI traceback. Invalid JSON names the input file and line/column; JSONL uses
  the physical file line, including blank lines. Missing or unreadable specs,
  invalid UTF-8, and non-object spec JSON also produce actionable errors.

### Added

- **`AggregatorConfig.audit_drain_per_step` — the L-value consumer, the audit
  queue's missing reader.** `AuditScheduler.submit()` computes
  `priority = blast_radius × uncertainty / trust` for every merge, but with the
  default `collect=False` nothing queues, and with `collect=True` nothing pops
  the heap — the audit that actually runs is the inline `force_oracle` threshold
  gate, which reads trust but not the ranking. `concepts.md` calls this *"the
  priority queue has no consumer"* and says to treat the ranking as a priority
  *model*, not work in flight.

  `_drain_audit_queue(max_per_step)` pops the highest-priority queued audits
  after each `finish_step`, runs `full_eval` on the stored `(base_state,
  candidate)` pair, and updates trust from the oracle's verdict. The scheduler's
  `collect` flag is forced on when `audit_drain_per_step > 0`. `EvolutionResult.
  audit_drained` reports how many ran. Default `0` = the old behaviour exactly.

- **`ReplaySampler` + `ProposalContext.rejected` — make the settled-evidence pool a consumer.** The aggregator's settled pool (stale / oversized / CAS-conflict evidence cards) was diagnostic-only: nothing read it back. Two consumers now close that loop.

  **`ReplaySampler`** (``agentdescent/sampling.py``) is a ``TaskSampler`` that up-weights tasks whose recent proposals were discarded — a signal that is different from pass rate (a task whose proposals keep going stale is not "easy", it is being out-competed by the parallel scheduler). Additive replay bonus, saturating at ``capped_at`` (default 10), scaled by ``temperature`` (default 0.0 = ``DifficultyWeighted`` identically). Wired via ``aggregator.set_settled_consumer(sampler.settle)`` on both the sync and async paths.

  **`ProposalContext.rejected`** — a new field carrying the most recent settled cards for the task being rolled out. The docstring said "'do not re-propose what was just rejected' is inexpressible" — this field makes it expressible.

  **`ReplayAwareProposal`** (``agentdescent/replay.py``) is a ``ProposalPolicy`` wrapper that reads ``ctx.rejected`` and, when non-empty, appends a summary of the rejected diffs (paths, base version, delta) to the context's ``output`` before delegating. A model-based inner policy that renders ``output`` — the shipped ``LLMAgent.propose`` — naturally sees "these were discarded, do not repeat them".

  ``recent_settled`` on both ``Buffer`` and ``Aggregator`` filter the pool by task id. ``Buffer.set_settled_consumer`` pushes cards to the sampler. 11 files, +580/-8. Full suite exit 0.
- **`Investigator` — a read-only delegation whose findings persist into the record.**
  The Genesis reference implementation has an `:read` investigator role
  (`agents/investigator.ex`) that a manager hands a question and that may write only
  `CONTEXT.md`. `--investigate` ports it: a manager marks a delegation `"investigate":
  true`, the delegation runs `Investigator` (`examples/genesis/_investigator.py`) instead
  of the recursion, and what it finds is folded into the node's `CONTEXT.md` under
  `## Findings` — so it enters the version and the next agent inherits it.

  It exists because of a blind spot in the port, not to complete a roster of roles:
  `_MANAGER_PROMPT` carries the `CONTEXT.md` chain and a *listing* of the files at a
  node, never their contents, since this port's agents have no read tool. A manager
  deciding from the code is deciding blind, and the investigator is the one actor
  allowed to read a node and leave a note about it.

  Deliberately narrow. The role's fan-out to `subagent_investigator` is **not** ported,
  because fanning out read-only sub-agents is what a manager with read tools of its own
  is for, and that is the port's session work. The one write it may make is the record
  at the node it was sent to; source edits and writes to an ancestor's record are
  refused and counted.

  14 tests in `tests/test_genesis_investigator.py`, offline and deterministic.

- **`SkillExtractor` — a skill is distilled from an accepted contribution, and the run
  inherits it.** `LocalWorld.skills()` already collected `.agents/skills/` along the node
  chain — `hierarchical_skill_names/2`, and what the paper lists among what an
  accepted version carries (§3.1) — the read half and nothing else.
  `--extract-skills` adds the write half: `SkillExtractor` (`examples/genesis/_skills.py`)
  distils the contribution a root just accepted into a skill file, checks the skills that
  already exist so it does not duplicate one, and places it at the node where it is
  relevant.

  It runs at the **root episode boundary**, not after the merge. A post-merge version
  is one no later episode of the run ever sees, so the loop would still be open; inside
  the episode the skill rides in the same proposal and the next agent inherits it.

  The flag also turns on `LocalWorld.skill_bodies`, which puts inherited skill *bodies*
  in the brief. Without it the feature is decorative — the brief otherwise carries only
  names, and fetching a body needs a `skill_read` tool this port's agents do not have, so
  "read one before using it" was an instruction none of them could follow.

  `skill_add`'s frontmatter is rendered here rather than emitted by the model — the
  format is the tool's, not the model's. Skills get their own trust-region bucket
  (`max_skill_edits=2` beside `max_edits=4`), because a lesson trimmed to make room for
  the file it distils is self-defeating. `skills=None` is the old behaviour exactly.

  22 tests in `tests/test_genesis_skills.py`, offline and deterministic.

- **`evolve(max_tokens=...)`: a budget in the unit that maps to the bill.**
  `max_calls` and `max_rollouts` count invocations, and a reasoning model can
  spend 40k tokens on hidden thinking in a single one -- so a 20-round run with
  4 workers is 80 rollouts and ~160 calls but, at 40k tokens each, 6.4M tokens.
  The cap is checked at the round barrier on the sync path (alongside the other
  budgets) and per-rollout on the async path, stops with
  `stop_reason="max_tokens"`, and reports the spend it actually incurred.

  Tokens only reach the meter when the model adapter reports them, and the
  adapters that do (`claude`, `openai_compatible`) take a `Usage` in their
  constructor. The caller who built `LLMAgent(claude(usage=u))` had to pass
  `usage=u` **again** to `evolve()` for the meter — and with it `max_tokens`,
  the `tokens=` column on every round, and `status` — to see the same numbers.
  Forgetting the second thread made the run report `tokens=0` while the bill
  went to `u`, an object only the caller held, and a token budget silently
  never fired. `evolve()` now adopts the agent's own `usage` when the caller
  did not pass one; an explicit `usage=` still wins, and a bare
  `run`/`propose` pair still reports zero (the honest reading, not an error).

  `RoundInfo.tokens` carries the cumulative spend to `on_round` (and so to
  `rounds.jsonl`), `RunStatus.tokens` lands it in `status.json`, and the CLI
  status line shows `tok=N` when the run has any.

- **`BudgetGovernor`: graceful degradation before the token wall.**
  `max_tokens` is a brake: when the spend reaches it the run stops. A brake is
  the floor of cost control, not its ceiling. The last 10% of a token budget is
  the most expensive part to waste — the search has paid for its exploration,
  and a round dispatched there may never finish. The governor degrades before
  the wall on both paths (sync and async):

  - at 75% of the budget (``soft_floor``), stop building fusion tournaments.
    The tournament is an extra held-out sweep of every survivor plus the fusion
    — pure ranking spend, and the ranking can survive without it for the last
    few rounds. The aggregator acceptance gate runs unchanged.
  - at 90% (``hard_floor``), stop the self-verify rollout. That rollout doubles
    the cost of every proposal for a delta the acceptance test folds in as a
    tie-breaker weight; skipping it costs ranking precision on the advantage
    signal and nothing on the commit gates.
  - the projection at the round barrier (sync) asks whether the *next* round
    fits the remaining budget; if it does not, the run ends with a clean merge
    at the last round it fully paid for.

  All of this is inert without `max_tokens`. `EvolutionResult.budget` carries
  the governor's summary (spent, remaining, whether degradation fired) so a
  caller can tell "the run finished because its budget ran out, degraded" from
  "it converged, at full quality".

- **`SelectionContext.budget_remaining` + `CostEfficient`: spend the budget, don't just count it.**
  The o1 test-time-scaling decision, as a search parameter. `SelectionContext`
  gains `budget_remaining` (the unspent fraction, `1.0` with no ceiling), and
  `Candidate` gains `cost` (`None` = unknown, never zero — the same distinction
  `score` and `prior` make). `CostEfficient` is `FlatPuct` with two additions:

  - the **exploration** term is divided by cost, so a speculative expansion of an
    expensive candidate must beat one of a cheap candidate at the same rank. The
    **exploitation** term is left alone: charging cost against a grounded pick
    would bias the search against complex solutions (longer artifacts cost more
    to expand) in favour of short ones, which is not a quality judgment and must
    not be smuggled in as one;
  - the exploration bonus **anneals by `budget_remaining`**, so a run explores
    while it has budget to exploit what it finds and exploits once it does not —
    what an anytime search has to do, and the difference between spending a
    budget and allocating it.

  `cost_exponent=0` is `FlatPuct` to the floating-point bit (tested), so the
  mechanism cannot change a cost-blind run; uniform costs are unchanged too.
  `anneal=False` is the ablation that separates "cost-aware" from "budget-aware".
  The population layer fills `Candidate.cost` with a deterministic content-size
  proxy (the prompt dominates the input tokens, and size is attributable to one
  candidate without racing a shared meter) and the engine feeds
  `budget_remaining` from the governor each round, on both sync and async paths.

- **`evolve(stop_on_diminishing_returns=True)`: stop when the search stops
  paying for itself.** `max_tokens` is a ceiling; the economic question is
  whether to *spend* it, and the answer is to keep buying compute while it
  still returns quality and stop when it does not — even with budget left. The
  governor measures the run's own return per token (`Δreward / Δtokens` per
  round, from the same reward and meter the run already has) and stops when the
  recent rate has fallen to `efficiency_floor` (default a quarter) of the run's
  *peak* rate. Self-calibrating: it compares the run against itself, so there
  is no absolute rate to guess — a reward is in `[0, 1]`, a token count is in
  the millions, and their quotient has no interpretable scale. Off by default,
  because a run whose reward only rises late would be cut short by it, and
  because it needs `max_tokens`. Stops with
  `stop_reason="diminishing_returns"`. A run that never improved has no peak to
  decline *from*, so this stays quiet and the patience counter handles it.

- **`baselines.Budget(tokens=...)`: the A/B framework can hold cost fixed.**
  The module's own argument is that a budget in one unit is not a comparison —
  "matched on rollouts alone, an arm that asks for more proposals per rollout
  spends more model and the table still says equal budget." That argument was
  missing the unit cost is measured in: two arms at equal rollouts and equal
  calls can differ 10x in tokens when one uses a reasoning model, and the table
  had no column that would show it. `Budget` gains `tokens` (split like the
  others), `Workload._evolve` passes it to `evolve(max_tokens=)`,
  `compare(fixed="tokens")` is a valid unit, `ArmResult.tokens` is the measured
  sum, and the markdown table gains a token column that reads `—` (not `0`)
  when the actor reported none. A comparison that fixes tokens is the only one
  that has shown an arm to be more *efficient* rather than merely given more
  model.

- **`evolve(checkpointing=True)`: the search survives a process restart.**
  `repo_path` already resumed the *artifact* -- the ledger's whole job -- but
  the aggregator's search state is only ever in memory, so a resume re-seeded
  an archive of one from the ledger head and threw away every count the search
  had accumulated: Beta posteriors, a population archive and its `selected`
  counts, the early-stop patience counter. An aggregator opts in by
  implementing `checkpoint()` / `restore()`; both are duck-typed and stay out
  of `AggregatorProtocol`, so an aggregator without them is untouched. Off by
  default, because a run on a throwaway repo would pay the write for nothing.
  The reference `Aggregator` and `PopulationAggregator` implement both, and
  `agentdescent status <run_id>` says which of the two kinds of resume a run is
  set up for.

### Fixed

- `agentdescent init selection --kind policy_slot` now finishes successfully and
  explains how to provide the `data.problems` ref and its import allowlist.
  It previously wrote the starter spec and then crashed looking for `data.path`,
  which policy slots do not use. The clean-wheel smoke check now exercises this
  installed command as well as the offline demo and MCP server.

- **Onboarding, published-version guidance and CI support coverage.** Lead with
  the packaged offline demo on current `main`, distinguish it from the v0.5.0
  source-example path, name NumPy as a runtime dependency, exercise Python 3.10
  in CI and strict-build docs on pull requests with read-only build permissions
  and Pages/OIDC permissions limited to the guarded deploy job.

- **Host setup now stops when the package install fails, before changing host
  configuration.** Isolated installer tests cover checkout and PyPI failures.
  CI and publishing now install the built wheel outside the checkout and smoke
  test its version, bundled skill and hooks resources, temporary host install,
  offline CLI demo, and MCP stdio behavior with a bounded session timeout.

- **Checkpoints are excluded from the ledger's index on every open, not only
  at genesis.** `Ledger._commit` runs `git add -A`. A committed
  `checkpoints/latest.json` exists on `dev` and not on `stable`, so the promote
  switch deletes it -- and once the next round has rewritten the now-tracked
  file, `git checkout` refuses to clobber it and *every* `snapshot(STABLE)` and
  `promote_to_stable` after that fails with "local changes would be
  overwritten". Writing the exclusion only when the repo is created missed
  precisely the long-lived ledgers a resume exists for, since those already
  have their `.git/`. It now goes in `.git/info/exclude` -- git's own space,
  never part of a branch -- rather than a working-tree `.gitignore`, which
  would be committed to one branch and vanish on the switch to the other.
- **A checkpoint that cannot be written or read no longer takes the run down.**
  The lock acquisition raises `TimeoutError`, and it sat outside the guard: the
  one race the lock exists for -- a resume while the previous process is still
  alive -- made the new run fail to start *and* killed the old one's next
  round. Serialisation, locking and IO now all degrade to "there is no
  checkpoint" on every entry point.
- **State that does not round-trip through JSON is refused rather than
  reshaped.** `json.dump(..., default=str)` wrote a `set` as its `repr`,
  reported success, and handed `restore` a candidate whose state had silently
  changed shape -- which `PopulationAggregator` then commits to the ledger as
  the next parent.
- **Round numbering continues across a resume.** Both drivers count rounds from
  zero per process, so a resumed run wrote `round_0.json` over the previous
  run's, reported `round 0` for a search several rounds deep, and gave the
  pruner a history that no longer sorted.
- **`PopulationAggregator` no longer files rendered candidates as artifact
  ids.** Its `_admit` dedup set *was* `Aggregator._seen`, which
  `_known_artifacts` unions into the promotion table -- so `finalize` called
  `_promote("# Playbook\n- change-t2\n...")`, a whole rendered artifact passed
  off as an id, once per distinct candidate. The two sets are now separate
  (`_keys` and `_seen`), which also stops the checkpoint carrying every
  candidate's full text twice.
- **Reads leave nothing behind.** `load_checkpoint` created a `.git/` directory
  to hold its lock file, so `agentdescent status` fabricated a bogus repository
  on whatever path it was handed. The lock now lives inside `checkpoints/`,
  which also means checkpoints work on a plain directory with no git repo.

- **Sparse audit: a cheap verifier paired against ground truth
  (`agentdescent.audit`).** When `reward` is an agent judging an output rather
  than a fact about it, the loop optimises a proxy, and every gate reads the same
  proxy -- so a change that games it is indistinguishable from a change that
  improves. `AuditedReward` wraps the cheap scorer, hands a sampled minority of
  its work to a truth source, and records the pair: `GoldAnswer` for truth that
  returns now, `DeferredOracle` for an experiment that returns next week and is
  resolved from another process against a JSONL that outlives the run.

  It hooks the **reward**, not the verifier. The verifier's layers score
  `(artifact, tasks) -> float`, an aggregate that has already averaged away the
  pairing a bias estimate needs, and a verifier-level oracle would be asked for a
  *fresh* measurement, folding rollout variance into the residual. At the reward
  level both sides score the same output. That placement also makes two
  properties structural rather than remembered: enabling the audit cannot change
  a run (`__call__` returns the verifier's score; tested end to end against the
  same seed with the audit off and on), and it cannot block the merge path.

  Records carry the three things that cannot be reconstructed later -- the
  `inclusion_prob` that drew the unit, the `verifier_version` that scored it, and
  the output itself. Calibration and improvement labels go into disjoint pools
  and `for_calibration` asserts the split rather than filtering for it. The
  estimator, the allocation and the gate that spends the answer are the entries
  below. [docs/audit.md](docs/audit.md)

- **Mutation tests for every statistic added after `ppi.py`.** The plan's rule
  is that new statistical logic ships with a test that fails when the logic is
  written wrong -- a test of the *suite*, not of the code. Four mutations, three
  of which were mistakes actually made here:

  | | the wrong way | what it does |
  |---|---|---|
  | A | discount a *posterior* variance, prior included | past ~40 commits the audit becomes a blanket veto: `p_improve` 0.51 for every candidate, which looks exactly like a mature artifact that stopped improving |
  | B | average the strata's own residual sds | reports `sigma = 0` for a verifier that is +0.4 generous in one stratum and exact in another -- the gate then spends the held-out set as if an oracle had scored it |
  | C | forget to divide `sigma_eps ** 2` by `n` | the discount falls with `n`: the more the run measured, the less of it the gate may believe |
  | D | a control band from one fixed sigma | alarms on a generation that merely bought fewer labels |

- **`EvolveSpec.audit` -- the audit reachable from the CLI and the MCP server.**
  Everything in the package was reachable only from Python, so a run started by
  `agentdescent evolve spec.json` or by the MCP `start` tool could not create an
  audit store -- which left the seven `audit_*` tools able to read a file that
  path could not produce.

  `"audit": {"oracle": "mypkg:exact_match", "sample_rate": 0.1}` is the whole
  minimum. Two of the defaults are decisions: the store is **derived**, landing
  at `audit.jsonl` beside the run's ledger, because `audit_status` takes a path
  and not a run id and a run that wrote its audit somewhere only the caller
  knows is a run whose audit nobody reads; and `enabled` is **false**, because
  collecting records is free while correcting the gate changes what commits, and
  a spec that merely names an oracle has not asked for that. `plan` reports the
  block before anything runs and `status <run_id>` reports `audit_store` once
  records exist -- only once, since a path to a file that was never written
  invites `audit_status` to report an empty store as an answer.

  `oracle` and `stratify` are refs through the spec's own allowlist, an
  acceptance policy the spec already named is wrapped rather than replaced, and
  a mistyped key is an error with the key list beside it -- otherwise
  `sample-rate` is a setting that silently did nothing.
  [docs/audit.md](docs/audit.md#from-a-spec-the-cli-and-mcp)

- **`attach()` -- one call that wires the six pieces.** Everything in this
  package is opt-in and nothing in the shipped runtimes builds any of it, so
  switching the audit on meant assembling `AuditStore`, `AuditedReward`,
  `Calibrator`, `RectifiedAcceptance`, `VerifierWatch` and `RenderTap` plus four
  cross-references, every one of which is silent when wrong: a
  `verifier_version` copied by hand and later changed leaves the gate asking
  about a verifier that never ran and getting a *stale* rectification, which
  reads exactly like "not enough labels yet"; a calibrator pointed at a second
  store is stale forever; a `RenderTap` around a different `run` records an
  empty signature on every unit; a bare oracle not wrapped in `GoldAnswer` can
  fail the rollout it was auditing.

  It changes no default -- not calling it is the current behaviour -- and
  `enabled=False` collects records while correcting nothing, which is the honest
  way to run a first round.

  With it, the plan's last acceptance criterion is a test rather than a plan: a
  whole `evolve()` with a string-similarity judge against exact match, so
  `f > Y` structurally. From nothing but the store -- 114 units seen, 75
  audited, `delta_hat` **+0.35**, `resid_sd` 0.38 -- and the acceptance rate over
  a fixed grid of merge decisions falls **0.60 -> 0.37**.
  [docs/audit.md](docs/audit.md#switching-it-on)

- **The audited units are not independent, and the estimator said they were.**
  `estimate.py` had a cluster bootstrap from Phase 0; `ppi.py` did not, so the
  *refinement* was less robust than the baseline on the one axis Phase 0 had
  already measured. A run scores the same task again for every artifact version,
  and a task the verifier is generous about it is generous about every time.

  On 200 tasks scored under four artifacts each, at a nominal 0.95:

  | treatment | coverage |
  |---|---|
  | independent (as shipped) | **0.79** |
  | cluster-robust variance | 0.91 |
  | cluster-robust, halves from disjoint tasks | **0.945** |

  `Stratum(clusters_lab=...)` makes the labelled term cluster-robust, the
  cross-fitting folds hold out whole groups, and the degrees of freedom count
  groups rather than units. `Calibrator` passes `task_id` by default: on the
  real Phase 0 records that widens `se` from 0.0363 to 0.0523, **44%**, with
  `delta_hat` unmoved. `cluster_var_of_mean` reduces to `s**2 / n` *exactly* on
  singleton groups, which is why the golden vectors did not move a digit -- two
  new fields appeared and every number stayed.

  **The last row is a sampling design, not an arithmetic fix**, and two
  alternatives were measured so nobody retries them: an exact unlabelled design
  effect instead of the borrowed one moves coverage 0.912 -> 0.921, and
  group-aware folds are worth well under a point. PPI assumes the two halves are
  independent samples and a per-unit draw puts the same task in both.

  So `AuditedReward` gained `draw_by`, **defaulting to `"task"`**: a task is
  audited whole or not at all. It is identical to the old per-unit draw when
  each task is scored once and differs exactly where the difference matters. It
  also makes the two pools disjoint at the task level, where a per-unit split
  had one task's units informing the improvement pool about a task the
  calibration pool was measuring.
  [docs/audit.md](docs/audit.md#the-audited-units-are-not-independent)

- **Phase 0 on a second workload, and the judge fails differently there.**
  `--workload bbh` runs the kill test on BIG-Bench Hard, sampled across six
  subtasks chosen for the *shape* of their answers.
  [reports/README.md](reports/README.md) (full report at `c86e216`).

  | | HotpotQA | BBH |
  |---|---|---|
  | `Delta` | +0.175 | **+0.327** |
  | `sigma` | 0.381 | 0.469 |
  | disagreement | 17.5% | 32.7% |
  | \|Delta\|/gate sd | 1.99 | **5.55** |

  PROCEED on both, and the second is not a second sample of the first. The
  residual is **entirely** in the two label-shaped subtasks --
  `salient_translation_error_detection` at +0.90 and `date_understanding` at
  +0.70 -- while `object_counting`, `word_sorting` and both Yes/No subtasks sit
  at `Delta` 0, `sigma` 0, zero disagreement. The judge's error is a property of
  the shape of the answer, so the report splits by subtask rather than quoting
  one average of two unrelated phenomena.

  And the failure is not the benign one. Exact match refuses
  `(B) Numerical Values` against a gold of `(B)`, and a judge accepting that is
  what this experiment was built to measure. Forgiving every formatting
  difference the judge is *told* to forgive -- comparing option labels alone --
  it still says right on **7 of 20** labelled-answer units where that lenient
  oracle says wrong. A merely generous judge scores near zero there. This one has
  stopped discriminating on label-shaped answers, which is a different failure
  from the one `Delta` describes and is **indistinguishable from it in
  `Delta`**.

  Also `audit-limited` is **True** on this workload: `SE(Delta)^2 > var_p`, so
  buying more in-loop evaluation cannot improve the criterion and the budget
  belongs on oracle labels.

- **A run can span a restart now, so the verifier watch outlives the process.**
  `evolve(checkpointing=True)` arrived on `main` while this branch was open,
  and it makes a latent gap reachable: a restart used to mean a new run, so
  "did the verifier change while we were down?" was not a question anyone
  could ask. `VerifierWatch` held its baseline in memory and `attach()` calls
  `check()` to *establish* one, so on a resume it re-baselined silently --
  measured, the same judge with an edited prompt came back `831c0a65` ->
  `103a89aa` with `check()` returning `False`.

  The correction was never at risk. Records are keyed by `verifier_version`,
  so the forty labels from before the restart were not applied to the new
  judge and the gate widened on its own. What was lost is the **diagnosis**:

  | | stale reason after a resume with an edited judge |
  |---|---|
  | before | `0 calibration labels for '103a89aa...'; 30 needed` |
  | after | `verifier fingerprint changed '831c0a65...' -> '103a89aa...'` |

  The first reads as "not enough labels yet", which is exactly the confusion
  `VerifierWatch`'s own docstring says a stale rectification causes.

  `AuditStore.remember_fingerprint` persists it, out-of-band like the
  priorities and for the same reason -- it belongs to the run, not to a unit --
  and writes only on a change, so a watch checked every round does not grow
  the file. A resume that changed nothing keeps its calibration.

- **The judge evolved against ground truth, live.**
  `scripts/audit_evolve_judge.py` run on MBPP: `sigma` **0.4747 -> 0.2549**
  (-46%), disagreement 29.9% -> 6.9%, false negatives 3.2% -> **0.0%**, fixed 21
  and broke 1 against a noise floor of 10. **Nothing blocks.** Two rules, both
  naming the dominant failure -- a candidate that is the expected *value* rather
  than an implementation.
  [reports/README.md](reports/README.md) (full report at `c86e216`)

  Run twice, and the pair is the result rather than either half:

  | | small pool | larger pool |
  |---|---|---|
  | the loop's own gate | 9 units (resolution 1/9) | **30** |
  | reward over ten rounds | 0.778, never moved | 0.800 -> **0.967** |
  | rules accepted | 7 | **2** |
  | scorecard | refused | **nothing blocks** |

  Fewer rules, more improvement. A gate that cannot see below 1/9 was selecting
  on noise, and the seven rules it took were never shown to be signal.

  The larger pool's `P(new error mode)` is **0.0000** -- no variety left at all
  -- and it passed on sufficiency alone, 15 examples of one mode. Under the
  variety-only gate this branch shipped earlier the same morning, the run that
  produced the cleanest result in the sequence would have been refused.

  Bounds, because the numbers above are narrower than they look: one model; a
  noise floor of **11.5%** (the judge disagrees with itself on more than a tenth
  of re-runs); an ordering "pair" between an artifact 70.5% right and one 0.0%
  right, where the report's own note says Kendall tau is uninformative; and a
  dominant error mode that is partly a distribution artifact -- see the entry
  below on the artifact/judge coupling.

- **One interface, one pipeline** -- the docs said ladder, the code never had
  one. `diagnose()` measures, something produces a
  `fix(record, task) -> float`, and `evaluate_fix` + `scorecard` + `rescan`
  verify it. A hand-written predicate, a `search` over combinations, a re-scored
  prompt and an evolved rubric are all that one type and all end at the same
  three functions. The only choice a caller makes is how to express the fix: a
  hard rule is cheap, attributable and mechanical; prompt text can express a
  semantic judgement and is a bundle by nature, so it changes one clause at a
  time. `diagnose()` is not a way to improve the verifier at all -- it says
  whether improving is worth it and where the floor is.
  [docs/audit.md](docs/audit.md)

  Also: the YES/NO reply parser existed in three scripts, identical in all
  three, and `verdict` meant two different things in two of them. One
  `read_verdict`, and the name no longer collides with Phase 0's own verdict.
  [reports/README.md](reports/README.md) indexes the twelve experiment reports,
  which nothing linked to.

- **A third workload, and the rung of the ladder it unblocks.**
  `scripts/audit_evolve_judge.py` evolves the judge's *rubric* with `evolve()`:
  the artifact is the rubric, the reward is agreement with ground truth on one
  judging decision, and the labels come from the improvement pool. It is the
  automated form of `scripts/audit_judge_repair.py`, which measured **one**
  hand-written clause -- that worked, and the next clause also has to be thought
  of.

  The rung sat unbuilt because the pool was **saturated**: across HotpotQA and
  BBH, `P(the next label shows an error mode nobody has seen)` had fallen to
  **0.0169** over nine and four disagreements, every one a shape already
  understood. More labels on those two shapes cannot move that. A third *shape*
  can, so `--workload gsm8k` came first:

  | workload | what the judge gets wrong there | the other two |
  |---|---|---|
  | HotpotQA | forgives paraphrase, extra words, partial names | -- |
  | BBH | stops discriminating on option labels | cannot produce it |
  | GSM8K | wrong final number, marked right because the working reads correctly | cannot produce it -- neither output carries a derivation |

  Adding it turned three single-workload assumptions into bugs, each of which
  had looked like shared code: the oracle (`exact_match` scores `18.00` wrong
  against `18`, so a workload whose oracle refuses its own correct answers
  measures the oracle), the dry run's near-miss (`The answer is 18.` is a
  near-miss for exact match and simply *correct* for `number_match`, so the
  GSM8K rehearsal had no disagreements to rehearse on), and the report's oracle
  label, which was a string literal reading "normalized exact match" over GSM8K
  numbers.

  Four refusals are built into the rung, each one an earlier finding turned into
  a gate:

  | | the refusal | what it stops |
  |---|---|---|
  | saturation | `--min-unseen`, default 0.25 | a rubric fitted to the thirteen errors that happen to be in hand |
  | task overlap | held-out is the calibration pool **minus every task the training set touched** | purpose is drawn per unit and inclusion per task, so a purpose-only split trains and tests on the same question |
  | class balance | the training set is down-sampled to equal right/wrong | agreement with the oracle rewards "reject everything" on a pool where most answers are wrong: 0.836 on the HotpotQA improvement pool, which is *exactly* the real judge's rate on those 55 units |
  | the noise floor | the control arm re-runs the **starting** rubric | on BBH the unchanged prompt scored a smaller residual than the run it was copied from |

  It hands back a `scorecard()` and a `rescan()`, never a judge: swapping the
  verifier invalidates the run's history in a way nothing in the run can see.

  Error modes moved to `scripts/audit_modes.py`, one definition per workload.
  The coverage number decides whether the improvement pool still gets budget,
  and two copies that drifted would answer that in two files with no way to tell
  which one ran. `reports/verifier_diagnosis_f55dec40cec559f7.md` regenerates
  byte-identical after the move, which is what makes the dedup safe to claim.
  [docs/audit.md](docs/audit.md)

- **`agentdescent.audit.propose` -- something that proposes a fix, not only
  scores one.** `diagnose` sorted the verifier's errors and measured a proposed
  rule; nothing proposed one, so the improvement pool's labels were paying for a
  diagnosis nobody acted on. `search` is the plan's first rung automated:
  enumerate the hard rules a person reaches for, score every combination on the
  whole labelled set, rank by the residual. No model, no training.

  On the Phase 0 audit, seven rules two at a time:

  | `sigma` | fixed | broke | FN | rules |
  |---|---|---|---|---|
  | 0.3812 | -- | -- | -- | *as it is* |
  | **0.2421** | 20 | **0** | 0% | `far-shorter(0.6)` + `far-longer(1.6)` |
  | 0.2521 | 19 | 0 | 0% | `far-shorter(0.6)` + `shares-no-token-with-reference` |

  The best pair cuts the residual **36% and breaks nothing** -- better than
  either rule picked by hand -- and `echoes-the-question`, the rule that cuts the
  bias 74% while making the verifier worse, ranks **28 of 28**. Nobody had to
  remember not to ship it. It lands 0.02 above the `floor_sigma` the classifier
  predicted, and the remainder is the AMBIGUOUS bucket.

  Three disciplines, each of which was a way to be wrong: a search is a bundle
  generator and a bundle launders whatever is in it, so every member is also
  scored alone and a combination carrying a non-helping member is flagged; a
  rule may only **reject**, never raise a score, or it could buy a lower
  residual with a higher false-negative rate in one move; and going below the
  floor is a warning rather than a result, because on a few hundred labels it is
  far more likely to be fitting the sample.
  [docs/audit.md](docs/audit.md#searching-for-a-fix-instead-of-guessing-one)

- **`agentdescent.audit.coverage` -- the improvement pool is not allocated like
  the calibration pool.** Neyman (`n_h ~ W_h * sd_h` on the residual) minimises
  the variance of the correction. The improvement pool's job is to find as many
  *distinct* things wrong with the verifier as possible, and the same rule keeps
  sending labels to the layer where the residual is largest long after the
  thirtieth example of the same formatting bug. `plan_coverage` allocates by
  Good-Turing unseen mass instead -- the share of observed items seen exactly
  once estimates `P(the next label shows something new)`.

  Validated against the Phase 0 audit, where all 31 disagreements are in hand so
  the true discovery rate is computable:

  | labels drawn | 5 | 10 | 15 | 20 | 25 | 30 |
  |---|---|---|---|---|---|---|
  | modes found | 3.13 | 4.41 | 5.15 | 5.82 | 6.37 | 6.89 |
  | Good-Turing | 0.360 | 0.192 | 0.143 | 0.123 | 0.109 | 0.099 |
  | **true** P(new) | 0.312 | 0.177 | 0.138 | 0.122 | 0.108 | 0.110 |

  Six times the labels for 2.2 times the modes. And frequency is not value: the
  most common mode in that audit is `echoes-question`, whose obvious hard rule is
  the one that cuts the bias 74% and makes the verifier worse.

  **A label on which the two scorers agreed is still a draw**, counted in the
  denominator as the species "no error". The first version counted only the
  disagreements, which made a layer with ninety-seven agreeing labels and no
  errors look *unsampled* -- it scored 1.0 and drew the whole budget. Ninety-
  seven labels that found nothing is strong evidence there is little to find; an
  absence of labels is no evidence at all.

  `plan.done` is a stopping rule: on the Phase 0 records overall P(new) is
  **0.0169**, so the improvement pool has learnt what it can and the budget
  belongs in the calibration pool, which never saturates.
  [docs/audit.md](docs/audit.md#where-the-improvement-labels-go)

- **`rebalance()` -- the calibration share stops being a constant.** Only one of
  the two pools saturates: calibration keeps buying a narrower interval forever
  at the usual `1/sqrt(n)`, and improvement stops buying anything once the
  labels stop showing new error modes. `plan.done` already said when that
  happened and nothing moved the budget.

  | P(new) | calibration share |
  |---|---|
  | >= 0.25 | 0.50 |
  | 0.10 | 0.77 |
  | **0.0169** (the Phase 0 audit) | **0.92** |
  | 0.0 | 0.95 |

  `Audit.rebalance(unseen)` sets it, for a round hook. All three dials are
  policy, not measurement -- what a label is worth in each pool depends on
  whether you are trying to fix the verifier or to correct for it -- and no
  labels yet returns the floor, because no evidence that the pool is done is not
  evidence that it is.

- **Constraint 7 is locked by a test, not by a convention.**
  `tests/test_gate_reads_the_full_set.py`. The repository shipped the violation
  once -- the regression guard read `cheap_eval`, so lowering `cheap_eval_tasks`
  silently made "quality dropped" a judgement from four tasks. `MergeContext`
  names the two differently to make it awkward to write again; awkward is not
  impossible. The cheap numbers set to nonsense in both directions must not move
  a verdict, and `cheap_eval_tasks` must not move a single commit across a whole
  six-round run whose trajectory is mid-climb -- which is what makes an
  identical history mean something.

- **`agentdescent.audit.queue` -- the merge path's ranking finally reaches
  someone.** `AuditScheduler` has ranked every merge decision since the
  beginning and nothing has ever popped its heap. That was right: `force_oracle`
  is a threshold, and on the shipped verifier an audit is free (`full_eval`
  measures the set the acceptance test just measured), so every qualifying merge
  gets one and a ranking has nothing to do. It starts mattering exactly where
  this package lives -- an oracle that is a wet-lab run or a person -- where the
  budget is smaller than the number of qualifying merges.

  `drain(scheduler, signature_of=...)` turns the heap into
  `artifact_signature -> priority`; `AuditStore.remember_priorities` persists it
  as a snapshot line; `audit_pending(path, order="priority")` hands a person the
  queue in that order, in another process, from nothing but the JSONL. The
  highest priority per signature wins rather than the latest, a diff whose
  signature the caller cannot resolve is counted rather than guessed, and asking
  for priority order with nothing drained says so instead of silently returning
  dispatch order.
  [docs/audit.md](docs/audit.md#which-pending-unit-to-do-first)

- **`agentdescent.audit.ranking` -- can the verifier order things at all?**
  Everything else in this package measures how *far* the verifier is from the
  truth: `delta_hat` its mean error, `resid_sd` the spread, `gain_factor` how
  much an estimator can borrow. The acceptance gate does one thing and it is not
  that -- it decides whether a candidate beats a baseline. A verifier can be
  badly wrong on all three and order every comparison correctly (add 0.2 to
  every score and nothing the gate decides changes), or close on all three and
  pick the wrong winner.

  On the Phase 0 audit, five artifacts from one run: **eight of ten pairs
  ordered the same way, two reversed.** The artifact the verifier ranks first
  (0.714) is third by ground truth, and one reversal is on an apparent 12-point
  improvement -- the size of gap the gate commits on.

  Unit-level Kendall tau is nearly uninformative here and is the number people
  ask for: 4753 concordant pairs, **zero** discordant, `tau_b = 0.681`. That is
  a restatement of the bias being one-directional, not evidence of ordering --
  two units are discordant only when the verifier prefers one and the truth
  prefers the other, and a verifier that answered 1.0 to everything scores zero
  discordant pairs too. `RankReport.one_directional` flags it and the report
  says it in place.

  `RankReport.above(gap)` filters to pairs whose *verifier* gap the gate could
  act on, because a reversal below the gate's own noise costs nothing. The
  scorecard carries an `ordering agreement` row that deliberately **does not
  block**: five artifacts from one run are a lineage, ten pairs is not a sample,
  and a binomial interval on 2-of-10 spans 0.03 to 0.56.
  [docs/audit.md](docs/audit.md#can-it-order-things-at-all)

- **`agentdescent.audit.drift` -- watching the correction over generations
  without alarming every generation.** One rectification says how biased the
  verifier is; a sequence says whether the loop is *finding* its blind spots,
  which is the failure no single measurement shows. `DriftMonitor` is an EWMA
  control chart, and the plan is explicit that a test per generation is the
  wrong instrument. Measured over two thousand runs of a hundred in-control
  generations:

  | | alarms per 100 generations | clean runs that alarm |
  |---|---|---|
  | a two-sided test per generation | 4.95 | **99.3%** |
  | this chart (lam=0.2, L=3) | 0.27 | 16.2% |

  Two departures from the textbook chart, both because the inputs are estimates.
  The limits are **recursive** -- `Var(z) = lam**2 se**2 + (1-lam)**2 Var(z_prev)`
  -- because each `delta_hat` arrives with its own standard error and the closed
  form assumes one shared sigma; the band widens after a noisy generation and
  narrows after a well-audited one. And **overlapping label sets invalidate the
  chart and are the default**: `Calibrator` recomputes from the whole store, so
  consecutive points share most of their labels, are positively correlated, and
  the true spread of `z` is wider than the recursion -- the limits are too tight
  and the chart alarms on a verifier that never moved. The monitor reads
  `Rectification.covers`, notices, and says so rather than charting silently.

  `gain_factor` is smoothed against a threshold rather than charted with limits,
  because it has no standard error. Below it the signal says what the number
  implies: replace the verifier, not buy more labels.

- **`agentdescent.audit.service` and the `audit_*` MCP tools.** `audit_status`,
  `audit_pending`, `audit_resolve`, `audit_recompute`, `audit_scorecard`,
  `audit_rescan`, `audit_drift`. They take a JSONL **path**, not a `run_id`,
  because constraint 2 of this package is that truth may take days: the process
  that dispatched a record is gone when a wet-lab result comes back, and the
  resolver has a file and nothing else. They never raise -- a tool call that
  throws gives a model a stack trace and no way to act.

  Three decisions rather than plumbing. A **missing file is an error**, not an
  empty store: `AuditStore` treats an absent path as one about to be written,
  which is right for a run and wrong for a question about one, and reading a
  typo as "no records yet" is how a caller tells a user their verifier is
  unbiased. `version=None` means the **busiest** version and the reply always
  names which it picked. And `audit_rescan` resolves a `module:attribute`
  reference through the same allowlist the spec system uses -- resolution runs
  whatever it imports, so widening it is the operator's decision and not one a
  calling model can make by naming a module.

  The tool descriptions carry the two warnings a model needs and cannot derive:
  `audit_resolve` REFUSES to overwrite a result, and `audit_scorecard` says in
  as many words not to recommend a verifier change because `delta_hat` fell.
  [docs/audit.md](docs/audit.md#watching-it-over-generations)

- **`agentdescent.audit.scorecard` -- what has to be true before a new verifier
  replaces the old one.** A verifier is the instrument every other number in a
  run is measured with, so changing it invalidates the run's history in a way
  nothing in the run can see. `verifier_scorecard` fills the plan's Phase 5 card
  -- with the top row changed.

  The plan leads with `delta_hat`, "the only real target: it should fall". That
  is the row the harmful rule above wins, so the card leads with `sigma`, treats
  a rise in it as a blocker, and reports `delta_hat` below it, never scored.
  `blockers` is the whole verdict rather than a weighted total, because a total
  would let a large fall in the metric that lies buy a small rise in the one that
  does not.

  `rescan` re-scores the outputs the audit store kept and reports the agreement
  rate, the per-artifact shift, and how many artifact *pairs* reverse order --
  the plan calls this a Ledger replay and says it is nearly free, but the Ledger
  stores artifact states and the outputs a verifier scores were never kept, so a
  true re-decide needs the rollouts back. Re-scoring the audited sample is what
  is free; it comes with an `n` and is weighted by `inclusion_prob`.

  `scripts/audit_diagnose.py` runs the whole Phase 5 diagnosis offline against
  the committed Phase 0 records and writes
  [reports/verifier_diagnosis_f55dec40cec559f7.md](reports/verifier_diagnosis_f55dec40cec559f7.md).
  [docs/audit.md](docs/audit.md#the-scorecard--before-a-new-verifier-replaces-the-old-one)

- **`agentdescent.audit.gate` -- the audit reaching the decision that commits.**
  `RectifiedAcceptance` wraps any acceptance policy and discounts the held-out
  evidence by however much the verifier disagrees with ground truth. On the
  Phase 0 audit that discount is **0.60**: thirty-two tasks judged by an LLM
  carry the information of nineteen judged by exact match, and a candidate
  scoring 0.625 -> 0.750 commits without the audit and does not commit with it.

  The plan's Phase 4 formula, `var_true = var_p + se(delta) ** 2`, is aimed at
  the wrong term. `delta_hat` is one number subtracted from **both** sides of a
  comparison, so it cancels out of `cand - base` exactly, and so does its
  standard error: a gate asking "is this better than that" is nearly immune to a
  uniformly generous verifier. What does not cancel is the *spread* of the
  verifier's error, `resid_sd ** 2 / n`, which each side carries independently
  -- and which the gate had been spending as evidence. Measured: 0.00454 against
  0.00082, **5.5x larger** than the term the plan carries.

  So `Rectification` gained `resid_sd`, the population sd of `f - Y` pooled
  across strata with the between-stratum term included (a verifier uniformly
  +0.4 generous in one stratum and exact in another has zero spread inside each
  and plenty across them). `delta_hat` is still applied, for the two smaller
  reasons that are real -- the Beta spread `p(1-p)` is 2.2x wrong when read at
  an uncorrected rate, and the rates in a refusal are read by a person -- and
  `se(delta)` is carried once rather than twice, named `drift`: the allowance
  for `delta` differing between the two sides, which is the failure this package
  exists to catch.

  Applied by **discounting the counts**, so the rate is untouched and the
  regression guard, `observed_delta` and the artifact's prior all see exactly
  what they saw before. `enabled=False` returns `inner.accept(ctx)` on the
  untouched context -- the same call, not an equivalent one -- so the audit can
  be switched on mid-run. A stale rectification widens instead of correcting; a
  rectification with no `resid_sd` is treated as stale, because the missing term
  is the one the variance is mostly made of. `VerifierWatch` withdraws the
  calibration when the instrument may have moved, by fingerprint, artifact id,
  governance layer or glob over a diff's keys.
  [docs/audit.md](docs/audit.md#spending-it--the-gate)

- **`agentdescent.audit.diagnose` -- what the verifier gets wrong, and whether a
  fix helps.** `classify_disagreements` sorts the residual into what it would
  take to fix -- FORMATTING (normalise), SPEC_GAP (a hard rule), AMBIGUOUS (a
  second oracle would disagree too; **not fixable**), JUDGMENT (a better judge) --
  and records the **direction** of each error beside its kind.

  `report.floor_sigma` is what the residual would be if every non-ambiguous
  disagreement were fixed. It is not zero, and aiming below it is a plan to
  redefine correctness rather than to improve the verifier.

  `evaluate_fix` scores a proposed change against **every** labelled pair rather
  than the disagreements it targets, and `FixReport.helps` reads the residual
  rather than the bias. Both because of what a real audit produced -- see
  `scripts/audit_diagnose.py`, which re-derives it offline from the committed
  Phase 0 records:

  | rule | fixed | broke | `sigma` | `delta` | false negatives | |
  |---|---|---|---|---|---|---|
  | A: answer echoes the question | 12 | 11 | 0.381 -> **0.410** | -74% | 0% -> **22.4%** | *does not help* |
  | B: answer far shorter than the gold | 10 | 0 | 0.381 -> **0.324** | -32% | 0% -> 0% | helps |
  | A + B | 19 | 11 | 0.381 -> **0.362** | -97% | 0% -> **22.4%** | helps |

  Rule A cuts the bias by three quarters and makes the verifier worse; restricted
  to the disagreements it targets it removes twelve errors and breaks nothing, a
  clean win by every number a person reaches for. And bundled with a rule that
  works it *passes* -- the bundle's residual improves, so nothing in the bundle's
  own numbers shows that it still rejects 22.4% of correct answers. A bundle
  launders whatever is in it, so `evaluate_fix` is meant to be run one rule at a
  time.

  The lesson is in the API: **do not optimise the verifier against `delta_hat`**.
  A mean can be driven to zero by adding errors in the opposite direction. The
  bias is what the calibrator already handles; optimising it breaks the residual,
  which is what the calibrator cannot handle.

  `FixReport.breakage_rate` and `FixReport.false_negative_before/after` were one
  field called `false_negative_rate` until the same data put them side by side at
  7.5% and 22.4%: the denominators are the judgements that were right and the
  *answers* that were right, and those differ whenever the verifier errs in one
  direction only -- the normal case.

- **`agentdescent.audit.sampler` -- Neyman allocation, as inclusion
  probabilities.** A flat rate spends the oracle budget where the *units* are;
  what sets the width of the correction is where the verifier is *unreliable*.
  `plan_audit` allocates `n_h` proportional to `W_h * sd_h` on the residual
  `f - Y`, converts it to a per-stratum rate, and hands it to the tap.
  `boundary_stratifier` splits units around the acceptance threshold, where a
  verifier's disagreement with the truth concentrates.

  `target_halfwidth` is a specification, not a wish: under Neyman allocation
  `se = sum(W_h sd_h) / sqrt(n)`, so the label budget follows from the requested
  half-width and halving it costs four times the labels.

  The plan called for a batch sampler that takes a generation's units and returns
  which to send. That does not fit the tap, which decides per unit from a
  unit-seeded draw -- the property that makes inclusion independent of thread
  scheduling. The allocation survives the translation: `n_h` out of an expected
  `W_h * N` is an inclusion probability, and per-stratum rates are what the tap
  already takes.

  Three decisions with reasons: floors are applied **after** allocation, so a
  bound stratum does not scale the others down; a stratum with no residual
  history is filled at the **largest measured** sd, because under-sampling an
  unmeasured stratum is self-perpetuating while over-sampling self-corrects; and
  a stratum the plan never saw gets a rate of **zero**, since sampling it would
  record an inclusion probability nobody chose.

- **`agentdescent.audit.calibrator` -- the join.** `Calibrator.current(version)`
  reads the store, assembles the strata, runs the estimator and returns a
  `Rectification`: `delta_hat = E[f] - E[Y]` with the standard error the
  acceptance gate adds to its own variance.

  `E[f]` is a **count**, not an estimate -- the tap saw every unit the run
  scored -- so only `E[Y]` carries sampling error, which is why `delta_se` and
  `se` are the same number and would stop being so if `E[f]` were ever computed
  from a subsample.

  Every failure returns a **stale** rectification rather than a number or an
  exception, because the caller is a merge decision and one has to be made: too
  few labels, an unknown verifier version, a `mark_stale()` after the verifier
  changed, or a converged run where both scorers saturate and there is no
  variance to estimate from. That last is not a correction of zero -- a converged
  run has no evidence about the verifier either way.

  Thin strata are **merged, not dropped**: records and their unlabelled moments
  together, pooled with Chan's parallel form. Dropping would remove those units
  from the population the estimate describes, turning the question into "the bias
  among units we sampled enough of" -- flattering exactly when the thin stratum
  is where the verifier is worst.

- **The store keeps the unlabelled half as three numbers per stratum.** PPI's
  entire dependence on the units nobody audited is a count, a mean and a
  variance, so `AuditStore` accumulates them in a Welford and never stores an
  unlabelled score. At a 1% sampling rate that is three numbers instead of a
  hundred thousand. Snapshots share the records' JSONL under a `kind` key and
  reconcile last-wins; `Stratum.from_moments` is the constructor that takes them.

- **`agentdescent.audit.ppi` -- the calibration estimator.** Prediction-powered
  inference for a stratified mean: `theta = lam * mean(f_unlab) + mean(y_lab -
  lam * f_lab)`, with `lam` chosen to minimise the variance and **cross-fitted**
  so the interval is honest. At `lam = 0` it degenerates to the labelled-only
  mean, so a useless verifier costs nothing; `gain_factor` reports the factor by
  which the oracle budget was effectively multiplied, and `1.0` means the
  verifier is buying nothing.

  Guarded three ways, because a wrong interval here makes the acceptance gate
  confident about a number it should be hedging. Coverage over 400 replications,
  with the reported SE checked against the actual spread. Three mutation tests
  whose coverage must collapse -- ignoring stratum weights (0.94 -> 0.01),
  imputing the verifier as truth (-> 0.00), dropping `lam**2 Var(f_unlab)/N`
  (-> 0.91) -- which test the coverage suite rather than the estimator. And six
  golden vectors exact to 1e-12, spanning the regimes coverage cannot separate.

  **This introduces numpy as the package's first runtime dependency.**

- **`numpy` is a dependency of the core.** `dependencies` was empty and
  `docs/plugin-design.md` said it must stay that way; the estimator above is the
  reason to change that, and the doc now says what is true. Everything else in
  the package, and the CLI, still run on the standard library alone.

  Installing it un-skipped `tests/test_era_srbench.py`, which gated on numpy
  alone while also needing scipy and sympy -- three tests that had never run
  failed immediately. The guards now name every optional dependency they use,
  matching how the same file already handles `pyarrow`.

- **`anthropic_compatible` -- an Anthropic-format endpoint with no SDK
  dependency.** The twin of `openai_compatible` on the other wire format.
  Anthropic-format endpoints now serve models that are not Claude, and reaching
  one meant either `pip install anthropic` -- an optional dependency, for a
  `base_url` override on a client whose retry and timeout defaults then have to
  be undone -- or nothing. Base URL and key are read from the environment at call
  time. Only `text` blocks are returned: a reasoning model answers with
  `thinking` blocks first, and concatenating them would put the reasoning into
  the artifact's output, where a scorer grades it and a diff might commit it.

- **`agentdescent.audit.estimate` -- the design-based bias estimator.** A Hajek
  (inclusion-probability-weighted) mean of `f - Y` with a percentile bootstrap
  interval, plus `residual_bias(records)` to read it straight off a store.
  Weighted from the start although every current sampler uses one probability:
  an unweighted mean is correct exactly while that stays true, and plausible
  either way once it stops.

  Two intervals are reported. The unit bootstrap is what a reader expects and is
  **too narrow for a run**, because the same task is scored again for every
  artifact version and those pairs are not independent draws. `ci_clustered`
  resamples tasks, and the gap between the two is the size of that dependence --
  on the Phase 0 run it is about 1.6x.

  Not included, still: prediction-powered inference. The two are a baseline and
  a refinement rather than alternatives, so PPI gets measured against this one.

- **`scripts/audit_phase0.py` -- the audit plan's kill test, and
  `reports/audit_phase0_*.md`.** Not the plan's original Phase 0, which replays
  the Ledger for past accept decisions: no run ever recorded an independent
  second opinion to replay against. It is a fresh measurement on HotpotQA with an
  LLM judge as the cheap verifier and normalized exact match as ground truth.

### Changed

- **`ThreeLayerVerifier.oracle_eval` is now `full_eval`**, and
  `oracle_shares_full_set` is now `full_eval_matches_counts`. The old name
  promised an independent source of truth and the method delivers the same
  `eval_fn` every other layer calls, differing only in **how many tasks** it
  scores -- so it bounds sampling error and is structurally unable to detect that
  the scorer is biased. Reading it as ground truth is what let the docs claim the
  loop audits itself against something outside itself, which it does not.

  Both old names still work; the `oracle_eval` compatibility path emits a
  `DeprecationWarning`. Both aliases are scheduled for removal in 0.7. `VerifierProtocol` declares `full_eval`; the engine reads a custom
  verifier through `verifier.full_eval_of` / `verifier.shares_eval_counts`, which
  accept either spelling -- so a verifier written against the pre-0.5.1 page keeps
  running instead of raising `AttributeError` in the middle of a merge. The
  `oracle_budget` argument, `VerifierBudget.oracle_calls_*` and the
  `oracle-rejected` merge outcome keep their names: they are the public spelling
  of a knob and a result category, and renaming them would break callers for no
  gain this rename has not already delivered.

## [0.5.0] — 2026-09-07

### Added

- **`scripts/setup-hosts.sh` and a single testing guide.** One command
  installs AgentDescent and wires up every agent CLI on the machine
  (`--with-clis` installs the four npm packages first, `--dry-run` shows the
  plan). It is idempotent, warns when `agentdescent` is not on the `PATH` hosts
  will use, and finishes with `doctor`. [docs/testing-guide.md](docs/testing-guide.md)
  is the path in order: install, prove it offline, check each host, drive it in
  plain language, then a real model -- with the measured quirks
  (`codex doctor` under-reports, Claude Code caches a failed connection) and the
  two failure modes worth recognising written down.
- **`host_model` works on every host, not only ones that implement sampling.**
  Measured by logging what each host sends at `initialize`: Claude Code 2.1.261
  declares `roots` and `elicitation`, OpenCode 1.18.29 declares `roots`, and
  dsh's `dsh-mcp-client` declares nothing -- **none of them supports sampling**,
  so a `host_model` that only spoke sampling would have been a feature nobody
  could use. It now falls back to running that host's own CLI (`claude`,
  `codex`, `dsh`, `opencode`) with the user's real configuration
  (`isolate=False`), chosen from the client name and only when it is on `PATH`.
  A bridge that dies mid-run -- the session closed -- falls back the same way
  instead of failing. `start` reports `host_model_route` so the caller knows
  which one it got. Verified end to end: a run started from a real Claude Code
  session, reflecting through `claude`, producing evidence cards and a commit.
- **`host_model`: run on the model the agent session is already using.** MCP's
  `sampling/createMessage` lets a server ask its client for a completion, so
  `{"reflect": {"ref": "host_model"}}` reflects on the host session's own model
  -- its authentication, its policy, no key and no model name in the spec.
  The obstacle was structural: `start` returns in milliseconds and the run
  proceeds in a **detached process** with no MCP session and no way to get one.
  `agentdescent.host_sampling` bridges that gap -- a token-authenticated
  loopback endpoint in the server process holding the live session, whose
  address the server puts in the environment of the runs it launches, turning
  each request back into `create_message` on the server's event loop.
  `start` now reports `host_model_available` and, when false,
  `host_model_unavailable`: a run whose reflector could not ask otherwise looks
  exactly like a run that learned nothing. Verified end to end through the real
  SDK -- a detached run reflecting on a model only the client holds, reaching
  reward 1.0. **Sampling is deprecated at the protocol level as of MCP revision
  2026-07-28 (SEP-2577)**; the shape is unchanged and the SDK still ships it, so
  this is the convenient route, not the durable one.

### Changed

- **Merging is the default now, wherever there is a model to merge with.**
  `reflective_merge` had to be asked for by name and the skill told agents not
  to ask, so a plugin driven in plain language never installed it -- and
  without it a one-key artifact (`SingleSlot`: every prompt, every
  `kind: "text"` target) has worker proposals that contradict by construction.
  Conflict resolution collapses them to one candidate, no fusion is built, and
  `n_workers=4` buys per-round best-of-N *selection* rather than the merge this
  project is named after. The pair is installed in `_build_engine` (so `evolve`
  and `async_evolve` both get it) when the agent exposes a completion, and in
  `compose` when a spec asks for no conflict or fusion rule of its own, using
  the model the spec already names in `reflect`. Callers with no model -- plain
  `run` / `propose` functions, the offline demo -- and specs whose only agent is
  a file-editing CLI keep the shipped rules, because there is nothing cheap to
  synthesise with. `DefaultConflict` and `DefaultFusion` are exported so the old
  behaviour can be named. Measured on one spec with no `policies` block, four
  workers: `conflicts_dropped` 3 -> 0, `n_candidates` 1 -> 4, `fused` 0 -> 1,
  ledger `merge synth(w0+w1+w2+w3)`.

### Fixed

- **`apply` overwrote its own backup.** On a single-file target it copied to a
  fixed `<target>.bak` every time, while the directory path had always numbered
  them. Apply, hand-edit, apply again, and the surviving backup was the
  hand-edit -- the original the first apply had saved was gone, from the command
  whose whole promise is that it keeps one. Both paths now write `.bak-N` and
  never touch an existing one.
- **A run's cause of death was truncated where it mattered, and named twice.**
  `executor` described the rollout failure, `supervisor` described it again over
  the queue, `evolve` a third time, and each hop capped it -- so the narrowest
  won. What reached `status` was `RuntimeError: RuntimeError: ...` cut at 200
  characters: a provider 404 for an unknown model ended at "does not support the
  c", losing the clause that named the fix. One `pipeline.describe()` now,
  shared by all of them and by the async path, capped at the 2000 `runstore`
  already stores. Measured on the same failure: 214 characters before, 422
  after.
- **The MCP server reported no version, and "no runs" reported nothing at all.**
  `initialize` answered with an empty version string, so any host that displays
  it displayed a blank; it is passed now wherever the SDK's constructor accepts
  one. And `status` with no `run_id` returned a bare list, which serialises to
  *zero* content blocks when empty -- an agent asking what was running on a
  fresh machine could not tell that from a failed call, on the one tool the
  skill says to poll. It returns `{store, runs}`, which names the store either
  way.
- **`plan` says when `n_workers` is selection rather than merging.** With the
  merge pair now default the warning has narrowed to the case still true -- a
  spec with no model to merge with -- and reads the composed bundle rather than
  the spec's `policies` block, so it cannot fire at a spec that already has what
  it recommends.
- **The suite did not collect on a clean checkout, and two tests assumed
  Linux.** `pyyaml` is used by the dsh patch tests and was never declared, so
  the documented `pip install -e ".[dev,mcp]"` left the suite unable to import.
  On macOS `mkdtemp` hands back `/var/folders/...` while a child resolves the
  symlink to `/private/var/...`, so `HOME` never equalled the worker's `getcwd()`
  -- the workspace root is `realpath`-ed now, which is what a host comparing the
  two would expect. `sandbox_wrapper` was passed a `str` where its signature says
  `Path`, which only the macOS Seatbelt branch dereferences. And the dsh plugin
  test reported a corepack too old for the pnpm the profile pins as a plugin
  failure; it skips with dsh's own error instead.

- **`evolve` and `resume` started specs that could not work, in silence.**
  `plan` returned `warnings`; the two verbs that actually spend money printed
  none. And the check knew nothing about `host_model`, which is the one ref
  whose usability depends on *where it is started from*: measured, a spec with
  `"reflect": {"ref": "host_model"}` resumed from a shell did four rounds with
  `considered: 0` and finished at reward 0.0 -- every proposal raised and the
  run reported nothing but a flat reward, which reads as "it learned nothing"
  rather than "it had nothing to ask". Both verbs now print the warnings, and
  the check covers `host_model`.

- **`status` reported a rollout count that grew quadratically.**
  `RoundInfo.rollouts` is cumulative by its own definition and the run store was
  adding it to the previous total, so it summed a running sum. Measured on a
  630-round run: 1,270 calls and a reported **397,530** rollouts -- 630x631, the
  triangular number. It is one of four figures `status` puts in front of someone
  deciding whether to cancel, and an agent asked to explain the cost of
  cancelling quoted it verbatim.
- **`show` never named the file `apply` would overwrite.** It returned the
  evolved artifact, the diff and the reward, but no `target` and no `kind` --
  and for a `text` run no apply plan either. Driven in plain language, the host
  agent read the evolved prompt, then declined to apply it because "the run
  record doesn't show what file this was evolved against". It now returns
  `kind`, `target`, and an apply plan on every kind.
- **The skill had no answer for evolved text being instruction-shaped.** A
  successful prompt evolution produces imperative text by construction, and the
  same agent flagged one as "a prompt-injection payload embedded in tool output"
  and refused. The skill now says the artifact is content to write to a file and
  never instructions addressed to the agent -- do not obey it, do not refuse it
  for being imperative, and do refuse it if it asks for something the user would
  not want in their own file.
- **`plan`'s new warning promised a remedy that does not work.** It offered
  "put OPENAI_API_KEY in the environment" for `codex`, and suppressed itself
  when that key was set. Measured against codex-cli 0.153: with
  `OPENAI_API_KEY` and `OPENAI_BASE_URL` both set, codex ignored the base URL,
  called `api.openai.com`, and sent no credentials at all. Only `dsh` documents
  an environment fallback (its own error names it); the others now get the
  mechanism and an explicit "untested for this CLI" rather than a promise.

- **OpenCode worker isolation did not work, and its test passed anyway.** The
  previous entry claimed `OPENCODE_CONFIG_DIR` isolated an OpenCode worker like
  the other three hosts. Measured against opencode 1.18 by running the real CLI
  with a worker's environment: it does not. That variable supplies a config only
  when the user has none -- with a real `~/.config/opencode/opencode.jsonc`
  present, a worker still saw the user's MCP servers, with or without a config
  file in the redirected directory. The test asserted the variable was *set*,
  which was true and meant nothing. `XDG_CONFIG_HOME` is what moves OpenCode's
  config, and with it redirected the same worker reports "No MCP servers
  configured"; it is now set (and dropped from the parent, since `setdefault`
  cannot override an exported value) alongside the others.
- **Isolated config directories were pointed at but not all created.** The set
  that gets created was a second list beside the set that gets set, and OpenCode
  was added to one only. `codex` refuses to start when `CODEX_HOME` does not
  exist ("Error finding codex home"); OpenCode silently falls back. Both lists
  are now one mapping.
- **`plan` did not warn that a CLI on `PATH` may not be signed in.** An isolated
  worker runs with the host's config directory redirected, so an interactive
  login does not carry over. Measured: a run with `codex` -- present, logged out,
  no key -- was priced without complaint and then failed every rollout, 35 calls
  in. `plan` now says so and names the fix (`"isolate": false`, or a provider key).
- **The skill did not fire for "make this skill better" when no test cases
  existed** -- the state most users start in, and the one its own procedure has
  a step for. The description ended "and has (or can write) examples with
  expected answers", which reads as a precondition. Reworded to lead with the
  user's words and say outright that drafting cases is step one. It now surfaces
  in that case; a bare "make it better" with no data still often gets a hand
  rewrite instead, which is honest to record rather than force.
- **The skill's example spec used a relative `data.path`.** It resolves against
  whichever directory read the spec -- the host's MCP server cwd, which the
  agent cannot see -- so the same spec found the file from one directory and
  not another. The skill now says to write every path absolute.
- **The skill let "just run it, don't ask" skip the cost estimate.** Asked for
  60 rounds and 32 workers with blanket permission, it started immediately
  without quoting a number. Pre-authorisation now waives the confirmation and
  never the estimate.

- **The shipped `SKILL.md` never said how to choose an `agent`, and `plan` did
  not check the choice.** Driving the skill in plain language against a real
  Claude Code session, three separate wrong answers came out of that gap: it
  picked `codex` (a file-editing CLI) to evolve a *prompt*, which is one agent
  session per case to answer a question a model answers in one call; then it
  invented `"model": "gpt-4o-mini"` for an OpenAI-*compatible* endpoint that
  serves nothing of the sort; then it named `claude` calling it "the local
  Claude CLI, no API key needed" -- `claude` is the Anthropic SDK completion,
  the CLI is `claude_code`, and neither the package nor the key was present.
  The skill now says which agent each `kind` wants, never to invent a model
  name, and what `claude` actually is; `doctor` reports `openai_base_url`
  itself rather than only that one is set; and `plan` returns `warnings` when
  a named agent or reflector cannot run on this machine, so the cost quote is
  not for a run that fails on its first call.

- **`agentdescent install dsh` could stop dsh from starting at all.** Both
  cordis patch files wrote the forwarded provider keys one `!!js
  process.env.X` per line. `env` is validated as `{[key: string]: string}` and
  an unset variable is `undefined`, so with no keys exported dsh rejected the
  entry -- and it fails the *whole plugin tree* on a bad entry, so dsh would
  not boot until the user edited the file by hand. `--dump-config` composed it
  happily, which is why it was missed: only loading the plugin tree
  (`dsh --profile headless`) shows it. Both files now share one quoted `!!js`
  expression that filters unset keys (quoted because the `": "` inside it is
  otherwise read as a nested YAML mapping). Reproduced and fixed against dsh
  0.1.2-rc.1; the suite now boots real dsh through both the `install` and the
  native-plugin routes.
- **`install` could add its dsh block but never repair it.** The check was
  "is this marker present", so an install that had written the broken entry
  above reported "already present" and changed nothing. The block is now
  delimited and rewritten when its content is out of date -- including blocks
  written before the end marker existed -- while the user's own rows are left
  alone.

- **An OpenCode worker read the user's real config; the other three hosts did
  not.** `worker_env()` redirected `CLAUDE_CONFIG_DIR`, `CODEX_HOME` and
  `DSH_HOME` into the rollout workspace and had no OpenCode entry, so an
  `opencode` worker started with the user's model, credentials, plugins and MCP
  servers -- not a policy difference, a host that was missed.
  `OPENCODE_CONFIG_DIR` is now redirected with the rest, and `OPENCODE_CONFIG`
  and `OPENCODE_CONFIG_CONTENT` are dropped, because redirecting the directory
  alone is not isolation: measured against opencode 1.18, a config named by
  `OPENCODE_CONFIG` still supplied its MCP servers with the directory pointed
  at an empty one. `OPENCODE_API_KEY` survives -- it is a provider credential,
  and a worker needs its keys.

### Added

- **Documented which model a run actually uses.** A worker is the host CLI as a
  subprocess with its config directory redirected, so it inherits environment
  keys but not the user's model choice or subscription login. The two switches
  that change that -- `extra_args: ["--model", ...]` to pin a model with
  isolation intact, and `isolate: false` to hand the worker the user's real
  setup -- existed but appeared in no document. Both are now in
  [docs/plugins.md](docs/plugins.md) with a worked "no provider key at all"
  spec that points `agent` *and* `reflect` at a host CLI, and in the shipped
  `SKILL.md`, so an agent that sees `doctor` report no key offers that instead
  of stopping.
- **`pip install "agentdescent[mcp]"` no longer fails the whole install on
  Python 3.9.** Every published `mcp` requires >= 3.10 while this project
  supports 3.9, and the unmarked extra meant the line every install doc gives
  ended in a screen of "Ignored the following versions that require a different
  python version" with *nothing* installed -- not even the CLI. The extra now
  carries `python_version >= '3.10'`, so 3.9 gets the CLI and the skill; and
  one helper, `cli.mcp_unavailable()`, gives `doctor`, `install` and
  `agentdescent mcp` the same true reason instead of advising a pip command
  that cannot help there. Reported from a real 3.9 install; verified on 3.9.23
  and 3.11.

- **`agentdescent demo` -- a complete evolution with no API key.** Every other
  entry point needs a worker agent on `PATH`, a provider key and a dataset with
  known answers, so the first thing a newcomer saw was an error about a missing
  file. `demo` builds a skill directory whose `references/rules.md` names the
  wrong column, twelve CSVs with known totals, and a spec pointing at the two
  public callables `agentdescent.demo:offline_agent` (a subprocess bound to the
  workspace, which really reads the candidate skill off disk) and
  `offline_reflector` -- then runs the real loop and prints what it learned.
  Nothing is mocked but the model, so staging, the layout, the ledger, the
  parallel workers, the merge and the held-out gate all run as they do against
  Claude Code. Any spec can point at the same pair to rehearse before spending
  anything. `agentdescent init` now says what to do next when the cases file
  does not exist yet. Beginner path:
  [docs/plugin-quickstart.md](docs/plugin-quickstart.md).
- **AgentDescent as a plugin for DeepSeek Harness, Claude Code and Codex.** A
  shared `SKILL.md` teaches the host agent when to call it; an MCP server
  (`agentdescent mcp`, `pip install "agentdescent[mcp]"`) exposes `doctor`,
  `plan`, `start`, `status`, `show`, `apply`, `cancel`, `resume`; and a CLI
  (`agentdescent`) mirrors them verb for verb. `agentdescent install
  <dsh|claude-code|codex>` writes the skill and each host's manifest -- for
  DeepSeek Harness an `mcp-client` entry and the `hooks-claude-code` bridge in
  `cordis.patch.yml` with provider keys forwarded explicitly (dsh scrubs them);
  for Claude Code a plugin directory, also published through
  `.claude-plugin/marketplace.json`; for Codex a `config.toml` entry.
  Guide: [docs/plugins.md](docs/plugins.md); design record:
  [docs/plugin-design.md](docs/plugin-design.md).
- **`EvolveSpec`: an `evolve()` call as data** (`agentdescent.evolvespec`). A
  JSON spec names the artifact (`kind`: `text` / `skill_dir` / `agent_dir` /
  `agent_code` / `plugin`), the data, the scorer, the agent and, optionally, a
  `policies` block and `agg_config`; `compose()` turns it into the `evolve()`
  call the quickstarts show and is tested against them field for field, so it
  is a wire format for callers that are not Python, not a second entry point.
  Agents, scorers and policies are `Ref`s resolved inside the `workspec`
  allowlist; a spec carries no code and no secrets. The `policies` block is
  possible because every shipped policy is now constructible from JSON scalars
  and installed by the aggregator through `bind` / `configure`.
- **A run store** (`agentdescent.runstore`): a run is a directory under
  `~/.agentdescent/runs/<id>/` plus a detached process. `status.json` is
  replaced atomically from `on_round`, `resume` re-launches on the same ledger,
  `cancel` signals the process group so worker CLIs die with the run. Relative
  paths in a spec (`target`, `data.path`, a `cmd` grader) are resolved where the
  spec is *read*, because the process that runs it is a different one with a
  different working directory; the copy stored beside the run is therefore
  re-runnable from anywhere.
- **`kind: plugin` -- the host plugins themselves are evolvable.**
  `runners.plugin_runner` is a per-host table over `code_runner`: the candidate
  plugin is loaded into an isolated copy of the host that lives inside the
  workspace (`dsh plugin add link:`, `claude -p --plugin-dir`, Codex's skills
  directory), validated (`pnpm test` + `--dump-config`, `claude plugin
  validate`), then the host runs the task. `PLUGIN_FROZEN` freezes hooks,
  permission config, lockfiles and tests by default; the layer is L1.
- **`evolve(stop_when=)` / `async_evolve(stop_when=)`**: the caller's own budget
  -- dollars from a shared `Usage`, a deadline, a kill file -- asked after
  `on_round` at the same point as `max_seconds` / `max_calls`, ending the run
  between rounds with `stop_reason="stop_when"`.
- **`rewards.command_scorer`**: grade with any program (task JSON on stdin,
  `$ANSWER` in the environment, a number in `[0, 1]` on stdout); a failing
  grader raises `GraderError` rather than teaching the optimiser zeros.
- **OpenCode is a fourth host.** `agentdescent install opencode` writes the
  skill to `~/.config/opencode/skill/agentdescent/SKILL.md` and merges an
  `mcp.agentdescent` entry into `~/.config/opencode/opencode.jsonc` in the shape
  `opencode mcp add` itself writes (`type: "local"`, the command as one array),
  preserving anything already in the file and refusing to touch one it cannot
  parse. Also `agents.opencode()` (`opencode run`), an `opencode_skill` layout,
  and an `opencode` entry in `PLUGIN_HOSTS`.
- **A native DeepSeek Harness plugin**, `integrations/dsh-agentdescent`,
  rendered from the Python package by `render_dsh_plugin()` so its embedded
  skill cannot drift from the one every other host installs. It registers the
  skill through `ctx.skills.register()` at runtime and its `cordis.patch.yml`
  adds the MCP row, so `dsh plugin --profile web add link:<path>` is the whole
  install. Verified by installing it into a real profile: `dsh --dump-config`
  composes both rows, and a test drives `apply()` through dsh's own
  `isSkillName` validator. Installing it needs
  `"dsh": {"bundle": {"patch": "./cordis.patch.yml"}}` in `package.json` --
  without it `dsh plugin add` warns "declares no dsh.bundle ... not a profile
  layer" and the plugin is installed but inert. It also contributes a **runs
  panel** to the dsh web UI -- a header action with the live run count and the
  list behind it -- as a hand-written client bundle in the shipped
  `window.__ModuleLoader__.load` shape, verified by loading it the way dsh's
  loader does and rendering it with real React.
- **`agentdescent serve`**: a read-only HTTP view of the run store on loopback
  (`GET /`, `/api/runs`, `/api/runs/<id>`), for a host UI to embed and for
  looking at runs in a browser. Cross-origin reads are answered only for a
  loopback `Origin` -- `*` would let any page the user visits read their run
  store off localhost, and a prefix check would accept
  `http://127.0.0.1.evil.com`.
- **Codex installs the same plugin as Claude Code.** Codex reads the Claude
  Code plugin format and marketplace, so `codex plugin marketplace add
  Birfy/agentdescent && codex plugin add agentdescent@agentdescent` installs the
  package this repository already publishes, and the MCP server comes with it --
  verified with `codex plugin list` and `codex mcp list` against codex-cli
  0.153.4. `agentdescent install codex` still writes the files for anyone who
  would rather not use a marketplace, and now prints the one-command
  alternative. (OpenCode's plugins are a different thing -- a JS hook API for
  intercepting tools and chat -- so its skill and MCP server stay plain
  configuration; see docs/plugins.md.)
- **Every host manifest is now verified against the real CLI** (dsh 0.1.2-rc.1,
  Claude Code 2.1.261, codex-cli 0.153.4, opencode 1.18.29), which corrected
  three things that documentation research had got wrong:
  a dsh `cordis.patch.yml` **overrides rows by id**, so the new rows have to sit
  under `- insert:` -- without it dsh warned `patch: entry "mcp-agentdescent"
  not found` and composed without the server; `codex exec` has **no
  `--full-auto`** flag (the real ones are `--sandbox workspace-write` and
  `--skip-git-repo-check`, the latter needed because a rollout workspace is not
  a git repo); and a Claude Code **plugin** namespaces its MCP tools
  `mcp__plugin_<plugin>_<server>__<tool>`, not `mcp__<server>__<tool>`, so an
  allowlist or a probe built from the documented name never matched.
- **`agents.dsh()`**: DeepSeek Harness's headless profile as a worker;
  `dsh_skill` and `agents_skill` layouts. **`agents.worker_env()`**: every
  `cli_agent` child drops the host session's markers (`CLAUDECODE`,
  `CLAUDE_CODE_*`, `CODEX_*`, `DSH_*`), is marked `AGENTDESCENT_NESTED=1`, and
  gets `CLAUDE_CONFIG_DIR` / `CODEX_HOME` / `DSH_HOME` inside its workspace, so
  a worker cannot recurse into the host session or this package's MCP server.

### Removed

- **The one-call wrappers: `evolve()` is the only entry point.** `evolve_skill`,
  `evolve_skill_dir`, `evolve_agent_dir` and `evolve_agent_code` (and the
  `agentdescent.skill` / `agentdescent.skilldir` modules) are gone. Each was
  thirty lines of wiring plus a set of defaults, and each was a second signature
  to learn, document and keep in step with the first. What they assembled is
  now public building blocks that go straight into `evolve()`: `tasks_from` and
  the new `scorer()` / `SCORERS` (moved to `agentdescent.rewards`) for a
  dataset, `SingleSlot` + `reflector` for an instruction, `load_tree` +
  `FileTree` + `tree_runner` / `code_runner` + `tree_reflector` for a directory,
  the new `runners.gated_reward` for a test-gated code run, and
  `governance.SKILL_BLAST_RADIUS` / `HARNESS_BLAST_RADIUS` for the layer. The
  two defaults the wrappers set for a real-agent workload
  (`self_verify=False`, `cheap_eval_tasks=4`) are now passed explicitly, and
  every doc page and the directory example show the full call. Migration is
  mechanical; the quickstarts are the template.

### Added

- **A systematic guide to the policy slots**, `docs/policy-guide.md`: where
  each of the eight slots sits in a round, what it is handed and must return,
  the five composition rules the engine enforces, the three ways to fill a
  slot (shipped, wrap, replace), the per-slot contract with its one common
  mistake, the `bind` / `configure` install hooks, how to prove a policy ran,
  a recipe table, and the pitfalls in the order people meet them. Linked from
  the policies catalogue and the nav.

### Fixed

- **A wrapper around a default policy can now be installed through `Policies`
  alone.** `DefaultConflict` and `DefaultFusion` take the verifier through an
  optional `bind(verifier)` hook, `DefaultAcceptance` and `DefaultPromotion`
  take their thresholds through `configure(config)` (or the pinned
  `from_config(cfg)`), and the aggregator now offers both hooks to every
  installed merge-side policy — previously only fusion was bound, so
  `Policies(conflict=AdvantageConflict(DefaultConflict(...)))` type-checked,
  installed, and died with `'NoneType' has no attribute 'cheap_eval'` on the
  first contradiction that reached the inner rule; the repository's own tests
  had to route it through an `aggregator_factory`. The wrappers
  (`AdvantageConflict`, `AdvantageAcceptance`, `StableDistanceAcceptance`)
  forward the hooks and default their `inner` to the shipped rule, so
  `AdvantageAcceptance()` reads the run's `agg_config=` instead of a hand-copied
  `DefaultAcceptance(0.5, 64, 4000)` that silently diverged from it. A default
  used without being installed raises `PolicyUnboundError` naming the missing
  piece. New: `agentdescent.aggregator.install_policy`,
  `tests/test_policy_install.py`; docs on
  [acceptance](docs/acceptance-policies.md) and
  [conflict](docs/conflict-policies.md) policies.

## [0.4.6] — 2026-08-28

### Added

- **ERA runs LLM-SRBench — a benchmark this repository did not build.**
  `examples/era/era_llm_srbench.py` is a fourth task on the same flat-PUCT tree
  search, the same aggregator, the same sandbox and the same governance layer,
  behind the same `Domain` seam the integrals and 2F1 tasks use. What is new is
  where the yardstick comes from:
  [LLM-SRBench](https://arxiv.org/abs/2504.10415) (ICML 2025 Oral) is a
  published set of scientific equation-discovery problems with its own metrics
  and its own leaderboard of LLM-based methods. The other two constructed tasks
  answer "is this search any good?" against a bar this repository set; this one
  does not.

  Scored on **LSR-Transform** — 111 Feynman equations rearranged so the closed
  form being asked for is not one a model has memorised — under the benchmark's
  own per-problem protocol, upstream's answer format, upstream's linear root and
  upstream's single `BFGS` from all ones:

  | LSR-Transform, all 111 | SA | Acc(0.1) | median NMSE | budget |
  |---|---|---|---|---|
  | LaSR (paper, best backbone) | 6.31% | 50.45% | 0.0011 | millions of GP mutations |
  | LLM-SR (paper, best backbone) | 31.53% | 39.64% | 0.0091 | 250 prompts |
  | here, `glm-5.2` | — | 36.9% | 0.102 | 18.5 calls |
  | **here, `deepseek-v4-flash`** | **41.4%** | **56.8%** | **2.15e-08** | **16.5 calls** |

  Five protocol settings here are stricter than the benchmark's own — 4 000
  training rows against 80 000, selection on a 25% validation slice rather than
  full-train MSE, and a non-finite prediction failing the whole problem where
  upstream drops the point — so those are floors.

  What it recovered is the part worth reading: Bohr energy levels solved for the
  principal quantum number, the Planck distribution solved for temperature,
  waveguide dispersion returned as `c*sqrt(k**2 + (pi/d)**2)` where the dataset
  poses it with `d` multiplied out, relativistic Doppler, and the paramagnetic
  two-level partition as the two-exponential expansion of `2n*cosh(mu*B/kT)`.

- **`python -m tools.score_symbolic_accuracy` scores the paper's third metric.**
  Acc(0.1) and NMSE ask whether an answer *predicts* the held-out samples;
  symbolic accuracy asks whether it **is the equation**, and that is the column
  separating discovery from interpolation. It is a separate tool because it needs
  the ground truth, which must never come near the search. A deterministic sympy
  check runs before the judge and settles what it can, putting a floor under the
  metric that depends on no model. The judge is whatever `--model` names and not
  the paper's GPT-4o, which the output file states.
- **A fourth ERA task: AlgoTune, scored in speedup rather than accuracy.**
  `examples/era/era_algotune.py` runs the *same* flat-PUCT search, aggregator,
  sandbox and governance layer as the other three ERA entry points over
  [AlgoTune](https://github.com/oripress/AlgoTune)
  ([arXiv:2507.15887](https://arxiv.org/abs/2507.15887)) — 147 of its 154 tasks,
  **one tree per task**, with the task's own reference implementation as the root
  node and its own `is_solution` as the correctness oracle.

  Every ERA task so far optimised accuracy. This one holds accuracy fixed and
  optimises time: a candidate is scored by how much faster than the reference it
  is, and a solution the checker rejects scores nothing at all, however fast —
  AlgoTune's own rule, and what keeps the benchmark about speed. The baseline is
  not a strawman: it is `scipy.linalg.eig`, `scipy.integrate.solve_ivp`,
  `scipy.signal.upfirdn`, `scipy.spatial.Delaunay`.

  Three things make the number mean something. The root node *is* the reference,
  lifted out of its `Task` class into a runnable program by an AST transform that
  raises rather than guesses when it cannot (`_algotune_tasks.derive_seed_program`),
  and a test checks the lifted program computes what the class computed. The
  problem sizes are **upstream's published ones**, read from AlgoTune's own
  `reports/generation.json`, so two runs are comparable without either
  calibrating against its host. And the reference is re-timed inside the sandbox
  beside the candidate, on the same problem, reference first — a baseline
  measured once on the host and reused would make the score move when the
  machine got busy rather than when the program got faster.

  147 rather than 154 because two want `os.urandom`, two do not lift out of
  their class, `lqr`'s own checker rejects its own baseline, and two need a
  dependency this repository does not carry. It was 72 until AlgoTune's own
  pinned dependency set went in (OR-Tools, networkx, scikit-learn, jax, SymPy,
  POT, PySAT, faiss, mpmath, hdbscan, cryptography); the gap had never been the
  benchmark, only this port's environment. Their
  reference does not lift out of its class. `lqr` clears both filters and is
  still excluded, and the reason is upstream's: its own `is_solution` calls
  `float()` on a 1×1 array, which NumPy has refused since 1.25, so the reference
  is invalid by the task's own oracle.

  `python -m examples.era.era_algotune --list-tasks` prints the runnable set;
  `--tasks all` runs it. Notes in [`docs/algo-era.md`](docs/algo-era.md), offline
  tests in `tests/test_era_algotune.py`.

  **Measured** on the eight tasks AlphaEvolve, MetaEvolve and OpenEvolve all
  publish, one run each against deepseek-v4-flash with 45 expansions per tree:
  harmonic-mean held-back speedup **2.195x**, against AlphaEvolve's 1.392x and
  MetaEvolve's 2.045x on the same eight, ahead on five of eight against each.
  The largest are algorithm changes rather than flag-twiddling: 540.172x on
  `polynomial_real` (a numba Aberth iteration for `np.roots`'s companion-matrix
  eigenvalue solve), 5.019x on `fft_cmplx_scipy_fftpack`, 3.995x on
  `psd_cone_projection` (`eigh` and a broadcast for `eig` and a materialised
  `np.diag`). One is not: `lu_factorization`'s 4.464x skips the reference's
  three `.tolist()` calls, which is legal and is 132.5 ms of its 183.2 ms, but
  is not a better factorisation — discounting it gives 1.782x. Two tasks finish
  *below* 1.0x on the held-back sets after improving on the sets they could see,
  which is what the split is for. Table, caveats and the per-task comparison in
  [`docs/algo-era.md`](docs/algo-era.md#measured-results--algotune) and
  [`bench/results/era-algotune-model-prior.md`](bench/results/era-algotune-model-prior.md).

- **A model prior in PUCT's `P(s,a)`, in place of ERA's uniform `1/N`.**
  `--prior-exponent` (default `0.0`, which is upstream to the floating-point
  bit) asks the mutation reply for a `PROMISE: <n>` line rating the *approach
  after tuning* 1–10, and uses `p^k / Σp^k` as the prior. The question is
  deliberately about the approach rather than this draft: asked the other way
  the rating collapses into the score the evaluator already produces, and what a
  prior is for here is separating "slow today, right idea" from "fine today,
  finished". It is read out of the reply the port was already paying for, so it
  costs no extra call, and a missing line takes the mean of the rated candidates
  rather than zero.

  It is predictive. Over 250 rated nodes, 60 of the 137 rated ≥8 reached 2x
  against 3 of the 93 rated ≤6 — Fisher one-sided p=2.2e-13, 92.3% recall at
  43.8% precision on a 26.0% base rate. Against *validity* it is worth nothing
  (Spearman 0.046): the model can say how fast an idea would be if it worked,
  not whether its own code runs, which is the right division of labour when the
  sandbox already measures correctness. Visits follow it (Spearman 0.49) and
  mean tree depth goes from 8.0 to 12.3.

  On AlgoTune's eight most-published tasks it takes the harmonic mean from
  1.440x to 2.195x, against AlphaEvolve's 1.392x and MetaEvolve's 2.045x on the
  same eight. It is not a general accelerator: on tasks gated by `is_solution`
  rather than by speed it aims the budget at the wall, leaving 3 valid nodes of
  46 on `least_squares` against a uniform prior's 20, and half as many on
  `affine_transform_2d` across three seeds. Per-task comparison and the caveats
  in [`bench/results/era-algotune-model-prior.md`](bench/results/era-algotune-model-prior.md).

### Fixed

- **The AlgoTune sandbox pinned the reference's threads and not the
  candidate's.** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` were all set to `1`, so the
  reference's LAPACK calls ran on one core — but none of them reach numba, which
  reads `NUMBA_NUM_THREADS` and defaults it to the core count. A candidate
  compiled `@njit(parallel=True)` was therefore timed on every core against a
  one-core reference.

  Capping numba as well would have been the wrong repair: writing a parallel
  implementation *is* an optimisation, and AlgoTune sets no thread policy at
  all. So neither side is capped now, and `--cpu-seconds` is multiplied by the
  core count, since `RLIMIT_CPU` sums CPU seconds over threads and would
  otherwise kill a parallel candidate for using what it was given.

- **A rejected answer could be told its numbers were correct.** The note
  attached to an `is_solution` rejection flattens both sides before comparing,
  so a solver returning the right values in the wrong container looked identical
  to it. `affine_transform_2d` checks `proposed.shape != image.shape` before it
  compares a value, so a flat list of the correct 20000 pixels was rejected and
  the model was told "off by 0.000e+00, 0x the tolerance" — 13 of 29 rejections
  in one run. The note now describes the container when it differs, and says so
  explicitly when every number agrees and `is_solution` still refused.

## [0.4.5] — 2026-08-19

### Fixed

- **`--eval-concurrency` was parsed by the ERA ports, recorded in their result
  files, and then ignored.** `run_agentdescent_era` hard-coded
  `"eval_concurrency": workers`, so a run launched with the shared flag's
  default of 8 wrote `eval_concurrency: 8` into its JSON while the engine
  actually got 3. A recorded configuration that disagrees with the run is worse
  than no record: it is the exact defect `examples/_common.py` was written to
  prevent, and every measured row in `docs/algo-era.md` carried it.

  Fixed the way that module documents: `eval_concurrency_default=None` on all
  three ports ("this runner has a rule of its own; only override it when the
  flag is given"), and the port's own rule — one evaluation per worker, because
  a held-out evaluation here is a sandboxed process rather than an API call —
  applies when it is not.

  Measured while finding it, on the endpoint these runs use: the sandbox is
  **0.7–2.5% of wall clock** and the model calls are ~95%, so concurrency here
  should be sized to the endpoint and not to the host. That endpoint takes 6
  concurrent requests with no latency penalty (17.9 → 90.2 tok/s aggregate from
  1 to 6) and degrades past it (74.2 tok/s at 12, tail latency 10s → 24s).

- **A hosted endpoint was damaging replies in transit, and the search was
  scoring the damage as if the model had written it.** Measured on a GLM-5.2
  endpoint behind an Anthropic-shaped API: roughly **one reply in five** of a
  few thousand characters came back with bytes spliced into the middle of
  tokens — `return val9.3192`, `c_orig,0$ zG$C$F1_orig` — pooling the certain
  cases (a reply that does not parse) across 58 sampled replies.

  It is not the client. The rate is the same through the Anthropic SDK, through
  the SDK's streaming API, and through a hand-rolled `urllib` request; 25
  fetches of a similarly-sized file over the same proxy returned one identical
  hash. It is also not free: on the first 2F1 run, 3 of 15 expansions died on a
  `SyntaxError` the model did not write.

  `examples/era/_era_support.py` gains `reply_is_intact` and
  `with_intact_replies`, and all three ERA entry points a `--reply-attempts`
  flag (default 4, `1` disables). **The distinction is the whole point**: a
  reply is redrawn only when it is not Python at all — it does not parse, or it
  holds a character Python source cannot hold. A program that is wrong, that
  never terminates, that raises, or that imports something the gate bans is
  *not* redrawn; it becomes a node scoring `-inf`, which is what upstream
  requires and what `test_a_bad_program_is_not_treated_as_damage` pins. Each
  run now records `reply_damage` beside its result, so the tax is in the data
  rather than in prose.

  Known limitation, stated rather than papered over: a splice that lands inside
  a numeric literal still parses, and nothing here can see it.

- **The ERA sandbox runner could not import a candidate that used
  `@dataclass`.** `dataclasses` is on the port's import allowlist, so a model
  writing `@dataclass class Split: ...` is writing an allowed program — and it
  died with `AttributeError: 'NoneType' object has no attribute '__dict__'`,
  scored `-inf`, and looked exactly like a model that had written a broken
  program. `dataclasses` resolves its own module through
  `sys.modules[cls.__module__]`, and `load_entrypoint` executed the candidate
  without ever registering it there. Both runners now register the module
  before executing it. Found while porting the second ERA task, whose *shared
  catalogue module* is a dataclass and hit it on the first run.

- **`test_the_sandbox_blocks_the_writes_and_network_it_claims_to_block`
  asserted something Bubblewrap deliberately does not promise.** The probe
  wrote to a path beside the scratch directory — which pytest puts under
  `/tmp`, and the Bubblewrap profile mounts a **fresh tmpfs over `/tmp`**. The
  write therefore succeeds *inside the sandbox* and reaches nothing, and the
  test failed on any ordinary Linux host with `bwrap` installed, on `main`,
  while the guarantee it exists to check (`not outside.exists()` on the host)
  held the whole time.

  The strict "was refused" assertion now targets a path under `/usr`, which no
  profile makes writable on either platform; the `/tmp` probe keeps the
  host-invisibility check that was always the real claim. A test that fails
  where the code is correct is worse than no test: it trains the reader to
  ignore the one assertion in this repository that decides whether running
  model-written code is safe at all.

- **A staleness test asserted a property of the machine, not of the policy.**
  `test_offline_examples.py::test_full_discards_nothing_and_guarded_discards_the_most`
  required `guarded.discarded_stale > 0` from `_async_stats` — an 8-second,
  wall-clock-bounded run of six real worker threads against one merger. Whether
  any diff's `eta` exceeds alpha there depends on whether the merger drains
  slower than the workers fill; on an idle host it keeps up, every card arrives
  at `eta == 0`, and Guarded correctly discards nothing.

  It failed on `main` in CI on Python 3.12, and on the same commit in a later run
  on 3.11 instead, passing the other two both times. The victim version rotating
  between runs is the signature of a load-sensitive assertion rather than of a
  regression — and unlike the `pipelined_gate` livelock below, which CI caught on
  *all three* versions, there is no committed change to bisect to.

  The assertion is now the part that is deterministic: Full discards nothing by
  definition, and Guarded can never discard *less* than Full — an inversion is a
  real bug and still fails. The strict claim keeps its home in two tests that can
  hold it on seeded, bounded runs:
  `test_staleness.py::test_a_refresh_interval_makes_every_staleness_action_reachable`
  (every branch including DISCARD is reachable) and
  `test_the_staleness_sweep_actually_varies_with_alpha` (a tight alpha discards
  strictly more than a generous one).

- **The pipelined-gate seam disabled the async path's livelock guard, in
  exactly the situation the guard exists for.** `pipelined_gate` needs the
  merger to skip poll sweeps whose candidate is still out being measured, and
  the skip it shipped was `if not reports: return`. On the **inline** path "no
  reports" also describes a sweep whose every card was discarded at the
  staleness gate -- and the return skipped `stall.note_sweep`, the counter
  behind `stall_patience`. Under `async_ratio >> alpha` that counter is the
  *only* thing that ends the cycle: head stops moving, so the lag budget never
  forces a worker refresh, and the forced-resync epoch is all that is left.
  The same return also broke `history`'s documented "one entry per merger
  sweep that had cards" and left the published head un-bumped on those sweeps.

  Caught by CI, on all three Python versions, in
  `test_a_large_lag_budget_does_not_livelock` -- the test named for the
  failure. It was first misread here as a pre-existing flake, on the strength
  of a stash test that removed only *uncommitted* work while the offending
  committed change stayed in place; `main` is green under the same load. The
  test is genuinely load-sensitive (a 3-second budget, four spinning workers
  against one merger), which made the local signal noisy -- never absent.

  The fix is the missing half of the condition: `if not reports and not
  batch`. A pure collect poll is skipped; a sweep that had cards and reported
  nothing is the livelock's signature and must be counted.

### Added

- **The 2F1 suite is 3000 points, because 240 could not resolve anything —
  including a result this changelog had already drawn a conclusion from.**
  Per-point correct digits have a standard deviation of **3.20**: the outcome is
  close to bimodal, a program either handles a region and scores near the
  12-digit cap or misses it and scores near zero. An 80-point gate therefore
  carries a standard error of **0.358 digits**, so the smallest gain it can tell
  from noise at two standard errors is 0.72 — larger than every number measured
  on it. Pairing does not help (paired SD 3.05 against unpaired 3.20; a program
  that changes a point changes it by ten digits, not a tenth).

  `tools/gen_hyp2f1_stress.py` now draws **250 points a shard**: a 1000-point
  acceptance gate and 1000 points held back. Twelve minutes of
  arbitrary-precision arithmetic, once, committed. Evaluation was never the
  constraint — a 250-point shard costs the baseline 0.30 s, and scoring a node
  across the full gate 1.7 s. The generator is now seeded **per shard**, so
  `test_the_committed_stress_file_redraws_identically` can redraw one shard and
  demand it back instead of regenerating the file; `--check` still verifies all
  of it.

  **This reverses a conclusion recorded above.** Re-scored on 1000 fresh points,
  the 48-expansion winner beats the baseline by **+0.347 ± 0.10 digits (3.2 SE)**
  with 737 of 1000 points solved against 642 — a real improvement the old split
  could not see. The earlier entry called that run a winner's curse on the
  strength of gate +0.56 and held-back −0.15; both were draws from a
  0.36-standard-error distribution, and the diagnosis was wrong. Corroborating
  it: on the new suite the gate and held-back halves score the baseline at 9.801
  and 9.836, a gap of 0.035, where the old halves differed by 0.49.

  What survives unchanged: the tree still spent its budget badly (best score at
  expansion 8, the remaining 40 expansions exact copies of it, chains 14 deep),
  and the reply channel still damaged 12 of 60 replies.

  **A run made under the resolving gate**, 48 expansions on `glm-5.2` at
  `--workers 6`: gate 9.801 → **11.737**, held-back 9.836 → **11.771**, 965 of
  1000 held-back points solved against 659. The two halves agree to **0.001
  digits** (+1.9359 against +1.9346), which is what the resize was for. At 11.74
  the program is past the 10.84 ceiling of picking the best of four textbook
  transformations by oracle — it carries its own Taylor summation, degenerate
  parameter handling and a continuation near `z = 1`. 598 s against 1,280 s for
  the same budget at `--workers 3`, the endpoint's measured concurrency knee
  being 6. Recorded in `bench/results/era-hyp2f1-run48-gate1000.json`.

  **A specialist review of the artifact found two wrong identities in it, both
  confirmed independently.** The `z→1−z` connection formula has both Γ
  coefficients wrong against DLMF 15.8.4 (zero correct digits at
  `a=0.3, b=0.7, c=1.9, z=0.6`), and the `a≈b` guard applies `₂F₁(a, a; c; ·)`
  where Pfaff requires `₂F₁(a, c−b; c; ·)` (267× too large at
  `a=b=−2.3, c=4.1, z=−6`, a point SciPy gets right). **Neither is reachable
  from this suite** — the first branch is called zero times on the 3000 points,
  and `|b−a| < 10⁻⁵` has probability ~3×10⁻⁷ under the draw — which indicts the
  suite's coverage rather than defending the code.

  The same review overturned a claim made here: the "oracle" ceiling of 10.98
  was computed over a basis that **excluded the identity the program uses**.
  With `z→1/z` included the oracle is 11.919 and the program's 11.738 is below
  it; that identity applied blindly, with no selection at all, already scores
  11.253. So 74% of the gain is one identity, and the honest result is that the
  search **rediscovered the z→1/z connection formula and a `z < −1` switching
  rule** — real, but not what was originally written down.

  Coverage gaps the review quantified on the committed file: zero points with
  `z ≥ 1`, one above `z = 0.99`, zero with `a` or `b` a non-positive integer
  (the terminating case, the most common practical use of ₂F₁), zero with `b−a`
  or `c−a−b` within 10⁻⁶ of an integer (the logarithmic cases), and 0.5% with
  all parameters under 5 in magnitude. The program is left exactly as the search
  produced it; hand-editing it would destroy its value as a record.

  It also earns the next lever: across 3000 points the winner gains 6,142 digits
  and loses 475, and **13 of the losses are points the baseline had to 10+
  digits and it has to under 1**. No numerical library ships that, whatever the
  mean does.

- **ERA's third task: double-precision evaluation of the Gauss hypergeometric
  function, and what replaces a leaderboard.** `examples/era/era_hypergeometric.py`
  asks for `hyp2f1(a, b, c, z)` over `a, b, c ~ U(-30, 30)`, `z ~ U(-40, 0.999)`.
  Nobody has run an agentic program search on it, so there is no ranking to
  join; three properties stand in for one.

  *The problem is hard on somebody else's authority.* The standard survey —
  Pearson, Olver & Porter, Numerical Algorithms 74:821–866 (2017) — exists
  because no single method covers the parameter space.

  *The baseline is the state of the practice.* `scipy.special.hyp2f1`, Cephes
  underneath, the function a working scientist already calls. On the declared
  distribution it loses more than six digits on roughly a third of points, some
  of them at zero correct digits — and a test fails if SciPy ever improves
  enough that the reported numbers need restating.

  *The reference cannot be argued with.* mpmath at 30 **and** 60 decimal digits,
  kept only where the two agree to 25, shipped as a committed file that
  `python -m tools.gen_hyp2f1_stress --check` re-derives byte for byte. One test
  re-computes every stored value at 60 digits; another reruns the generator and
  demands the same file back, which is what rules out points having been picked
  after seeing how an implementation did on them. `mpmath`, `decimal` and
  `fractions` are off this task's import allowlist, because the deliverable is a
  float64 routine and a candidate reimplementing arbitrary precision would be
  answering a different question.

  The headroom is real and was checked before the task was built: a five-line
  rule that never sees the answer — Pfaff when `z < -1`, choosing the branch by
  parameter size — moves 400 sampled points from 10.93 to 11.84 mean correct
  digits and cuts failures from 144 to 101.

- **ERA's second task: the paper's own "numerical solution of integrals".**
  `examples/era/era_hard_integrals.py` runs the *same* flat-PUCT search as the
  Kaggle port — same tree, same visit reservation, same aggregator, same
  sandbox profile, same governance layer — over a different kind of program.
  A candidate is `integrate(f, a, b)` and is handed a **black-box scalar
  integrand**: no formula, no parameters, no family name, over `[0, 1]`,
  `[0, inf)` or `(-inf, inf)`. The seam is a `Domain` (seed program, evaluator,
  prompt, metric name) and nothing algorithmic lives in it; `domain=None` is
  upstream's task, so a caller who names no domain gets the port upstream ships.

  **The reference is a closed form, not another integrator.** Nine integrand
  families, each breaking a different assumption a quadrature rule makes —
  singular at both endpoints, oscillation accumulating into a corner, a peak of
  width 1e-7, an oscillation on a half-line that never decays, cancellation to
  four orders of magnitude below the integrand — and every one of them has an
  exact value in terms of `Γ`, `atan` and `π/sin(πs)`. Scoring against another
  numerical method would have meant penalising a candidate for being *better*
  than the reference. `test_the_closed_form_matches_high_precision_quadrature`
  checks each identity against mpmath at 30 digits through a per-family
  substitution, because mpmath's own quadrature on the raw integrand disagrees
  with the closed form in the third decimal place on the Fresnel family — which
  is the benchmark working, not a defect in the reference.

  The metric is **mean correct significant digits**, `min(12, -log10(relative
  error))`, with the cap set by the precision of the references themselves. A
  problem that raises, returns `nan` or overruns its budget scores 0 and the
  rest of the set still counts. Every problem has a **hard cap on calls to the
  integrand**, enforced in the runner: without it the best program is whichever
  one is allowed to spend the most, which is not a question about method.

  Measured live on `glm-5.2`, 12 expansions, 3 workers, barrier-free: mean
  correct digits **8.86 → 10.21** on 36 held-back integrals, 20/36 → 28/36
  solved to 10+ digits, in 391 s. The winner keeps `quad` as its kernel and adds
  the transformation and error control around it. Recorded in
  `bench/results/era-integrals-run.json`; full notes in `docs/algo-era.md`.

  **The first run of this task found a hole in the task**, which is the part
  worth keeping. The whole-line family was `exp(-x²)·cos(bx)` — an *even*
  integrand — and the winning program mapped both halves onto `[0, inf)` and
  doubled, which is a plain bug that scores 12 digits on an even function. The
  family now carries a nonzero offset and a test pins the asymmetry. The
  reported numbers are from a rerun on the corrected suite; the first run's are
  not reported at all, because they were measured against a suite that could be
  passed without solving it.

- **An eighth benchmark-faithful port: ERA, Google Research's empirical-software
  search.** `examples/era/` runs upstream's own bundled task — Kaggle Playground
  S3E1, `train_and_predict(train_path, test_path)`, RMSE — under upstream's own
  search, Flat UCB tree search. FUTS is small enough to port line for line and
  the port does: the PUCT formula, `c_puct = 1.0`, the rank normalisation with
  its single-node 0.5 case, the uniform `1/N` prior, visits backpropagated up the
  parent chain, and **a node appended for every expansion including a failed
  one** — upstream scores those `-inf` and keeps them, and dropping them would
  shrink the rank denominator and raise the prior on every later iteration.
  Upstream's own `futs_test.py` fixtures are reproduced as tests rather than
  paraphrased.

  The selection rule went to the engine as
  `selection.FlatPuct`, beside `ParetoFrontier`, `Archive` and `MCTS`. It is not
  `MCTS` with different constants: the tree is *flat* (every node competes, no
  descent from the root) and the exploitation term is a **normalised rank**
  rather than a value, which is what makes one exploration constant work across
  RMSE, log-likelihood and accuracy without retuning.

  **Parallelising it moved exactly one thing, and it is not a semantics change.**
  Upstream backpropagates a visit after `execute_fn` returns; this reserves it at
  *selection*. With one proposal in flight nothing can observe the tree between
  those two points, so every selection sees identical visit counts —
  `test_serial_tree_reproduces_upstream_futs` drives this port's tree and a
  line-by-line transcription of `futs.search` with the same mock generator and
  executor and asserts the same node is expanded at every step, with the same
  final visit vector. Without the reservation `argmax(puct)` is deterministic and
  a batch of N workers would all be handed the same parent.

  **Upstream ships no sandbox** — `implementation/sandbox.py` is an abstract class
  whose `run` raises "Must provide a sandbox for executing untrusted code" — so
  the sandbox is entirely this port's, and the page says plainly that it is
  weaker than the OpenEvolve port's: the benchmark requires pandas/numpy/
  scikit-learn, so the AST gate cannot be the boundary and the Bubblewrap profile
  binds the root read-only rather than a handful of directories. What it does
  enforce is checked against the kernel, not by reading the profile back.

  Measured, `glm-5.2`, `async_evolve(n_workers=3)`, 6 expansions: test RMSE
  **0.7297 → 0.5913** (−19.0%) on 2,476 rows the search never scored, from
  upstream's `LinearRegression` seed to a gradient-boosting model with ten
  engineered features and early stopping. Two things the score does not show and
  the page records anyway: the held-out gate is **95% of the wall clock** (four
  full trainings per card, on the merger thread), so more `--workers` buys almost
  nothing here; and the barrier-free schedule makes the tree **root-heavy**, five
  of six expansions attaching to the root because sweep 1 dispatched proposals
  before any sibling had been inserted. One run, one seed, and no `--serial`
  control, so no speedup is claimed. See `docs/algo-era.md`.

- **Evaluation can now be its own stage: `async_evolve(pipelined_gate=True)`.**
  Two modules claim to implement FlashEvolve's stage orchestration and both
  implement two stages — workers and a merger — with *Evaluate* folded into the
  merger. The counters above say what that costs: the merger runs ~90% busy,
  94% of it in the gate, and 4.5 of 8 workers sit blocked at the backpressure
  gate at any moment.

  A merge is three phases, and only the middle one is expensive: prepare (drain,
  staleness, fusion), **measure** (the four verifier calls), decide (accept,
  audit, CAS commit). `Aggregator` now exposes them as `begin_step` / `measure` /
  `finish_step`, with `step()` defined in terms of the three — so the published
  contract is unchanged and every custom aggregator keeps working. `measure()`
  writes only into the candidate it was handed, which is what lets the async
  driver run it on its own threads.

  **No commit semantics change.** `begin_step(skip_in_flight=True)` bounds the
  pipeline to **one candidate per artifact**, so every candidate is still
  committed against the head it was prepared and measured on — there is no
  candidate-level staleness to have a policy about, which is the difference
  between this and a change that quietly loosens what a commit means. Cards
  arriving meanwhile accumulate in the buffer, so batches get larger rather than
  more numerous.

  **It does what it says, and on the stub workload that buys nothing.** Both
  halves are measured, at `n_workers=8`, `async_ratio=3`, held-out scoring at 5x
  a rollout. The mechanism works -- merger occupancy over four runs each falls
  from 40-67% to **24-41%**, with fewer and larger merges, which is what
  one-candidate-per-artifact predicts. Throughput does not follow: over seven
  runs each in a 6s window, rollouts min/median/max read 1008/**1202**/1320
  inline against 862/**1162**/1480 pipelined -- a **0.97x** median with fully
  overlapping distributions, and 1.02x where the gate costs what a rollout does.

  The counters say why: freeing the merger only helps if the merger is the
  binding constraint, and here it is not. Workers gate on
  `len(intake) > async_ratio`, the merger polls every 5ms, and eight workers
  producing a card every 20ms refill the queue past 3 between sweeps whatever
  the merger is doing.

  An earlier draft of this entry read **+42% rollouts and +70% merges**. That
  was one run per arm, and the spread inside a single configuration is wider
  than the effect -- inline alone ranges 1008-1320 at n=7. The two runs happened
  to land at opposite ends of it, in the direction that flattered the change.

  **Off by default**, because it is a third pool and the ceiling a provider sees
  becomes `n_workers + gate_workers * eval_concurrency`. Warns and does nothing
  on the synchronous path, where the round barrier idles every worker for the
  whole merge whatever the gate runs on.

  **It reaches no port yet, and says so.** Every algorithm in `examples/`
  supplies its own `aggregator_factory` implementing `AggregatorProtocol` from
  scratch, so none has the three phases and `pipelined_gate=True` warns and runs
  inline -- which includes all eleven runtime-matrix rows. Porting one means
  expressing it in phases and leaving `step()` inherited.

  Having the three methods is **not** sufficient, and that was a real bug in the
  first version of the check: `PopulationAggregator` derives from `Aggregator`,
  inherits all three, and overrides `step()` to admit the pre-merge head into
  its archive and consult its selection policy. A `hasattr` test passes there,
  and driving the phases directly would have skipped every line of the override
  -- running a different algorithm while reporting the requested one. The check
  now also requires `step()` to be the base implementation.

  On a live model (**GLM-5.2**, GEPA/HotpotQA, 16 rollouts, 4 workers) the
  profile is sharper than the stub's: `merge_gate_seconds == merge_seconds`
  **exactly** -- the merger spends all of its busy time in the gate -- with
  `worker_starved_seconds=54.6`. `eval_seconds` read 1894s against a 740s
  process wall-clock, which is the whole reason it cannot answer this question.

  The first version of it made runs *slower* — 462 rollouts inline against 389
  pipelined — by skipping the merger's poll sleep whenever a measurement was in
  flight rather than *finished*. That is a busy-wait, and a busy-wait holds the
  GIL against the workers it was meant to free, while `merge_seconds` reported a
  confident 92% occupancy because spinning is occupancy. Both the fix and the
  counter that caught it are in this release.

- **The merger's own profile, so "is the gate the bottleneck" stops being a
  guess.** `docs/efficiency.md` could say whether a run got faster and not why,
  and the counter a reader would reach for cannot answer it: `eval_seconds` sums
  across the evaluation pool, so eight threads scoring for a second each reads
  `8.0` whether the gate cost eight seconds or one. Three wall-clock counters on
  the one thread that merges now do — `merge_seconds` (merger occupancy),
  `merge_gate_seconds` (the part of it blocked on evaluation, a **subset**), and
  `worker_starved_seconds` (async only: workers held at the backpressure gate
  with a finished card and nowhere to put it), plus `EvolutionResult.gate_share()`
  and `.merger_occupancy()`.

  `merge_seconds` was not new. It has been declared in `MeterSnapshot`, in
  `Meter`, and in `Meter.snapshot()` since the first version of `metrics.py` and
  **written by nothing**, so every run ever published reported `0.0` — which is
  indistinguishable from "merging was free", and is the opposite of what the
  first measurement says.

  Measured (`python -m examples.efficiency --only stages`, `n_workers=8`):

  ```
  workload                             wall  merger busy  of it, gate  starved/s
  sync, uniform 20ms                   0.5s          59%         100%          —
  async, uniform 20ms                  4.1s          94%          82%       4.5x
  async, uniform 20ms, eval_conc=1     4.3s          99%          95%       6.9x
  sync, heavy tail                     1.9s          43%         100%          —
  async, heavy tail                    4.3s          90%          94%       5.6x
  ```

  The third row is the controlled one: same workload, same rollouts, same
  evaluations, only the gate's own pool narrows — and starvation rises from 4.5
  to 6.9 of eight workers. On the synchronous path `of it, gate` reads 100%,
  because everything the driver does after the barrier *is* evaluation.

  The counters also record a trap they were nearly written into. Starvation
  alone does not implicate the gate: with this domain's microsecond rollout the
  workers lap the merger between sweeps whatever the gate costs (at
  `async_ratio=8`, four workers starved for ~4s of a 1s window with nothing
  slowed down). `gate_share()` is what separates a busy merger from a blocking
  one, which is why the pair ships rather than either half.

### Fixed

- **The A/B harness could not tell "the rule did not help" from "the rule never
  ran".** `bench/ab_run.py` exists so that the three borrowed RL decision rules
  are validated before they are documented, and its own docstring promises to
  report "whether the mechanism fired at all". It reported `fusion.contested` —
  a *fusion* counter, non-zero whether or not an advantage was ever recorded —
  for **every** rule, so three of the four were audited by a counter belonging
  to the fourth. Each rule now carries its own: cards that arrived with an
  advantage, decisions that had a measured distance from `stable`, and whether
  the adaptive trust region ever left its initial value.
- **`--rule advantage` at the harness's own default would have measured almost
  nothing, and said nothing about it.** A group is one base version and one task
  cluster; `GroupAdvantage` returns `None` until the group *reaches*
  `min_group=4`, so the first three rollouts of every round can never carry a
  value. At `--workers 4` that is one rollout in four — measured at 25% with
  rewards that have spread, and 0% at `--workers 2`, where both arms are the
  control. The sweep now states this ceiling before it spends anything, refuses
  outright below `min_group`, and names the lever (`--workers`). `docs/concepts.md`
  attributed the missing signal to task clusters, which is the visible half and
  not the half that bites: the HotpotQA workload the harness runs carries no
  clusters at all.

### Added

- **`bench/ab_run.py --no-thinking`.** `completion_for` has honoured the flag for
  a while and this sweep had no way to pass it, so every A/B ran with reasoning
  tokens on. Measured on GLM-5.2 behind an Anthropic-shaped endpoint: 14.9 s and
  379 output tokens against 6.2 s and 44 with it disabled. That is a change to
  the model's output and not only its latency, so it has to be identical across
  arms — it is applied to both by construction, and recorded in the result JSON
  next to the model id, because a reader cannot otherwise tell which of the two
  models the arms were comparing.

### Fixed

- **A declared `Beam(k)` was `Beam(1)`, and `ParetoFrontier` never left the
  candidate it was handed first.** `PopulationAggregator` asks for one starting
  point per merge — the ledger holds one live head — and built its
  `SelectionContext` without `round=`, leaving it at 0 for every call. `MCTS` and
  `Archive` rotate on state they read per candidate (`selected`), so they
  coped; `Beam` and `ParetoFrontier` rank a pool and have nothing per-candidate
  to read, so they returned the same entry forever. Measured over eight
  selections against a growing archive:

  ```
  before                                  after
  Beam(4)               [3,3,3,3,...]     [3,2,1,0,3,2,1,0]
  Pareto(per_instance)  [0,0,0,0,...]     [0,3,0,3,0,3,0,3]
  Archive(novelty)      [2,2,2,1,2,2,2,2] [2,2,1,1,3,3,1,0]
  ```

  `ParetoFrontier` was the worse of the two: the front is built in archive
  order, so `front[0]` is the earliest admitted non-dominated candidate —
  usually the seed — and a run could watch far higher scorers arrive and keep
  expanding the seed, the exact opposite of what keeping a specialist is for.

  The layer now counts its selections and passes one, and `Beam` /
  `ParetoFrontier` offset their round-robin by it, so the walk is continuous
  across calls rather than restarted at the best member. `Archive`'s seeded rng
  was also rebuilding an identical `Random` every call for the same reason.

  **No published number moves**, and it is pinned rather than asserted: at
  `round == 0` the new expression is the old one (`(0 + i) % len == i % len`)
  and a test compares the concrete answers. Every configuration this repository
  runs is unaffected for a separate reason each — SICA's `Archive('best')` never
  draws, DGM and GEPA pass their own rng so `ctx.round` is unread, AFlow and
  PromptBreeder hold their own stateful rng, and ADAS and EvoSkill use `Beam(1)`
  whose single slot cannot rotate. The one configuration that changes is
  `Archive(sampling='novelty'|'performance'|'uniform', seed=N)` under the
  population layer, which nothing here uses and which was drawing the same
  random number every step.

### Added

- **`ParetoFrontier(mode="win_frequency")` — GEPA's Algorithm 2, shipped.**
  Per-instance winners → dominance pruning → a draw weighted by how many
  instances each survivor still wins, as
  `agentdescent.selection.pareto_win_frequency`. GEPA kept its own copy for a
  stated reason ("the shipped policy walks the frontier round-robin, GEPA
  samples it weighted by unique wins — close enough to look right and wrong
  enough to change a measured run"), and the reason was correct. The port now
  uses the shipped mode.
- **`Archive(sampling="sigmoid_novelty")` — DGM's `choose_selfimproves`,
  shipped**, as `agentdescent.selection.sigmoid_novelty_weights`. Same story:
  `novelty` is a temperature-1 softmax, DGM's is `sigmoid(10·(s−0.5))`, and a
  sigmoid at gain 10 is nearly a step where the softmax is nearly flat — they
  disagree most exactly where an archive spends its time. `examples/dgm` and
  `examples/adas` each carried a byte-identical copy of the formula; there is
  one now. ADAS keeps a function rather than the policy because it uses the same
  weights for a different draw: five entries without replacement to condition
  the meta-agent, against DGM's one parent.
- **`rng=` on `ParetoFrontier` and `Archive`.** A port migrating off a
  hand-written rule has to keep drawing from *its* rng in *its* order; a policy
  that re-seeded per call would agree on the distribution and disagree with
  every number the port has published. `seed=` still builds a fresh stream.
- **`tests/test_port_selection_equivalence.py`.** What makes "we did not change
  the upstream selection rule" a claim a test can check rather than a sentence a
  reader has to trust: the shipped mode and the rule it replaced are stepped
  through **one shared rng in lockstep**, 200 consecutive draws, and the
  frontier is compared against GEPA's own implementation rather than a
  transcription of it.

### Fixed

- **`ParetoFrontier(mode="per_instance")` claimed to be GEPA and is not.** It is
  plain Pareto walked round-robin: it keeps candidates that are best at nothing,
  and it does not weight the draw. `docs/port-fidelity.md` recorded GEPA as
  using it, which was wrong twice over — the port did not use the shipped policy
  at all. The mode keeps its behaviour and loses the claim; `win_frequency` is
  what the name meant. The same entry recorded DGM as `sampling="novelty"`,
  which was the softmax it specifically did not use.

### Changed

- **`Policies(selection=…)` now takes effect, from either driver.** The
  population layer that made a `SelectionPolicy` mean something on a one-branch
  ledger lived in `examples/_population.py` and was installed by the MethodPolicy
  runner, so one bundle field had three readings: `evolve()` refused it,
  `async_evolve()` dropped it, and the port runner honoured it through
  `aggregator_factory=`. It is `agentdescent.population` now, and `_build_engine`
  installs it for any declared policy, so both drivers run the same search. The
  runner keeps the factory route only for a port whose rule is not expressible as
  a `SelectionPolicy` at all — PromptBreeder's tournament evaluates and replaces,
  where a policy is handed cached scores and returns one of them.
- **`Beam(1)` is no longer the same *run* as `SingleHead`.** It is still the same
  answer on the pool `SingleHead` sees, and the tests pin that; but over an
  archive it restarts from the best scorer, which differs from "continue from the
  head" the moment the two differ. That is what beam search of width one means —
  before the population layer it had nowhere to show. No recorded measurement
  moves: every result in the repository was produced with no policy declared, or
  through the runner that already installed this layer.
- **A declared `selection` policy alongside `aggregator_factory=` is refused.**
  Both fill the aggregator seat, and resolving it silently would leave a caller
  who passed both with nothing to read that says which one ran.

### Fixed

- **A parent switch kept keys the chosen parent did not have.** The switch was
  written as `head.apply(Diff(ops=target_state))`, and `apply` only *sets* keys —
  so on a grow-only key space (`AppendRules`) every key the head had survived the
  switch, and "start from candidate C" quietly meant "start from C plus the
  incumbent". Exploration silently became hill climbing. It was unreachable in
  the ports, where every declared policy rides a fixed-key artifact, and moving
  the layer into the engine makes the pairing reachable by any caller. The switch
  now carries explicit `None` deletions for the keys the target does not have —
  the sentinel `apply` already understood and `diff` already emits.

### Removed

- **`_check_selection`.** It asked the policy once per round against a context
  holding one candidate and refused any answer but the head, which was the honest
  thing to do while nothing could honour a different answer. `PopulationAggregator`
  now can, for every policy and on both drivers, so the check had one live branch
  left. The refusal it existed for moved to `PopulationAggregator._offered`,
  where the menu is the real archive rather than a pool of one, and still raises
  `MultiHeadUnsupported`.

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

[Unreleased]: https://github.com/Birfy/agentdescent/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/Birfy/agentdescent/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Birfy/agentdescent/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Birfy/agentdescent/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/Birfy/agentdescent/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Birfy/agentdescent/compare/v0.4.6...v0.5.0
[0.4.6]: https://github.com/Birfy/agentdescent/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/Birfy/agentdescent/compare/v0.4.2...v0.4.5
[0.4.2]: https://github.com/Birfy/agentdescent/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Birfy/agentdescent/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Birfy/agentdescent/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Birfy/agentdescent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Birfy/agentdescent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Birfy/agentdescent/releases/tag/v0.1.0
