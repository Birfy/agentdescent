# API reference

Every name `agentdescent` exports, grouped by the module it comes from.
**Generated** from the package's own signatures and docstrings by
`python -m tools.gen_api_docs` — `tests/test_api_reference.py` fails if this
page and the code disagree, so a signature here is the signature you get.

Each section links to the page that explains *why* the module is shaped the
way it is; this page is the *what*.

165 public names across 31 modules.

---

## The loop

`evolve()`, the artifact, the actor, and what a run returns. &nbsp;·&nbsp; `agentdescent.evolution` &nbsp;·&nbsp; [guide](evolution.md)

### `Agent`

Convenience actor: bundles running a task and proposing an improvement.

### `EvolutionResult(...)`

| method | what it does |
|---|---|
| `cost_summary() -> str` | One line: what the run cost. Complements `outcomes()`, which says why it went as it did. |
| `cost_to_quality(target: float) -> Optional[int]` | Rollouts spent up to the first round that reached `target`. |
| `duplicate_rate() -> float` | Cache hits as a fraction of lookups -- work that did *not* have to be redone. In one process this is memoisation working; across processes it is the figure that says how much a shared cache would be worth. |
| `outcomes() -> Dict[str, int]` | Merge outcomes for the whole run, by category -- *why* it went as it did. |
| `save(path: str) -> None` | Write the evolved artifact and its run summary to a JSON file. |
| `stale_rate() -> float` | Discarded evidence as a fraction of evidence considered; 0.0 if none. |
| `time_to_quality(target: float) -> Optional[float]` | Wall-clock at the first round whose held-out reward reached `target`. |
| `write_to(...)` | Install a file-tree artifact back into a real directory. |

### `EvolvingArtifact(...)`

An `Evolvable`: flat state + a strategy.

| method | what it does |
|---|---|
| `cheap_eval(evidence: EvidenceCard) -> float` | Score this artifact on the trajectories an evidence card carries. |
| `evidence_eval(evidence: EvidenceCard) -> float` | Score this artifact on the trajectories an evidence card carries. |
| `full_eval(task_set: Sequence[Task]) -> Dict[str, float]` | Score on a task set. No longer part of the `Evolvable` protocol -- the engine reaches ground truth through the verifier's `eval_fn` -- and kept because it is a convenient thing for a caller to have. |
| `score(tasks: Sequence[Task]) -> float` | Mean reward over `tasks`, evaluated concurrently. |

### `LLMAgent(...)`

Adapt a `Completion` (from `agents`) into an `Agent`.

### `ProposalContractError`

`propose` returned something that is not text (or `None`).

### `RewardContractError`

The caller's `reward` returned something outside the documented contract.

### `RoundInfo(...)`

### `Task(id: str, prompt: str, meta: Dict[str, Any] = <factory>) -> None`

One unit of work the artifact is evaluated on.

### `claude_agent(model: str = 'claude-opus-4-8', max_tokens: int = 1024) -> LLMAgent`

Convenience: `LLMAgent(claude(model))` (provider code lives in `agents`).

### `evolve(...)`

Evolve an artifact. Provide either `agent` (with `solve`/`propose`) or the `run` / `propose` callables directly.

### `reflector(...)`

Use any model as the *reflector* for an agent you already have.

### `tasks_from(...)`

Turn a list of dicts -- a dataset -- into `Task` objects.

---

## One-call skill evolution

The shortest path from a dataset to an evolved instruction. &nbsp;·&nbsp; `agentdescent.skill` &nbsp;·&nbsp; [guide](quickstart-skill.md)

### `evolve_skill(...)`

Evolve one instruction (a "skill") against a dataset, in one call.

---

## One-call directory evolution

The same, for a skill folder, an agent folder, or its code. &nbsp;·&nbsp; `agentdescent.skilldir` &nbsp;·&nbsp; [guide](directory-evolution.md)

### `evolve_agent_code(...)`

Evolve **agent code**: the tree is executed, and a test gate guards it.

### `evolve_agent_dir(...)`

Evolve an **agent directory** (subagent definitions, tool config, harness).

### `evolve_skill_dir(...)`

Evolve a **skill directory**, executed by a real agent that reads it.

---

## Agents and models

Any `prompt -> text` is a completion; a `WorkspaceAgent` also has a directory. &nbsp;·&nbsp; `agentdescent.agents` &nbsp;·&nbsp; [guide](agents.md)

### `AgentError`

A tool-using agent failed; the message carries its stderr / exit status.

### `Usage(...)`

What a run cost: calls, tokens, and wall-clock spent in the model.

| method | what it does |
|---|---|
| `estimated_cost(per_1m_prompt: float, per_1m_completion: float) -> float` | Cost at the given per-million-token prices (both provider-specific). |

### `WorkspaceAgent`

A `Completion` that can additionally be bound to a directory.

### `claude(...)`

A Claude-backed completion (requires `pip install anthropic` + creds).

### `claude_code(...)`

Claude Code in non-interactive print mode, as a `Completion`.

### `cli_agent(...)`

Run any **command-line** coding agent as a `Completion`.

### `codex(...)`

OpenAI Codex CLI in non-interactive exec mode, as a `Completion`.

### `echo(transform: Optional[Callable[[str], str]] = None) -> Completion`

A deterministic, no-network completion for tests and dry runs.

### `from_callable(fn: Completion) -> Completion`

Identity adapter -- documents that any `prompt -> text` callable works.

### `metered(completion: Completion, usage: Usage) -> Completion`

Count calls and model wall-clock for *any* completion.

### `openai_compatible(...)`

A completion for any OpenAI-compatible chat endpoint (GLM/Zhipu, proxies, local servers, OpenAI itself).

### `with_retries(...)`

Wrap a completion with exponential-backoff retries on any exception.

---

## Directories as state

Load a directory into state, materialise it back, serialise it losslessly. &nbsp;·&nbsp; `agentdescent.filetree` &nbsp;·&nbsp; [guide](directory-evolution.md)

### `TreeError`

A directory could not be represented as evolvable state, or vice versa.

### `TreeSpec(...)`

Which files make up an evolvable tree, and how big it may get.

| method | what it does |
|---|---|
| `validate_against(trust_region_chars: int) -> None` | Fail now if the loader admits files the optimizer can never accept. |

### `canonical(state: Mapping[str, str]) -> str`

A lossless, stable serialisation of a file tree.

### `load_tree(path: str, spec: Optional[TreeSpec] = None) -> Dict[str, str]`

Read a directory into `{relpath: text}`.

### `materialize(...)`

Write a tree into `dest` (optionally under `prefix`); return the paths.

### `parse_tree(rendered: str) -> Dict[str, str]`

The inverse of `canonical`.

### `tree_summary(state: Mapping[str, str], limit: int = 40) -> str`

A human/LLM-readable listing (paths + sizes), for prompts and logs.

---

## The file-tree strategy

One state key per file, plus the multi-file proposal protocol. &nbsp;·&nbsp; `agentdescent.treestrategy` &nbsp;·&nbsp; [guide](directory-evolution.md)

### `FileTree(...)`

The artifact **is a directory**; each state key is a relative file path.

| method | what it does |
|---|---|
| `frozen_files(source: Mapping[str, str]) -> Dict[str, str]` | The pristine content of every frozen path, for the runner's overlay. |
| `keys() -> Sequence[str]` | The declared key space, for `TensorParallel`. |
| `writable(path: str) -> bool` | May the loop write this path? `frozen` beats `editable`. |

### `parse_edits(proposal: str) -> Dict[str, Optional[str]]`

Parse a reflector reply into `{path: new_content}` (`None` = delete).

### `tree_reflector(...)`

A `propose` callable that asks `complete` for multi-file edits.

---

## Runners

Give a real agent the candidate directory, one workspace per rollout. &nbsp;·&nbsp; `agentdescent.runners` &nbsp;·&nbsp; [guide](directory-evolution.md)

### `code_runner(...)`

Run **candidate code** on a task: materialise, gate, execute.

### `tree_runner(...)`

Build a `run(rendered, task)` that gives `agent` the evolving directory.

---

## The data model

What a unit of evolution is, and what a gradient looks like here. &nbsp;·&nbsp; `agentdescent.evolvable` &nbsp;·&nbsp; [guide](data-model.md)

### `Contract(...)`

The externally-visible interface of an artifact.

### `ContractError`

The caller's own code broke a documented contract.

### `Diff(...)`

A proposed change to an artifact's state.

| method | what it does |
|---|---|
| `size() -> int` | A crude "number of edited lines" proxy used by the trust-region cap (design doc, section 4.4). |

### `EvidenceCard(...)`

The "gradient metadata" carried by every diff (design doc, section 3.3).

| method | what it does |
|---|---|
| `rebased_onto(head: VersionVector) -> 'EvidenceCard'` | Return a copy whose base is advanced to `head` for touched keys. |

### `Evolvable`

The single interface every unit of evolution must satisfy.

### `stable_hash(key: Any) -> int`

A process-independent hash for seeding and partitioning.

### `vv_dominates(a: VersionVector, b: VersionVector) -> bool`

Return True if `a` is at least as new as `b` on every shared key.

### `vv_staleness(head: VersionVector, base: VersionVector) -> int`

Per-diff staleness `eta` (design doc, section 4.2).

---

## The aggregator (the optimizer)

Staleness filter, conflict resolution, fusion, acceptance, commit. &nbsp;·&nbsp; `agentdescent.aggregator` &nbsp;·&nbsp; [guide](aggregator.md)

### `Aggregator(...)`

Per-artifact optimizer step over the ledger.

| method | what it does |
|---|---|
| `finalize() -> None` | Publish the current dev head to stable at the end of a clean run. |
| `step() -> List[MergeReport]` | Fire every artifact bucket that is ready and return per-artifact reports. |

### `AggregatorConfig(...)`

### `AggregatorContractError`

A custom aggregator returned something `step()` may not return.

### `AggregatorProtocol`

The contract a custom aggregator must satisfy to plug into `evolve`.

### `EvidenceBuffer() -> None`

Cards bucketed by target artifact (design doc, section 4.1).

| method | what it does |
|---|---|
| `settle(cards: List[EvidenceCard]) -> None` | Keep discarded-diff evidence addressable, under a hard bound. |

### `MergeOutcome`

The vocabulary of `category`.

| member | value |
|---|---|
| `COMMITTED` | `'committed'` |
| `BELOW_THRESHOLD` | `'below-threshold'` |
| `ALL_STALE` | `'all-stale'` |
| `OVERSIZED` | `'oversized'` |
| `ORACLE_REJECTED` | `'oracle-rejected'` |
| `CAS_CONFLICT` | `'cas-conflict'` |
| `UNKNOWN_ARTIFACT` | `'unknown-artifact'` |

### `MergeReport(...)`

### `diffs_conflict(a: Diff, b: Diff) -> bool`

Syntactic overlap: do two diffs edit an overlapping set of keys?

### `diffs_contradict(a: Diff, b: Diff) -> bool`

Semantic contradiction: same key, different proposed value.

### `fuse_diffs(diffs: List[Diff]) -> Diff`

Merge complementary (non-contradicting) diffs into one candidate.

---

## The ledger

The git-backed, compare-and-swap artifact store. &nbsp;·&nbsp; `agentdescent.ledger` &nbsp;·&nbsp; [guide](ledger.md)

### `CASConflict`

Raised when a commit's declared base version is stale.

### `ContractRejected`

Raised when a commit would change an artifact's contract major.

### `GitError`

A git command failed; the message carries git's own stderr.

### `Ledger(...)`

A git-backed, version-vectored artifact store with dual branches.

| method | what it does |
|---|---|
| `close() -> None` | Refuse further use of this ledger. Idempotent. |
| `commit(...)` | Compare-and-swap commit of a single artifact. |
| `commit_atomic(...)` | Two-phase, all-or-nothing commit of several artifacts. |
| `promote_to_stable(artifact_id: str) -> Optional[int]` | EMA-style confirmation: copy dev's current artifact onto stable. |
| `register(artifact: Evolvable, branch: str = 'dev') -> None` | Add a brand-new artifact at version 1 on both branches. |
| `snapshot(branch: str = 'dev') -> Snapshot` | Materialize every artifact on `branch` into live Evolvables. |

### `Snapshot(artifacts: Dict[str, Evolvable], version: VersionVector) -> None`

An immutable view of one branch at one point in time.

---

## The verifier

Rule / learned / oracle, and the budget that bounds the expensive one. &nbsp;·&nbsp; `agentdescent.verifier` &nbsp;·&nbsp; [guide](verifier.md)

### `ThreeLayerVerifier(...)`

Rule / learned / oracle backend for the aggregator.

| method | what it does |
|---|---|
| `cheap_eval(artifact: Evolvable) -> float` | The signal used everywhere a budget-free score is needed. |
| `eval_counts(artifact: Evolvable) -> Tuple[float, float]` | Return (successes, failures) on the full held-out set. |
| `learned_eval(artifact: Evolvable) -> Tuple[float, float]` | Noisy proxy that also returns an uncertainty estimate. |
| `oracle_eval(artifact: Evolvable) -> float` | Ground truth on the full held-out set. Consumes audit budget. |
| `rule_eval(artifact: Evolvable) -> float` | Cheap, deterministic-ish check on a tiny subset. |

### `VerifierBudget(oracle_calls_remaining: int = 200, oracle_calls_used: int = 0) -> None`

Oracle call budget, consumed by `oracle_eval`.

---

## Governance

L0 frozen / L1 slow / L2 fast, assigned by blast radius. &nbsp;·&nbsp; `agentdescent.governance` &nbsp;·&nbsp; [guide](governance.md)

### `GovernanceError`

Raised when the evolution loop tries to mutate a frozen (L0) artifact.

### `L1SerialGate(_in_flight: Dict[str, str] = None, _lock: threading.Lock = <factory>) -> None`

Enforces "at most one L1 diff in evaluation at a time" (section 6).

### `Layer`

| member | value |
|---|---|
| `L2_FAST` | `2` |
| `L1_SLOW` | `1` |
| `L0_FROZEN` | `0` |

### `assert_mutable(artifact: Evolvable) -> None`

Guard invoked before applying any diff (design doc, section 6, L0).

### `classify(artifact: Evolvable) -> Layer`

Assign an artifact to a governance layer.

---

## Staleness policies

What to do with a diff proposed against a version that has moved. &nbsp;·&nbsp; `agentdescent.staleness` &nbsp;·&nbsp; [guide](staleness.md)

### `FullStaleness()`

Use stale diffs directly regardless of `eta` (max throughput).

### `GuardedStaleness()`

Version-gated with rebase in the middle band (AgentDescent's default).

### `ReflectiveStaleness()`

Always rebase + re-verify; discard only if the improvement no longer holds.

### `StaleAction`

What the aggregator should do with a (possibly stale) evidence card.

| member | value |
|---|---|
| `ACCEPT` | `'accept'` |
| `REBASE` | `'rebase'` |
| `DISCARD` | `'discard'` |

### `StalenessPolicy`

### `get_policy(name: str) -> StalenessPolicy`

---

## Parallelism methods

How a round's work is split across workers: DP / TP / PP. &nbsp;·&nbsp; `agentdescent.parallel` &nbsp;·&nbsp; [guide](parallelism.md)

### `ClusterParallel(...)`

DP over task **clusters**, leased by UCB instead of sharded round-robin.

| method | what it does |
|---|---|
| `observe(unit: WorkUnit, task_id: str, score: float) -> None` | Feed one rollout's outcome back into the cluster's UCB estimate. |

### `DataParallel(name: str = 'DP') -> None`

DP -- every worker holds the same artifact; the *tasks* (keys) are sharded across workers and their diffs are merged. Coverage rotates each round.

### `ParallelMode`

| member | value |
|---|---|
| `DP` | `'data_parallel'` |
| `TP` | `'tensor_parallel'` |
| `PP` | `'pipeline_parallel'` |

### `ParallelStrategy`

How a round of work is partitioned across `n_workers`.

### `PipelineChain(stages: List[str]) -> None`

An ordered artifact dependency chain, upstream -> downstream.

| method | what it does |
|---|---|
| `blame(stage_success: Dict[str, bool]) -> Optional[str]` | Back-propagate blame to the *earliest* failing stage. |
| `counterfactual_pairs(stage: str) -> List[Tuple[str, str]]` | The {old x new} version swaps to replay for minimal factor analysis. |

### `PipelineParallel(stages: Sequence[str], name: str = 'PP') -> None`

PP -- artifacts form a dependency chain; each worker drives one stage, and a downstream failure back-propagates blame to the earliest failing stage (via `PipelineChain`).

### `SectionViolation`

Raised when a worker's diff touches a key outside its section.

### `TensorParallel(...)`

TP -- one hot artifact is split into `n_sections` disjoint sections; each worker owns a section, so edits are conflict-free *by construction* and the merge is a union (concatenation + a consistency check).

| method | what it does |
|---|---|
| `section_map() -> Dict[str, int]` | `artifact key -> section`. Empty when no key space was declared. |

### `TensorParallelMerge(n_sections: int, keys: Optional[Sequence[str]] = None) -> None`

Merge section-scoped diffs into one artifact (concatenation + review).

| method | what it does |
|---|---|
| `merge(base: Evolvable, section_diffs: List[Tuple[int, Diff]]) -> Tuple[Evolvable, bool]` | Return (merged_artifact, consistency_ok). |
| `owner_of(key: str) -> int` | Which section owns `key` -- via the declared partition when there is one. |

### `WorkUnit(worker: int, keys: List[str], stage: int = 0, section: Optional[int] = None) -> None`

What one worker is responsible for in one round of a parallel plan.

### `assign_key_sections(keys: Sequence[str], n_sections: int) -> Dict[str, int]`

Partition a **known** artifact key space into balanced, disjoint sections.

### `assign_sections(worker_ids: Sequence[str], n_sections: int) -> Dict[str, int]`

Authorize each worker for exactly one section (round-robin).

### `section_of(key: str, n_sections: int) -> int`

Hash an artifact key to a section id.

### `shard_round_robin(items: Sequence, n_shards: int) -> List[List]`

Split a task list into `n_shards` disjoint shards, round-robin.

---

## Task sampling

Which task a worker rolls out next. &nbsp;·&nbsp; `agentdescent.sampling` &nbsp;·&nbsp; [guide](sampling.md)

### `DifficultyWeighted(...)`

UCB over tasks, weighted by how much learning signal each one carries.

| method | what it does |
|---|---|
| `stats() -> Dict[str, Tuple[float, float]]` | Copy of the per-task (passes, trials) counters -- for inspection/tests. |

### `RoundRobin()`

Cycle through the shard in order -- the deterministic default.

### `TaskSampler`

Chooses the next task id for a worker, and learns from the outcome.

| method | what it does |
|---|---|
| `pick(keys: Sequence[str], round_index: int) -> str` | Return one task id from `keys` (never mutate `keys`). |
| `record(task_id: str, score: float) -> None` | Report the reward a rollout of `task_id` achieved (0..1). |

---

## Scheduling and audits

Duration-aware dispatch, straggler handling, and the oracle audit queue. &nbsp;·&nbsp; `agentdescent.scheduler` &nbsp;·&nbsp; [guide](duration-scheduling.md)

### `AuditScheduler(max_queued: int = 4096, collect: bool = False) -> None`

Allocates oracle budget by estimated value G-hat (design doc, 5.3).

| method | what it does |
|---|---|
| `force_oracle(blast_radius: float, artifact_id: str) -> bool` | High-impact or low-trust changes are forced through the oracle. |
| `update_trust(artifact_id: str, oracle_agreed: bool) -> None` | Raise trust when cheap eval agreed with the oracle, lower it when not. |

### `DurationEstimator(...)`

Predicts a rollout's wall-clock cost from a task's *size* (e.g. prompt length), calibrated online from observed rollouts.

### `ResumeQueue(p90_multiplier: float = 2.0) -> None`

Turn-level checkpoints of timed-out rollouts (partial rollout).

### `TaskCluster(...)`

### `TaskScheduler(clusters: List[TaskCluster], c: float = 1.4) -> None`

UCB over task clusters, with a difficulty (zero-advantage) filter.

| method | what it does |
|---|---|
| `lease_one() -> TaskCluster` | Atomically pick the single highest-UCB cluster (async worker pull). |
| `lease_round_robin() -> TaskCluster` | Async worker pull that spreads concurrent workers across clusters. |
| `select_batch(k: int) -> List[TaskCluster]` | Lease `k` clusters to workers, UCB-ordered, cycling if `k` exceeds the number of clusters. |

### `fifo_makespan(weights: List[float], n_workers: int) -> float`

Makespan of naive round-robin dispatch (the baseline LPT improves on).

### `lpt_schedule(weights: List[float], n_workers: int) -> Tuple[List[int], float]`

Longest-Processing-Time-first assignment of items to workers.

---

## The data layer

Datasets, splits, and cached fetches from HuggingFace or raw URLs. &nbsp;·&nbsp; `agentdescent.dataloader` &nbsp;·&nbsp; [guide](dataloader.md)

### `Dataset(...)`

A dataset partitioned into **train / val / test** splits.

| method | what it does |
|---|---|
| `map(fn: Callable[[Any], Any]) -> 'Dataset'` | Apply `fn` to every item in every split, returning a new Dataset. |

### `split_dataset(...)`

Partition `items` into a `Dataset` by `ratios` (train, val, test).

---

## Barrier-free evolution

`evolve()` without the round barrier. &nbsp;·&nbsp; `agentdescent.async_evolve` &nbsp;·&nbsp; [guide](async.md)

### `async_evolve(...)`

Evolve an artifact **without a round barrier**.

---

## The async orchestrator

The reference barrier-free runtime and its statistics. &nbsp;·&nbsp; `agentdescent.async_runtime` &nbsp;·&nbsp; [guide](async.md)

### `AsyncAgentDescent(...)`

### `AsyncConfig(...)`

### `AsyncStats(...)`

---

## The reference orchestrator

The round loop the research results were measured with. &nbsp;·&nbsp; `agentdescent.orchestrator` &nbsp;·&nbsp; [guide](orchestrator.md)

### `AgentDescent(...)`

The merge-based parallel self-evolution system, on the general engine.

### `RoundStat(...)`

### `run_fork_baseline(...)`

DGM-style archive/fork control: parallel but never merged (RQ1).

---

## The worker

One worker's rollout and proposal. &nbsp;·&nbsp; `agentdescent.worker` &nbsp;·&nbsp; [guide](orchestrator.md)

### `Worker(...)`

| method | what it does |
|---|---|
| `run(...)` | One rollout: classify tasks, propose a corrective diff for failures. |

---

## Document backends

A tool-using agent over a document that is too big for a prompt. &nbsp;·&nbsp; `agentdescent.backends` &nbsp;·&nbsp; [guide](backends.md)

### `AgentBackend`

A base agent that answers a question about a document, possibly using tools.

### `document_agent(...)`

Turn **any** `Completion` into an `AgentBackend` for document questions.

### `openhands(...)`

A **real OpenHands agent** (SDK v1.x) as a workspace-bindable Completion.

### `openhands_backend(...)`

`document_agent(openhands(...))` -- the document task on OpenHands.

### `tool_loop_backend(...)`

A dependency-free `grep`/`read` ReAct loop over the document.

---

## Ready-made scorers

The reward functions everyone writes, with the details right. &nbsp;·&nbsp; `agentdescent.rewards` &nbsp;·&nbsp; [guide](rewards.md)

### `contains(gold_key: str = 'gold', *, normalise: bool = True) -> Callable`

1.0 when the gold answer appears anywhere in the output.

### `exact_match(gold_key: str = 'gold', *, normalise: bool = True) -> Callable`

1.0 when the output equals the gold answer.

### `last_number(gold_key: str = 'gold', *, tolerance: float = 0.0) -> Callable`

1.0 when the **last** number in the output matches the gold number.

### `numeric_close(gold_key: str = 'gold', *, tolerance: float = 0.01) -> Callable`

`last_number` with a relative tolerance -- for rounded answers.

---

## Type aliases and constants

Values rather than classes or functions.

### `AcceptDecision`

Commit or not, and -- when not -- which of the merge categories it was.

### `AcceptancePolicy`

Whether a candidate is committed.

### `AggregatorFactory`

`(ledger, verifier, audit, config, policy) -> AggregatorProtocol` — how a custom optimizer is installed.

### `AppendRules`

Accumulate a deduped list of rules/lessons (append-only, content-addressed).

### `CacheProtocol`

Somewhere to keep evaluations. In one process, across many, or on disk.

### `Completion`

`Callable[[str], str]` — the one contract every model and agent satisfies.

### `ConflictPolicy`

Which of a batch of mutually contradictory changes survive.

### `EDIT_PROTOCOL`

The multi-file proposal format a `FileTree` reflector is told to emit.

### `Executor`

Runs rollouts somewhere. Threads here, processes and hosts later.

### `FAST_MAX`

The L2/L1 blast-radius boundary (`0.30`).

### `FROZEN_IDS`

Artifact ids the loop may read but never mutate (L0).

### `FileCache`

A directory of evaluations, so separate processes can share them.

### `FusionPolicy`

How complementary diffs become one candidate.

### `KeyedRules`

One entry per *category*: competing proposals contradict and are resolved.

### `LAYOUTS`

Where a runner writes the evolving tree inside a workspace (`claude_skill`, `skill_library`, `claude_agent`, `root`).

### `LedgerFailure`

The exception tuple a caller catches to treat any ledger problem as recoverable.

### `LedgerProtocol`

Seven methods: four the aggregator calls, three more the engine calls.

### `LocalWorkspaceSandbox`

A throwaway directory on this machine -- what a rollout has always got.

### `MemoryCache`

In-process, single-flight, counted.

### `MergeContext`

Everything an `AcceptancePolicy` is allowed to look at.

### `Policies`

Every replaceable piece, in one argument.

### `ProcessExecutor`

Persistent worker processes, with re-dispatch when one dies.

### `Promotion`

One artifact the `PromotionPolicy` believes `stable` should hold.

### `PromotionPolicy`

Which artifacts `dev` has proved well enough to copy onto `stable`.

### `ProposalContext`

What a `ProposalPolicy` is given for one rollout.

### `ProposalPolicy`

How a rollout becomes candidate changes.

### `Ref`

A callable named rather than sent: `"module:attribute"` plus config.

### `RefError`

A reference could not be resolved, and why -- never a bare ImportError.

### `Result`

What one rollout produced, or why it did not.

### `RolloutSpec`

One rollout, described completely enough to run somewhere else.

### `SOLVED`

Reward at or above which a task counts as solved (`0.999`). Lower it for a graded scorer, or every rollout asks the reflector to fix an answer that was already good.

### `Sandbox`

One acquired execution environment.

### `SandboxPool`

The single gate on how many sandboxes exist at once.

### `SandboxProvider`

Where sandboxes come from and go back to.

### `SandboxSpec`

What environment one rollout needs. Must survive JSON: it crosses processes.

### `SharedSandboxPool`

A pool whose ceiling is the machine's, not this process's.

### `SingleSlot`

The artifact **is one value**, and each accepted proposal replaces it.

### `Strategy`

Defines *what evolves and how* -- the representation and the merge rule.

### `TEST_FAILURE_MARKER`

Prefix of the output `code_runner` produces when the frozen gate fails, so the failure scores 0 and the reflector can read it.

### `ThreadExecutor`

The default: a bounded pool of threads in this process.

### `VerifierProtocol`

Four methods, from `grep 'self\.verifier\.' agentdescent/aggregator.py`.

### `VersionVector`

`Dict[str, int]` — artifact id to version.

### `WorkspaceProvider`

Provisions `LocalWorkspaceSandbox` -- `mkdtemp`, plus a lease file.

### `backends`

Agentic backends -- a base agent that *navigates documents with tools*, not just maps a prompt to text.

### `dataloader`

Dependency-free dataset loading -- the *data layer* for examples/experiments.

### `rule_id`

Content-address a proposal so identical proposals dedupe automatically.
