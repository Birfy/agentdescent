# API reference

Every name `agentdescent` exports, grouped by the module it comes from.
**Generated** from the package's own signatures and docstrings by
`python -m tools.gen_api_docs` — `tests/test_api_reference.py` fails if this
page and the code disagree, so a signature here is the signature you get.

A signature too long for its heading is printed in full below it, one
parameter per line, followed by a table of what each one does — type,
default, and the docstring's own prose. `*required*` in the default column
means the parameter has none.

Each section links to the page that explains *why* the module is shaped the
way it is; this page is the *what*.

219 public names across 34 modules.

---

## The loop

`evolve()`, the artifact, the actor, and what a run returns. &nbsp;·&nbsp; `agentdescent.evolution` &nbsp;·&nbsp; [guide](evolution.md)

### `Agent`

Convenience actor: bundles running a task and proposing an improvement.

### `EvolutionResult(...)`

```python
EvolutionResult(
    state: Dict[str, str],
    rendered: str,
    final_reward: float,
    history: List[RoundInfo],
    ledger_log: List[str],
    error: Optional[str] = None,
    stop_reason: str = 'rounds',
    forced_refreshes: int = 0,
    stragglers: int = 0,
    retired_workers: int = 0,
    usage: Usage = <factory>,
    wallclock: float = 0.0,
    rollouts: int = 0,
    rollout_seconds: float = 0.0,
    eval_seconds: float = 0.0,
    merge_seconds: float = 0.0,
    merge_gate_seconds: float = 0.0,
    worker_starved_seconds: float = 0.0,
    evals_skipped: int = 0,
    bounded_scans_cut: int = 0,
    stale_considered: int = 0,
    stale_discarded: int = 0,
    redispatched: int = 0,
    duplicates_dropped: int = 0,
    cas_conflicts: int = 0,
    cache_hits: int = 0,
    cache_misses: int = 0,
    sandbox_wait_s: float = 0.0,
    sandbox_setup_s: float = 0.0,
    sandboxes_created: int = 0,
    sandboxes_reused: int = 0,
    sandbox_failures: int = 0,
    fusion_trials: List['FusionTrial'] = <factory>
) -> None
```

| method | what it does |
|---|---|
| `cost_summary() -> str` | One line: what the run cost. Complements `outcomes()`, which says why it went as it did. |
| `cost_to_quality(target: float) -> Optional[int]` | Rollouts spent up to the first round that reached `target`. |
| `duplicate_rate() -> float` | Cache hits as a fraction of lookups -- work that did *not* have to be redone. In one process this is memoisation working; across processes it is the figure that says how much a shared cache would be worth. |
| `fusion_stats() -> 'FusionStats'` | How often merging beat the best single diff -- and how badly it lost. |
| `gate_share() -> float` | How much of the merger's busy time went to evaluation, in `[0, 1]`. |
| `load(path: str) -> 'EvolutionResult'` | Read back a result written by `save`. |
| `merger_occupancy() -> float` | Merger busy time over wall-clock. Above ~0.8 it is the critical path. |
| `outcomes() -> Dict[str, int]` | Merge outcomes for the whole run, by category -- *why* it went as it did. |
| `save(path: str) -> None` | Write the evolved artifact and its run summary to a JSON file. |
| `stale_rate() -> float` | Discarded evidence as a fraction of evidence considered; 0.0 if none. |
| `time_to_quality(target: float) -> Optional[float]` | Wall-clock at the first round whose held-out reward reached `target`. |
| `write_to(...)` | Install a file-tree artifact back into a real directory. |

### `EvolvingArtifact(...)`

An `Evolvable`: flat state + a strategy.

```python
EvolvingArtifact(
    id: str,
    state: Optional[Dict[str, str]] = None,
    version: int = 1,
    blast_radius: float = 0.2,
    runtime: Optional['_Runtime'] = None,
    strategy: Optional[Strategy] = None
) -> None
```

| method | what it does |
|---|---|
| `cheap_eval(evidence: EvidenceCard) -> float` | Score this artifact on the trajectories an evidence card carries. |
| `evidence_eval(evidence: EvidenceCard) -> float` | Score this artifact on the trajectories an evidence card carries. |
| `full_eval(task_set: Sequence[Task]) -> Dict[str, float]` | Score on a task set. No longer part of the `Evolvable` protocol -- the engine reaches ground truth through the verifier's `eval_fn` -- and kept because it is a convenient thing for a caller to have. |
| `score(tasks: Sequence[Task]) -> float` | Mean reward over `tasks`, evaluated concurrently. |
| `score_bounded(tasks: Sequence[Task], floor: float) -> float` | Mean reward, abandoned once it **provably** cannot exceed `floor`. |

### `FusionStats(...)`

The fusion tournament's record, with every denominator it needs.

```python
FusionStats(
    trials: int = 0,
    contested: int = 0,
    unranked: int = 0,
    single_candidate: int = 0,
    contradiction: int = 0,
    nothing_to_fuse: int = 0,
    dominant_single: int = 0,
    synthesis_failed: int = 0,
    synthesized_wins: int = 0,
    fused_wins: int = 0,
    single_wins: int = 0,
    neither: int = 0,
    ties: int = 0,
    mean_gain: float = 0.0,
    negative: int = 0,
    mean_loss: float = 0.0,
    worst_loss: float = 0.0,
    below_baseline: int = 0
) -> None
```

| method | what it does |
|---|---|
| `summary() -> str` | One line, and it says when there is nothing to report. |

### `LLMAgent(...)`

Adapt a `Completion` (from `agents`) into an `Agent`.

```python
LLMAgent(
    complete: Completion,
    solve_template: str = 'You are executing an artifact defined below.\n\n{artifact}\n\nApply it to this input and output ONLY the result, nothing else.\n\nInput:\n{prompt}',
    propose_template: str = "The artifact just failed a task (score {reward:.2f} out of 1.0).\n\nArtifact so far:\n{artifact}\n\nTask input:\n{prompt}\n\nIt produced:\n{output}\n{expected}\nPropose exactly ONE concise, general rule (a single imperative sentence) to improve the artifact for this and similar cases. State the rule in general terms -- it will be applied to other tasks, so do NOT mention this task's specific values or answer. Output only the rule text, or NONE if no rule would help.",
    show_meta: bool = True,
    meta_chars: int = 600,
    _empty_replies: int = 0
) -> None
```

### `ProposalContractError`

`propose` returned something that is not text (or `None`).

### `RewardContractError`

The caller's `reward` returned something outside the documented contract.

### `RoundInfo(...)`

```python
RoundInfo(
    round: int,
    held_out_reward: float,
    n_items: int,
    committed: int,
    rejected: int,
    reasons: Dict[str, int] = <factory>,
    elapsed_s: float = 0.0,
    rollouts: int = 0,
    calls: int = 0,
    considered: int = 0,
    discarded_stale: int = 0,
    conflicts_dropped: int = 0,
    fused: int = 0
) -> None
```

### `Task(id: str, prompt: str, meta: Dict[str, Any] = <factory>) -> None`

One unit of work the artifact is evaluated on.

### `claude_agent(model: str = 'claude-opus-4-8', max_tokens: int = 1024) -> LLMAgent`

Convenience: `LLMAgent(claude(model))` (provider code lives in `agents`).

### `evolve(...)`

Evolve an artifact. Provide either `agent` (with `solve`/`propose`) or the `run` / `propose` callables directly.

```python
evolve(
    tasks: Sequence[Task],
    reward: Reward,
    *,
    agent: Optional[Agent] = None,
    run: Optional[Run] = None,
    propose: Optional[Propose] = None,
    strategy: Optional[Strategy] = None,
    parallel: Optional['ParallelStrategy'] = None,
    task_sampler: Optional['TaskSampler'] = None,
    initial_state: Optional[Dict[str, str]] = None,
    blast_radius: float = 0.2,
    artifact_id: str = 'artifact',
    rounds: int = 15,
    n_workers: int = 4,
    max_concurrency: int = 1,
    refresh_interval: int = 1,
    round_timeout: Optional[float] = None,
    target_reward: Optional[float] = None,
    patience: Optional[int] = None,
    max_worker_errors: int = 3,
    eval_concurrency: int = 8,
    asynchronous: bool = False,
    async_ratio: int = 3,
    resync_on_commit: bool = True,
    pipelined_gate: bool = False,
    gate_workers: int = 2,
    max_seconds: Optional[float] = None,
    max_rollouts: Optional[int] = None,
    max_calls: Optional[int] = None,
    self_verify: bool = True,
    held_out_frac: float = 0.4,
    repo_path: Optional[str] = None,
    agg_config: Optional[AggregatorConfig] = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    aggregator_factory: Optional[AggregatorFactory] = None,
    oracle_budget: int = 200,
    cheap_eval_tasks: Optional[int] = None,
    fusion_tournament: Optional[bool] = None,
    solved_threshold: float = 0.999,
    shuffle: bool = False,
    seed: int = 0,
    on_round: Optional[Callable[['RoundInfo'], None]] = None,
    verbose: bool = False,
    usage: Optional[Usage] = None,
    policies: Optional['Policies'] = None
) -> EvolutionResult
```

| parameter | type | default | what it is |
|---|---|---|---|
| `tasks` | `Sequence[Task]` | *required* | The work the artifact is evaluated on. Split into train / held-out **by position** -- the last `held_out_frac` of the sequence is held out, in the order given. At least 4 are required and ids must be unique. |
| `reward` | `Reward` | *required* | `(task, output) -> [0, 1]`. Scores in `[0, 1]`; the engine treats `>= solved_threshold` as a pass (no proposal is requested). |
| `agent` | `Optional[Agent]` | `None` | An object with `solve` + `propose`. Provide this **or** `run` and `propose`; both signatures are checked before the first rollout. |
| `run` | `Optional[Run]` | `None` | `run(rendered, task) -> output` and `propose(rendered, task, output, reward) -> str \| None`. |
| `propose` | `Optional[Propose]` | `None` | As `run`. |
| `strategy` | `Optional[Strategy]` | `None` | How the artifact is represented and how a proposal becomes a `Diff`. |
| `parallel` | `Optional['ParallelStrategy']` | `None` | How a round's tasks are partitioned across workers. `DataParallel` (default) shards them; `TensorParallel(n_sections, keys=, route=)` also gives each worker a disjoint **section of the artifact** and rejects out-of-section edits -- counted as `section-violation` in `outcomes`. The pairing is validated before the first rollout: a strategy with no declared key space (`AppendRules`) or fewer keys than sections is refused rather than silently dropping most of its proposals. `PipelineParallel` raises (see above). |
| `task_sampler` | `Optional['TaskSampler']` | `None` | **Which** task a worker rolls out next, from its shard. Defaults to `RoundRobin`; use `DifficultyWeighted` to spend rollouts on tasks that still carry a learning signal. |
| `initial_state` | `Optional[Dict[str, str]]` | `None` | Seed the artifact instead of starting from `strategy.initial()`. Ignored when resuming an existing `repo_path`. |
| `blast_radius` | `float` | `0.2` | Governance layer, in `[0, 1]` (see above). |
| `artifact_id` | `str` | `'artifact'` | Name of the evolving artifact; becomes a filename, so it must match `[A-Za-z0-9_.-]+`. |
| `rounds` | `int` | `15` | Number of round barriers to run. Under `asynchronous=True` this becomes a worker-rollout budget of `rounds * n_workers` instead. |
| `n_workers` | `int` | `4` | Workers per round (`>= 1`). |
| `max_concurrency` | `int` | `1` | How many of them actually run at once (see above). |
| `refresh_interval` | `int` | `1` | How many rounds a worker keeps its ledger snapshot before taking the round's fresh one. `1` (default) is what this loop always did: every worker proposes against the current head, so a diff's staleness `eta` is **0 by construction** -- and that made `staleness_policy=` a knob with nothing to decide on this path (measured over an 8-round run: all 15 staleness decisions saw `eta=0` and returned ACCEPT, so Full, Guarded and Reflective were indistinguishable). Above `1`, workers hold a spread of versions -- the refresh is staggered by worker id -- so their diffs arrive with a spread of `eta` and the staleness policy, the `alpha` tolerances in `agg_config` and the `all-stale` outcome all become reachable synchronously. Costs no extra ledger read: a worker either adopts the snapshot the round already took, or keeps the older one it has. Ignored under `asynchronous=True`, where the lag budget is `async_ratio`. |
| `round_timeout` | `Optional[float]` | `None` | Seconds a round will wait for its concurrent workers before giving up on the slow ones. `None` (default) waits forever, which is what you want when every rollout is bounded -- but a single hung rollout then stalls the run, because the aggregator is a barrier. Abandoned work keeps running in the background (Python cannot cancel a thread) and is simply not waited for; it is reported when `verbose`. Only applies when `max_concurrency > 1`. |
| `target_reward` | `Optional[float]` | `None` | Stop as soon as held-out reward reaches this. Without it a run always spends all `rounds`, including after it has converged -- measured at 43% of rollouts wasted on an artifact that had stopped changing. |
| `patience` | `Optional[int]` | `None` | Stop after this many consecutive rounds with no improvement in held-out reward. `None` disables it. Cheap insurance for a run that plateaus below `target_reward`. |
| `max_worker_errors` | `int` | `3` | How much total failure to tolerate before giving up -- and only while *no* worker has ever completed a rollout, which reads as a misconfiguration (wrong key, dead endpoint). Once any worker has succeeded the backend demonstrably works, so failures are treated as transient and the run continues on whatever evidence it did gather. Counts consecutive failed rollouts per worker on the async path (see `result.retired_workers`) and consecutive rounds in which *every* worker failed on the sync path. |
| `eval_concurrency` | `int` | `8` | How many held-out tasks to score at once. Every gate goes through this -- each round's measurement and, far more often, the aggregator's per-candidate comparisons -- so it is the merge half of the run's parallelism, independent of `n_workers`. `1` restores the old sequential behaviour. |
| `asynchronous` | `bool` | `False` | Delegate to `async_evolve` -- no round barrier, with `async_ratio` as the staleness lag budget. |
| `async_ratio` | `int` | `3` | As `asynchronous`. |
| `resync_on_commit` | `bool` | `True` | Asynchronous path only. Refresh every worker's snapshot as soon as a sweep commits, so no one *starts* a rollout against a superseded artifact. See `async_evolve`, which documents what it does and does not fix -- a commit landing mid-rollout still produces a stale card. |
| `pipelined_gate` | `bool` | `False` | Under `asynchronous=True`, run a merge's **measurement** phase on its own threads instead of on the merger, so the merger goes back to draining while the gate runs. Off by default; documented in full on `async_evolve`, which implements it. Warns and does nothing on the synchronous path, where the round barrier idles every worker for the whole merge regardless. |
| `gate_workers` | `int` | `2` | As `pipelined_gate`. |
| `max_seconds` | `Optional[float]` | `None` | Wall-clock budget. `None` (default) means unbounded; the async path uses `20.0` when unset. |
| `max_rollouts` | `Optional[int]` | `None` | The budget in the two units a comparison has to hold fixed: rollouts completed, and actor invocations (`run` + `propose`). `rounds` is not one of them -- configurations differ in how much model a round buys, so a budget fixed in rounds hands the wider configuration more model and then reports the extra model as a win for parallelism. Either bound stops the run with `stop_reason` `"max_rollouts"` / `"max_calls"`. **Checked at the round barrier, so a run overshoots by up to one round.** A round is dispatched or it is not; stopping halfway would leave a half-merged round, and the states a comparison compares are the ones a merge produced. So a budget is a *bound on where to stop*, never the number to compare on: read the spend the run actually reported (`result.rollouts`, `result.usage.calls`), which is what `baselines` does -- it refuses to call two arms equal-budget when their measured spends differ. The async path has no barrier and enforces both per rollout, so it overshoots by at most the rollouts already in flight. |
| `max_calls` | `Optional[int]` | `None` | As `max_rollouts`. |
| `self_verify` | `bool` | `True` | Re-run the trajectory with the diff applied to record a local before/after delta. Doubles the rollouts spent per proposal; ports that score candidates only on held-out should pass `False`. |
| `held_out_frac` | `float` | `0.4` | Fraction of `tasks` reserved for held-out scoring, in `(0, 1)`. |
| `repo_path` | `Optional[str]` | `None` | Where the git-backed ledger lives. Omit for a throwaway repo that is removed when this call returns (not held until interpreter exit, so a sweep does not accumulate one git repo per run); **passing the same path again resumes** that ledger, and a caller-supplied path is never deleted. Git runs with an isolated config, so a personal `~/.gitconfig` (`commit.gpgsign`, `core.hooksPath`) cannot fail the ledger's own bookkeeping commits. |
| `agg_config` | `Optional[AggregatorConfig]` | `None` | Tuning for the reference aggregator (batching, acceptance risk, trust region, staleness tolerance). |
| `staleness_policy` | `Optional[StalenessPolicy]` | `None` | What to do with a diff proposed against an out-of-date version -- `full` / `guarded` (default) / `reflective`. |
| `aggregator_factory` | `Optional[AggregatorFactory]` | `None` | Replace the optimizer entirely; receives `(ledger, verifier, audit, config, staleness_policy)`. |
| `oracle_budget` | `int` | `200` | Hard cap on full held-out oracle evaluations during audits. Once spent, the verifier falls back to its cheap layer -- which only saves anything when `cheap_eval_tasks` makes that layer genuinely cheaper, so the two knobs go together. |
| `cheap_eval_tasks` | `Optional[int]` | `None` | How many held-out tasks the *cheap* layer scores when the aggregator is merely **ranking** candidates -- conflict resolution, and the fusion tournament when it is on. `None` (default) is **8**, or the whole held-out set when that is smaller. It used to mean the whole set unconditionally, which made the cheap layer cost exactly what the oracle costs: ranking one candidate bought a full sweep of real agent calls, and `oracle_budget`'s fallback saved nothing because it was the same measurement. Nothing in `bench/` or `examples/` ever passed this, so every real run paid it. The cost of the new default is **ranking resolution**: 8 binary-scored tasks resolve 0.125, so two candidates closer than that are ordered by whichever the sample happens to favour. That is bounded to *which* candidate goes forward -- both commit gates read `eval_counts` on the full set, so it cannot decide whether a change is safe. Pass `len(held_out)` to restore the exact behaviour. The sample is fixed for the run, so candidates are always compared like-for-like. |
| `fusion_tournament` | `Optional[bool]` | `None` | Rank the surviving diffs against their fusion before putting one forward. `None` (default) defers to `agg_config`, which is **off**. Off, because the ranking is paid every round while the only decision it changes from the acceptance gate's is recoverable: the union is a superset of every single diff, so committing it unranked loses no proposal. `DefaultFusion` carries the case analysis. On, because it is the only way to *measure* `win_rate` -- `best_single_score` exists only where a single was actually scored. That number is a property of the workload, not of the mechanism, so it is worth measuring per workload and not worth paying for on every run. |
| `solved_threshold` | `float` | `0.999` | A reward at or above this counts as solved, so no proposal is requested and the task sampler counts a pass. The default (`SOLVED`, 0.999) is right for a binary scorer. **Lower it for a graded one** -- a ROUGE score or an LLM judge rarely reaches 0.999, so every rollout would ask the reflector to "fix" an answer that scored 0.95, and the run reports `below-threshold` as if the reflector were the problem. |
| `shuffle` | `bool` | `False` | Shuffle `tasks` before that positional split. Off by default, which keeps a run reproducible and keeps `val_frac`'s promise that the engine's held-out split is exactly that `Dataset`'s `val`. Turn it on for **grouped** data -- anything ordered by category, source, difficulty or date -- where the tail of the file is a different distribution from the head, and every gate in the run (the acceptance test, `target_reward`, `final_reward`) would then be measured against it. |
| `seed` | `int` | `0` | As `shuffle`. |
| `on_round` | `Optional[Callable[['RoundInfo'], None]]` | `None` | Called with each `RoundInfo` as the round completes -- progress for a long run, which otherwise reports nothing until it returns. An exception raised here is reported but does not abort the run. |
| `verbose` | `bool` | `False` | Print a line per round. Independent of the `RuntimeWarning` emitted when a run ends early -- that always fires. |
| `usage` | `Optional[Usage]` | `None` | Share one `Usage` with your model adapters (`claude(usage=u)`, `openai_compatible(usage=u)`) and the result's token counts become real. Without it the run still reports calls, seconds and failures -- `run` is `(rendered, task) -> str`, so an opaque actor has no way to surface tokens, and inventing a number would be worse than reporting zero. |
| `policies` | `Optional['Policies']` | `None` | Bundle of replaceable pieces (`Policies`). Every field defaults to `None` meaning "current behaviour", so `Policies()` and passing nothing are the same run. The individual keyword arguments -- `task_sampler`, `staleness_policy`, `aggregator_factory` -- are shortcuts onto its fields and keep working; an explicit argument wins over a bundle default rather than being silently ignored. Fields whose implementations have not landed yet raise rather than being accepted and ignored: a caller who passes a custom acceptance rule and sees a finished run would reasonably conclude it ran. New capabilities go here rather than adding another parameter to a function that already has thirty-five. |

### `reflector(...)`

Use any model as the *reflector* for an agent you already have.

```python
reflector(
    complete: Completion,
    template: str = "The artifact just failed a task (score {reward:.2f} out of 1.0).\n\nArtifact so far:\n{artifact}\n\nTask input:\n{prompt}\n\nIt produced:\n{output}\n{expected}\nPropose exactly ONE concise, general rule (a single imperative sentence) to improve the artifact for this and similar cases. State the rule in general terms -- it will be applied to other tasks, so do NOT mention this task's specific values or answer. Output only the rule text, or NONE if no rule would help.",
    show_meta: bool = True
) -> Propose
```

### `tasks_from(...)`

Turn a list of dicts -- a dataset -- into `Task` objects.

```python
tasks_from(
    rows,
    prompt: str = 'prompt',
    gold: str = 'gold',
    id: Optional[str] = None,
    **meta_keys: str
) -> List['Task']
```

---

## Meta-evolution

Evolve a decision slot of `evolve()` itself, and validate it elsewhere. &nbsp;·&nbsp; `agentdescent.meta` &nbsp;·&nbsp; [guide](meta-evolution.md)

### `MetaOutcome(...)`

What one inner run did under a candidate slot value.

```python
MetaOutcome(
    curve: List[float] = <factory>,
    final: float = 0.0,
    rollouts: int = 0,
    detail: Dict[str, Any] = <factory>
) -> None
```

| method | what it does |
|---|---|
| `from_result(result: EvolutionResult, **detail: Any) -> 'MetaOutcome'` | Read an inner `EvolutionResult`. |

### `ParamSlot(...)`

The numeric hyper-parameters of any policy class, one key each.

```python
ParamSlot(
    factory: Callable[..., Any],
    params: Mapping[str, float],
    bounds: Mapping[str, Tuple[float, float]] = <factory>,
    title: str = '# Policy parameters',
    invalid_proposals: int = 0,
    _lock: threading.Lock = <factory>
) -> None
```

### `PrioritySelection(...)`

A `SelectionPolicy` driven by a `priority` rule.

```python
PrioritySelection(
    source: str = 'def priority(rank, visits, total, prior, depth, n_nodes):\n    # Flat PUCT (ERA, futs.py): exploit by rank, explore by visit count.\n    c = 1.0\n    return rank + c * (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)\n'
) -> None
```

### `SlotSpec`

A `Strategy` that also compiles.

### `SourceSlot(...)`

One slot of validated source, compiled by `build`.

```python
SourceSlot(
    initial_value: str = '',
    key: str = 'value',
    empty_render: str = '(no instruction yet)',
    min_chars: int = 1,
    validate: Optional[Callable[[str], str]] = None,
    build: Optional[Callable[[str], Any]] = None,
    description: str = 'The value is source text; reply with the complete revised text.',
    invalid_proposals: int = 0,
    _lock: threading.Lock = <factory>
) -> None
```

### `auc(outcome: MetaOutcome) -> float`

Mean best-so-far held-out reward over the inner run: how *fast* it rose.

### `compile_policy_source(...)`

Gate `source`, instantiate its `class_name`, and check it fits `slot`.

```python
compile_policy_source(
    slot: str,
    source: str,
    *,
    class_name: str = 'Policy',
    smoke: Optional[Callable[[Any], None]] = None
) -> Any
```

### `compile_priority(source: str) -> Callable[..., float]`

AST-gate `source` and return its `priority` function.

### `evolve_problem(...)`

An inner `evolve()` as a `Problem`.

```python
evolve_problem(
    tasks: Sequence[Task],
    reward: Callable[[Task, str], float],
    *,
    slot: str,
    base: Optional[Policies] = None,
    **evolve_kwargs: Any
) -> Problem
```

### `final_reward(outcome: MetaOutcome) -> float`

The inner run's own final held-out reward, clipped to `[0, 1]`.

### `meta_evolve(...)`

Evolve one decision slot of the engine against a set of inner problems.

```python
meta_evolve(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slot: str,
    spec: SlotSpec,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Completion] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0,),
    blast_radius: float = 0.6,
    artifact_id: str = 'policy-slot',
    **evolve_kwargs: Any
) -> EvolutionResult
```

| parameter | type | default | what it is |
|---|---|---|---|
| `problems` | `Union[Sequence[Problem], Mapping[str, Problem]]` | *required* | The inner problems, each `(value, seed) -> MetaOutcome` -- a list, or a mapping from a name to a problem (the name appears in task ids and in the reflector's prompt). `evolve_problem` builds one from the arguments of an inner `evolve()`. |
| `slot` | `str` | *required* | Which `Policies` field the value fills; one of `SLOTS`. Recorded, and checked -- machinery fields refuse. |
| `spec` | `SlotSpec` | *required* | How a value is represented, gated and compiled -- a `SlotSpec` such as `priority_selection` or a `ParamSlot`. |
| `propose` | `Optional[Callable[[str, Task, str, float], Optional[str]]]` | `None` | The reflector. Pass `propose` directly, or `model` to get `slot_reflector` over the spec. One of the two is required. |
| `model` | `Optional[Completion]` | `None` | As `propose`. |
| `meta_reward` | `Optional[MetaReward]` | `None` | `MetaOutcome` to `[0, 1]`; `None` is `auc`. |
| `seeds` | `Sequence[int]` | `(0,)` | Inner seeds per problem; each `(problem, seed)` pair is one outer task, so `len(problems) * len(seeds)` tasks in all, split into train and held-out by `held_out_frac` as `evolve()` always does. |
| `blast_radius` | `float` | `0.6` | Governance. `0.6` is L1: the value is a harness and every merge also passes the oracle. |
| `artifact_id` | `str` | `'policy-slot'` | As `blast_radius`. |
| `**evolve_kwargs` | `Any` |  | Everything else `evolve` takes -- `rounds`, `n_workers`, `max_concurrency`, `held_out_frac`, `max_rollouts` ... `strategy`, `run` and `reward` are this function's and cannot be passed. Returns the ordinary `EvolutionResult`; `spec.compile(result.rendered)` is the evolved value, and `result.rendered` is what to hand `meta_validate`. |

### `meta_validate(...)`

Score `before` and `after` on problems the outer loop never saw.

```python
meta_validate(
    spec: SlotSpec,
    before: str,
    after: str,
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    seeds: Sequence[int] = (0,),
    meta_reward: Optional[MetaReward] = None
) -> Dict[str, Dict[str, float]]
```

### `policy_source(...)`

The general spec: `slot`'s value is the source of a class satisfying its Protocol.

```python
policy_source(
    slot: str,
    seed: Optional[str] = None,
    *,
    class_name: str = 'Policy',
    smoke: Optional[Callable[[Any], None]] = None,
    notes: str = ''
) -> SourceSlot
```

### `priority_selection(...)`

The shipped spec for the `selection` slot of a tree search.

```python
priority_selection(
    seed: str = 'def priority(rank, visits, total, prior, depth, n_nodes):\n    # Flat PUCT (ERA, futs.py): exploit by rank, explore by visit count.\n    c = 1.0\n    return rank + c * (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)\n'
) -> SourceSlot
```

### `rollouts_to(target: float) -> MetaReward`

`1 / (1 + sweeps until the curve first reaches target)`; 0 if never.

### `seed_source(slot: str) -> str`

The engine's default behaviour for `slot`, as candidate source.

### `slot_reflector(...)`

A `propose` for `meta_evolve`: one model call per failing rollout.

```python
slot_reflector(
    complete: Completion,
    spec: SlotSpec,
    *,
    max_outcome_chars: int = 2000
) -> Callable[[str, Task, str, float], Optional[str]]
```

### `transfer_ratio(...)`

Gain on `target` over gain on `source`, from a `meta_validate` report.

```python
transfer_ratio(
    report: Mapping[str, Mapping[str, float]],
    source: str,
    target: str
) -> Optional[float]
```

---

## Agents and models

Any `prompt -> text` is a completion; a `WorkspaceAgent` also has a directory. &nbsp;·&nbsp; `agentdescent.agents` &nbsp;·&nbsp; [guide](agents.md)

### `AgentError`

A tool-using agent failed; the message carries its stderr / exit status.

### `Usage(...)`

What a run cost: calls, tokens, and wall-clock spent in the model.

```python
Usage(
    calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    seconds: float = 0.0,
    failures: int = 0,
    failure_seconds: float = 0.0,
    _lock: threading.Lock = <factory>
) -> None
```

| method | what it does |
|---|---|
| `estimated_cost(per_1m_prompt: float, per_1m_completion: float) -> float` | Cost at the given per-million-token prices (both provider-specific). |

### `WorkspaceAgent`

A `Completion` that can additionally be bound to a directory.

### `claude(...)`

A Claude-backed completion (requires `pip install anthropic` + creds).

```python
claude(
    model: str = 'claude-opus-4-8',
    max_tokens: int = 4096,
    client: Optional[object] = None,
    usage: Optional[Usage] = None,
    retries: int = 3,
    timeout: float = 120.0,
    **create_kwargs
) -> Completion
```

### `claude_code(...)`

Claude Code in non-interactive print mode, as a `Completion`.

```python
claude_code(
    *,
    workspace: Optional[str] = None,
    extra_args: Sequence[str] = (),
    **kwargs
) -> Completion
```

### `cli_agent(...)`

Run any **command-line** coding agent as a `Completion`.

```python
cli_agent(
    command: Sequence[str],
    *,
    workspace: Optional[str] = None,
    via_stdin: bool = False,
    timeout: float = 600.0,
    env: Optional[Dict[str, str]] = None,
    usage: Optional[Usage] = None
) -> 'WorkspaceAgent'
```

### `codex(...)`

OpenAI Codex CLI in non-interactive exec mode, as a `Completion`.

```python
codex(
    *,
    workspace: Optional[str] = None,
    extra_args: Sequence[str] = (),
    **kwargs
) -> Completion
```

### `echo(transform: Optional[Callable[[str], str]] = None) -> Completion`

A deterministic, no-network completion for tests and dry runs.

### `from_callable(fn: Completion) -> Completion`

Identity adapter -- documents that any `prompt -> text` callable works.

### `metered(completion: Completion, usage: Usage) -> Completion`

Count calls and model wall-clock for *any* completion.

### `openai_compatible(...)`

A completion for any OpenAI-compatible chat endpoint (GLM/Zhipu, proxies, local servers, OpenAI itself).

```python
openai_compatible(
    model: str,
    *,
    base_url_env: str = 'OPENAI_BASE_URL',
    api_key_env: str = 'OPENAI_API_KEY',
    default_base_url: str = 'https://api.openai.com/v1',
    max_tokens: int = 4096,
    timeout: float = 120.0,
    usage: Optional[Usage] = None,
    retries: int = 3,
    **create_kwargs
) -> Completion
```

### `with_retries(...)`

Wrap a completion with exponential-backoff retries on any exception.

```python
with_retries(
    completion: Completion,
    attempts: int = 3,
    backoff: float = 0.5,
    sleep: Callable[[float], None] = <built-in function sleep>
) -> Completion
```

---

## Directories as state

Load a directory into state, materialise it back, serialise it losslessly. &nbsp;·&nbsp; `agentdescent.filetree` &nbsp;·&nbsp; [guide](directory-evolution.md)

### `TreeError`

A directory could not be represented as evolvable state, or vice versa.

### `TreeSpec(...)`

Which files make up an evolvable tree, and how big it may get.

```python
TreeSpec(
    include: Sequence[str] = ('**/*.md', '**/*.txt', '**/*.py', '**/*.json', '**/*.yaml', '**/*.yml', '**/*.toml', '**/*.sh', '**/*.cfg', '**/*.ini'),
    exclude: Sequence[str] = ('**/.git/**', '**/__pycache__/**', '**/node_modules/**', '**/.venv/**', '**/*.egg-info/**', '**/.pytest_cache/**', '**/.DS_Store'),
    max_file_bytes: int = 28000,
    max_files: int = 200,
    max_total_bytes: int = 2000000
) -> None
```

| method | what it does |
|---|---|
| `validate_against(trust_region_chars: int) -> None` | Fail now if the loader admits files the optimizer can never accept. |

### `canonical(state: Mapping[str, str]) -> str`

A lossless, stable serialisation of a file tree.

### `load_tree(path: str, spec: Optional[TreeSpec] = None) -> Dict[str, str]`

Read a directory into `{relpath: text}`.

### `materialize(...)`

Write a tree into `dest` (optionally under `prefix`); return the paths.

```python
materialize(
    state: Mapping[str, str],
    dest: str,
    *,
    prefix: str = '',
    exec_patterns: Sequence[str] = ('**/*.sh', 'scripts/**', '**/bin/**')
) -> List[str]
```

### `parse_tree(rendered: str) -> Dict[str, str]`

The inverse of `canonical`.

### `tree_summary(state: Mapping[str, str], limit: int = 40) -> str`

A human/LLM-readable listing (paths + sizes), for prompts and logs.

---

## The file-tree strategy

One state key per file, plus the multi-file proposal protocol. &nbsp;·&nbsp; `agentdescent.treestrategy` &nbsp;·&nbsp; [guide](directory-evolution.md)

### `FileTree(...)`

The artifact **is a directory**; each state key is a relative file path.

```python
FileTree(
    initial_files: Mapping[str, str] = <factory>,
    editable: Sequence[str] = ('**',),
    frozen: Sequence[str] = (),
    max_files_per_diff: int = 2,
    max_file_bytes: int = 28000,
    planned_paths: Sequence[str] = ()
) -> None
```

| method | what it does |
|---|---|
| `frozen_files(source: Mapping[str, str]) -> Dict[str, str]` | The pristine content of every frozen path, for the runner's overlay. |
| `keys() -> Sequence[str]` | The declared key space, for `TensorParallel`. |
| `writable(path: str) -> bool` | May the loop write this path? `frozen` beats `editable`. |

### `parse_edits(proposal: str) -> Dict[str, Optional[str]]`

Parse a reflector reply into `{path: new_content}` (`None` = delete).

### `tree_reflector(...)`

A `propose` callable that asks `complete` for multi-file edits.

```python
tree_reflector(
    complete: Completion,
    *,
    strategy: 'FileTree',
    context_files: Sequence[str] = ('**/SKILL.md', '**/AGENT.md', '*.md'),
    max_context_chars: int = 12000,
    max_output_chars: int = 2000,
    template: str = 'You maintain the files below. An agent used them to do a task and did poorly (reward {reward:.2f} out of 1.00). Improve the files so this class of failure stops happening -- generalise, do not hard-code this one case.\n\nFILES IN THE ARTIFACT:\n{listing}\n\n{contents}\nTASK THE AGENT WAS GIVEN:\n{prompt}\n\nWHAT THE AGENT PRODUCED:\n{output}\n{expected}\n{protocol}'
) -> Any
```

---

## Runners

Give a real agent the candidate directory, one workspace per rollout. &nbsp;·&nbsp; `agentdescent.runners` &nbsp;·&nbsp; [guide](directory-evolution.md)

### `code_runner(...)`

Run **candidate code** on a task: materialise, gate, execute.

```python
code_runner(
    entrypoint: Sequence[str],
    *,
    layout: str = 'root',
    name: str = 'agent',
    setup_cmd: Optional[Sequence[str]] = None,
    test_cmd: Optional[Sequence[str]] = None,
    overlay: Optional[Mapping[str, str]] = None,
    fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
    timeout: float = 120.0,
    env: Optional[Mapping[str, str]] = None,
    workspace_root: Optional[str] = None,
    sandbox_pool: Optional['SandboxPool'] = None
) -> Callable[[str, Task], str]
```

### `gated_reward(reward: Callable[[Task, str], float]) -> Callable[[Task, str], float]`

`reward`, with a failed `code_runner` gate scoring 0.

### `tree_runner(...)`

Build a `run(rendered, task)` that gives `agent` the evolving directory.

```python
tree_runner(
    agent: Completion,
    *,
    layout: str = 'claude_skill',
    name: str = 'artifact',
    prompt_template: str = '{prompt}\n\n(The files under {tree_dir} in this directory are available to you; read them and follow them. Reply with only the final answer.)',
    overlay: Optional[Mapping[str, str]] = None,
    fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
    answer_file: Optional[str] = None,
    keep_failed: bool = False,
    workspace_root: Optional[str] = None,
    sandbox_pool: Optional['SandboxPool'] = None
) -> Callable[[str, Task], str]
```

---

## The data model

What a unit of evolution is, and what a gradient looks like here. &nbsp;·&nbsp; `agentdescent.evolvable` &nbsp;·&nbsp; [guide](data-model.md)

### `Contract(...)`

The externally-visible interface of an artifact.

```python
Contract(
    input_schema: str = 'any',
    output_schema: str = 'any',
    side_effects: Tuple[str, ...] = (),
    major: int = 1
) -> None
```

### `ContractError`

The caller's own code broke a documented contract.

### `Diff(...)`

A proposed change to an artifact's state.

```python
Diff(
    diff_id: str,
    target: str,
    ops: Dict[str, Any] = <factory>,
    contract_breaking: bool = False,
    author: str = 'unknown'
) -> None
```

| method | what it does |
|---|---|
| `size() -> int` | A crude "number of edited lines" proxy used by the trust-region cap (design doc, section 4.4). |

### `EvidenceCard(...)`

The "gradient metadata" carried by every diff (design doc, section 3.3).

```python
EvidenceCard(
    diff: Diff,
    base_version: VersionVector,
    touched: List[str],
    before_after_delta: float = 0.0,
    trajectory_refs: List[Any] = <factory>,
    advantage: Optional[float] = None,
    cost_tokens: int = 0,
    cost_wallclock: float = 0.0
) -> None
```

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

```python
Aggregator(
    ledger: Ledger,
    verifier: ThreeLayerVerifier,
    audit: AuditScheduler,
    config: Optional[AggregatorConfig] = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    meter: Optional['Meter'] = None,
    conflict: Optional['ConflictPolicy'] = None,
    fusion: Optional['FusionPolicy'] = None,
    acceptance: Optional['AcceptancePolicy'] = None,
    promotion: Optional['PromotionPolicy'] = None
) -> None
```

| method | what it does |
|---|---|
| `begin_step(*, skip_in_flight: bool = False) -> List[Union['_Candidate', MergeReport]]` | Phases 1 and 2: tick, drain what is ready, choose candidates. |
| `finalize() -> None` | Publish the current dev head to stable at the end of a clean run. |
| `finish_step(items: List[Union['_Candidate', MergeReport]]) -> List[MergeReport]` | Phase 3: decide the measured candidates, then age and promote. |
| `measure(items: List[Union['_Candidate', MergeReport]]) -> List[Union['_Candidate', MergeReport]]` | Phase 2 for a batch from `begin_step`. **Off-thread safe.** |
| `step() -> List[MergeReport]` | Fire every artifact bucket that is ready and return per-artifact reports. |

### `AggregatorConfig(...)`

```python
AggregatorConfig(
    batch_trigger: int = 4,
    max_wait_rounds: int = 3,
    base_delta: float = 0.5,
    alpha_head: int = 5,
    alpha_tail: int = 1,
    trust_region_ops: int = 6,
    trust_region_chars: int = 32000,
    trust_region_policy: Optional[Any] = None,
    promote_after_k: int = 3,
    anneal_half_life: int = 64,
    accept_samples: int = 4000,
    cas_attempts: int = 3,
    cas_backoff: float = 0.05,
    fusion_tournament: bool = False,
    bounded_gate: bool = False
) -> None
```

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

```python
MergeReport(
    artifact_id: str,
    accepted: Optional[Diff],
    fused: bool,
    considered: int,
    survived_staleness: int,
    discarded_stale: int,
    conflicts_dropped: int,
    prob_improve: float,
    committed_version: Optional[int],
    reason: str = '',
    category: str = ''
) -> None
```

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

```python
Ledger(
    repo_path: str,
    serialize: Serializer,
    deserialize: Deserializer,
    author: str = 'agentdescent <bot@agentdescent.local>'
) -> None
```

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

```python
ThreeLayerVerifier(
    eval_fn: EvalFn,
    held_out: Sequence,
    rule_subset: int = 8,
    learned_noise: float = 0.04,
    seed: int = 0,
    budget: VerifierBudget = <factory>
) -> None
```

| method | what it does |
|---|---|
| `cheap_eval(artifact: Evolvable) -> float` | The signal used everywhere a budget-free score is needed. |
| `eval_counts(artifact: Evolvable, floor: Optional[float] = None) -> Tuple[float, float]` | Return (successes, failures) on the full held-out set. |
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

```python
ClusterParallel(
    cluster_of: Callable[[str], str],
    c: float = 1.4,
    pass_threshold: Optional[float] = None,
    name: str = 'CP'
) -> None
```

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

```python
TensorParallel(
    n_sections: int,
    keys: Optional[Sequence[str]] = None,
    route: Optional[Callable[[str], str]] = None,
    name: str = 'TP'
) -> None
```

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

```python
DifficultyWeighted(
    c: float = 0.2,
    pass_threshold: Optional[float] = None,
    prior: float = 0.5
) -> None
```

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

## Candidate selection

Which candidate the next batch of workers starts from. &nbsp;·&nbsp; `agentdescent.selection` &nbsp;·&nbsp; [guide](selection.md)

### `Archive(...)`

DGM's and ADAS's archive sampling: performance, tempered by novelty.

```python
Archive(
    sampling: str = 'novelty',
    temperature: float = 1.0,
    seed: int = 0,
    rng: Optional['random.Random'] = None
) -> None
```

### `Beam(k: int = 1) -> None`

Keep the `k` best-scoring candidates and spread the workers over them.

### `Candidate(...)`

One starting point the next batch could be launched from.

```python
Candidate(
    artifact_id: str,
    version: int,
    state: Mapping[str, str] = <factory>,
    score: Optional[float] = None,
    per_task: Mapping[str, float] = <factory>,
    selected: int = 0,
    parent: Optional[int] = None,
    prior: Optional[float] = None
) -> None
```

### `MCTS(exploration: float = 1.4) -> None`

UCT over the candidate tree: one evolve step is one rollout.

### `MultiHeadUnsupported`

A policy named a starting point the ledger cannot hold yet.

### `ParetoFrontier(...)`

Three published frontier rules, as one class and one argument.

```python
ParetoFrontier(
    mode: str = 'per_instance',
    k: int = 5,
    seed: int = 0,
    rng: Optional['random.Random'] = None
) -> None
```

### `SelectionContext(...)`

What a `SelectionPolicy` is allowed to look at.

```python
SelectionContext(
    head: Candidate,
    candidates: Sequence[Candidate] = (),
    round: int = 0,
    n_workers: int = 1
) -> None
```

### `SelectionPolicy`

Given the candidates, return the `n` starting points for the next batch.

### `SingleHead()`

Every worker starts from the current head. Today's behaviour, exactly.

### `pareto_front(candidates: Sequence[Candidate], *, tasks: Sequence[str]) -> List[Candidate]`

Candidates no other candidate beats on every task and betters on one.

---

## The population layer

What makes a selection policy take effect on a one-branch ledger. &nbsp;·&nbsp; `agentdescent.population` &nbsp;·&nbsp; [guide](selection.md)

### `PopulationAggregator(...)`

The shipped merge pipeline plus an archive and a selection policy.

```python
PopulationAggregator(
    ledger,
    verifier,
    audit,
    config,
    staleness_policy = None,
    *,
    selection: SelectionPolicy,
    artifact_id: str,
    meter = None,
    conflict = None,
    fusion = None,
    acceptance = None,
    promotion = None
)
```

| method | what it does |
|---|---|
| `finalize() -> None` | Leave the best-scoring candidate on the head, then promote. |
| `step() -> List[MergeReport]` | Fire every artifact bucket that is ready and return per-artifact reports. |

### `population_factory(...)`

The `aggregator_factory=` adapter for one run.

```python
population_factory(
    selection: SelectionPolicy,
    artifact_id: str,
    *,
    meter = None,
    conflict = None,
    fusion = None,
    acceptance = None,
    promotion = None
)
```

---

## Model-assisted fusion

Combine competing values for the same key, when a dict update cannot. &nbsp;·&nbsp; `agentdescent.fusion` &nbsp;·&nbsp; [guide](aggregator.md)

### `KeepContradictions()`

A conflict policy that leaves contradicting diffs for fusion to resolve.

### `ReflectiveFusion(...)`

Combine contradicting diffs by asking a model to synthesise their values.

```python
ReflectiveFusion(
    complete,
    *,
    verifier: Any = None,
    max_chars: int = 8000,
    max_proposals: int = 6,
    validate: Optional[Callable[[Any], Any]] = None
) -> None
```

| method | what it does |
|---|---|
| `bind(verifier: Any) -> None` | Receive the engine's verifier, if the caller did not supply one. |
| `select(artifact: Evolvable, diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]` | Build the union and hand it straight to the acceptance gate. |

### `reflective_merge(complete, **kwargs) -> Dict[str, Any]`

The two policies model-merging needs, as `Policies` keyword arguments.

---

## Borrowed RL decision rules

Group-relative advantage, an adaptive trust region, distance from stable. &nbsp;·&nbsp; `agentdescent.advantage` &nbsp;·&nbsp; [guide](concepts.md)

### `AdaptiveTrustRegion(...)`

Widen the diff-size cap while merges land; tighten when they do not.

```python
AdaptiveTrustRegion(
    *,
    initial: TrustRegion = TrustRegion(ops=6, chars=32000),
    minimum: TrustRegion = TrustRegion(ops=1, chars=2000),
    maximum: TrustRegion = TrustRegion(ops=64, chars=512000),
    window: int = 10,
    widen: float = 1.25,
    tighten: float = 0.5,
    accept_rate_to_widen: float = 0.5
) -> None
```

| method | what it does |
|---|---|
| `observe(outcome: str) -> TrustRegion` | Record one merge outcome and return the region for the next merge. |

### `AdvantageAcceptance(inner = None, strength: float = 1.0) -> None`

Shift the acceptance prior by how well a proposal did against its group.

### `AdvantageConflict(inner = None, margin: float = 0.5) -> None`

Break a contradiction by group-relative advantage, not raw score.

### `GroupAdvantage(min_group: int = 4, max_groups: int = 4096) -> None`

Standardise a rollout's reward against the group it belongs to.

| method | what it does |
|---|---|
| `key(base_version: int, cluster: str = '') -> str` | The group a rollout belongs to. Same base, same cluster. |
| `observe(key: str, reward: float) -> Optional[float]` | Record a reward and return its advantage, or `None` if unknown yet. |

### `StableDistanceAcceptance(inner = None, strength: float = 0.1) -> None`

Penalise candidates that drift far from the confirmed branch.

### `TrustRegion(ops: int, chars: int) -> None`

How large one diff may be: operations, and characters.

### `state_distance(a, b) -> float`

Fraction of keys on which two artifact states differ, in `[0, 1]`.

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

```python
DurationEstimator(
    prior: float = 0.05,
    min_samples: int = 3,
    _n: int = 0,
    _sx: float = 0.0,
    _sy: float = 0.0,
    _sxx: float = 0.0,
    _sxy: float = 0.0,
    _lock: threading.Lock = <factory>
) -> None
```

### `ResumeQueue(p90_multiplier: float = 2.0) -> None`

Turn-level checkpoints of timed-out rollouts (partial rollout).

### `TaskCluster(...)`

```python
TaskCluster(
    id: str,
    tasks: List[Any],
    recent_value: float = 0.5,
    n_evidence: float = 0.0,
    pass_rate: float = 0.5
) -> None
```

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

```python
Dataset(
    train: List[Any] = <factory>,
    val: List[Any] = <factory>,
    test: List[Any] = <factory>,
    name: str = ''
) -> None
```

| method | what it does |
|---|---|
| `map(fn: Callable[[Any], Any]) -> 'Dataset'` | Apply `fn` to every item in every split, returning a new Dataset. |

### `split_dataset(...)`

Partition `items` into a `Dataset` by `ratios` (train, val, test).

```python
split_dataset(
    items: Sequence[Any],
    *,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    shuffle: bool = True,
    stratify_key: Optional[Callable[[Any], Any]] = None,
    name: str = ''
) -> Dataset
```

---

## Barrier-free evolution

`evolve()` without the round barrier. &nbsp;·&nbsp; `agentdescent.async_evolve` &nbsp;·&nbsp; [guide](async.md)

### `async_evolve(...)`

Evolve an artifact **without a round barrier**.

```python
async_evolve(
    tasks,
    reward: Reward,
    *,
    agent: Optional[Agent] = None,
    run: Optional[Run] = None,
    propose: Optional[Propose] = None,
    strategy: Optional[Strategy] = None,
    initial_state: Optional[Dict[str, str]] = None,
    blast_radius: float = 0.2,
    artifact_id: str = 'artifact',
    n_workers: int = 4,
    async_ratio: int = 3,
    resync_on_commit: bool = True,
    max_seconds: float = 20.0,
    max_iters: Optional[int] = None,
    max_calls: Optional[int] = None,
    target_reward: Optional[float] = None,
    patience: Optional[int] = None,
    max_worker_errors: int = 3,
    eval_concurrency: int = 8,
    pipelined_gate: bool = False,
    gate_workers: int = 2,
    held_out_frac: float = 0.4,
    repo_path: Optional[str] = None,
    agg_config = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    aggregator_factory = None,
    oracle_budget: int = 200,
    cheap_eval_tasks: Optional[int] = None,
    fusion_tournament: Optional[bool] = None,
    solved_threshold: float = 0.999,
    shuffle: bool = False,
    seed: int = 0,
    self_verify: bool = True,
    shutdown_grace: float = 2.0,
    stall_patience: int = 50,
    duration_estimator: Optional['DurationEstimator'] = None,
    straggler_factor: float = 3.0,
    task_sampler: Optional['TaskSampler'] = None,
    on_round: Optional[Callable[[RoundInfo], None]] = None,
    verbose: bool = False,
    usage: Optional[Usage] = None,
    policies: Optional['Policies'] = None
) -> EvolutionResult
```

| parameter | type | default | what it is |
|---|---|---|---|
| `tasks` |  | *required* | Exactly as in `evolve`, which documents them. (Listed rather than left to the paragraph above: a completeness check that reads prose cannot tell a documented parameter from a mentioned one.) |
| `reward` | `Reward` | *required* | As `tasks`. |
| `agent` | `Optional[Agent]` | `None` | As `tasks`. |
| `run` | `Optional[Run]` | `None` | As `tasks`. |
| `propose` | `Optional[Propose]` | `None` | As `tasks`. |
| `strategy` | `Optional[Strategy]` | `None` | As `tasks`. |
| `initial_state` | `Optional[Dict[str, str]]` | `None` | As `tasks`. |
| `blast_radius` | `float` | `0.2` | As `tasks`. |
| `artifact_id` | `str` | `'artifact'` | As `tasks`. |
| `n_workers` | `int` | `4` | Producer threads (`>= 1`). The train tasks are sharded round-robin across them; a worker with an empty shard is not started. |
| `async_ratio` | `int` | `3` | The lag budget, in two senses: a worker refreshes its snapshot once head drifts more than this far ahead, **and** it stops producing while more than this many cards sit un-merged. The second bound matters at cold start, before any commit has moved head. |
| `resync_on_commit` | `bool` | `True` | Refresh every worker as soon as a sweep commits, whatever the ratio. On by default: a worker that starts a rollout against a version a finished sweep has already replaced is doing work the merger will discard, and no workload wants that. This does **not** remove staleness where a real workload gets it. A worker snapshots, then spends the rollout in `run`, then pushes; a commit landing anywhere in that window makes the card stale no matter what the top of the loop does. What it removes is the other source: *starting* a rollout against a snapshot a finished sweep has already superseded. The two coincide only when rollouts are short relative to sweep cadence -- as they are in this repo's synthetic tests and bench workloads, where a rollout is a dictionary lookup and turning this on does collapse η to 0. Those are the cases that need `False`: anything measuring what the lag budget alone does has to switch this off, or the budget is no longer the only resync trigger and the measurement is of something else. Turn it on when the artifact's *content* is what workers reason from, so an out-of-date copy makes the work void rather than merely stale. Evolving a skill library from empty is the case that motivated it: the lag budget fires at `head_v - base_v > async_ratio`, so with the default 3 the first three commits leave every worker still proposing against no library at all, re-deriving what head already has for the merger to discard. |
| `max_seconds` | `float` | `20.0` | Wall-clock budget for the **production phase only**. Two things still happen after it, so budget for them: a bounded shutdown (`shutdown_grace`, since an in-flight rollout cannot be cancelled) and **one held-out scoring pass** to compute `final_reward`. That pass is memoised per (artifact, task), so it is free when the final head was already scored by a sweep and costs a full held-out sweep of the backend when it was not -- which is exactly the case when the budget was too short for any sweep to finish. |
| `max_iters` | `Optional[int]` | `None` | Stop after this many worker rollouts in total (a budget, not a barrier). |
| `max_calls` | `Optional[int]` | `None` | Stop after this many actor invocations (`run` + `propose`) in total. The second half of an equal-budget comparison: two configurations matched on rollouts still differ in model spend whenever one of them asks for more proposals per rollout, and the cheaper unit is the one a reader assumes was held fixed. Both bounds are checked as each rollout lands, so a run overshoots only by what was already in flight. |
| `target_reward` | `Optional[float]` | `None` | Stop as soon as a sweep's held-out reward reaches this. Compared against the real reward, never against an acceptance probability. |
| `patience` | `Optional[int]` | `None` | Stop after this many consecutive merge sweeps that fail to beat the best held-out reward seen so far. The async analogue of the synchronous knob: there are no round barriers here, so a *sweep* (one drain-and-merge by the merger) is the unit. `None` disables it. |
| `max_worker_errors` | `int` | `3` | Consecutive failed rollouts before a worker that has *never* succeeded gives up. Workers that have succeeded at least once never retire; they back off and keep trying until the run's own budget ends it. |
| `eval_concurrency` | `int` | `8` | How many held-out tasks the merger scores at once. `1` restores the old sequential behaviour. |
| `pipelined_gate` | `bool` | `False` | Run a merge's **measurement** phase on its own threads instead of on the merger. Off by default. The merger is one thread and it does three things per merge: drain and filter (cheap), score the base and the candidate (expensive), then accept, audit and commit (cheap). Measured on the stub workload, the middle phase is **94% of the merger's gate time** and the merger is ~90% busy, which leaves 4.5 of 8 workers blocked at the backpressure gate at any moment (`docs/efficiency.md`). This lets the merger go back to draining while the measurement runs, so the workers keep producing. **It changes no commit semantics.** At most one candidate per artifact is measured at a time, so every candidate is still committed against the head it was prepared and measured on -- there is no candidate-level staleness to have a policy about. Cards arriving meanwhile accumulate in the aggregator's buffer, which is what the buffer is for, so batches get larger rather than more numerous. Requires an aggregator with `begin_step` / `measure` / `finish_step` (the shipped one has them). A custom one that predates the seam warns and keeps the inline path. |
| `gate_workers` | `int` | `2` | Threads for the measurement phase when `pipelined_gate` is on. Bounded in practice by one candidate per artifact, so the default of 2 is enough for a single-artifact run with one measurement finishing as the next starts. This is a **third** pool -- `n_workers` rollouts, `gate_workers` measurements, each of which fans out over `eval_concurrency` tasks -- so the ceiling your provider sees is `n_workers + gate_workers * eval_concurrency`. |
| `held_out_frac` | `float` | `0.4` | As `tasks`. |
| `repo_path` | `Optional[str]` | `None` | As `tasks`. |
| `agg_config` |  | `None` | As `tasks`. |
| `staleness_policy` | `Optional[StalenessPolicy]` | `None` | As `tasks`. |
| `aggregator_factory` |  | `None` | As `tasks`. |
| `oracle_budget` | `int` | `200` | As `tasks`. |
| `cheap_eval_tasks` | `Optional[int]` | `None` | As in `evolve`: how many held-out tasks the cheap layer scores when ranking candidates. `None` is 8, or the whole held-out set when that is smaller. |
| `fusion_tournament` | `Optional[bool]` | `None` | As in `evolve`: rank the survivors against their fusion before putting one forward. `None` defers to `agg_config`, which is off. The cost/benefit is identical on this path -- there is one merger thread here too, and it pays the ranking on the critical path of every commit. |
| `solved_threshold` | `float` | `0.999` | As in `evolve`: the reward at which a task counts as solved and no proposal is requested. Lower it for a graded scorer. |
| `shuffle` | `bool` | `False` | As in `evolve`: shuffle before the positional train/held-out split. Off by default. |
| `seed` | `int` | `0` | As `shuffle`. |
| `self_verify` | `bool` | `True` | As in `evolve`. `False` skips the extra per-trajectory rollout, which is what ports that judge candidates only on held-out want. |
| `shutdown_grace` | `float` | `2.0` | Total seconds to wait for the worker and merger threads after the budget expires -- shared across all of them, not per thread. An in-flight rollout cannot be cancelled, so a slow backend can still overrun it; a warning says so and work already merged is kept. |
| `stall_patience` | `int` | `50` | Merger sweeps that may pass with cards arriving and nothing committing before every worker is forced to resync, regardless of `async_ratio`. Without it a lag budget larger than the staleness tolerance **livelocks** under the Guarded policy: workers propose against a snapshot too old for the policy to accept, every card is discarded, head never moves, and the lag budget therefore never triggers a refresh either. |
| `duration_estimator` | `Optional['DurationEstimator']` | `None` | Pass a `DurationEstimator` to fit `seconds ~ intercept + slope * len(prompt)` online and count rollouts that overran their own estimate by more than `straggler_factor` (`result.stragglers`). This is the design's **L-traj** mechanism, which until now lived only in the reference runtime and so was unreachable from the API a real workload uses. Detection only: resuming a partial rollout would need it to expose its turns, and `run(rendered, task) -> output` is opaque. |
| `straggler_factor` | `float` | `3.0` | As `duration_estimator`. |
| `task_sampler` | `Optional['TaskSampler']` | `None` | Which task a worker takes next from its shard. |
| `on_round` | `Optional[Callable[[RoundInfo], None]]` | `None` | Called with each `RoundInfo` as a merger sweep completes -- progress for a long run. It runs on the merger thread and must be cheap and thread-safe; an exception is reported, not fatal. |
| `verbose` | `bool` | `False` | Print one line per merger sweep. |
| `usage` | `Optional[Usage]` | `None` | Share one `Usage` with your model adapters (`claude(usage=u)`, `openai_compatible(usage=u)`) and the result's token counts become real. Without it the run still reports calls, seconds and failures -- `run` is `(rendered, task) -> str`, so an opaque actor has no way to surface tokens, and inventing a number would be worse than reporting zero. |
| `policies` | `Optional['Policies']` | `None` | Bundle of replaceable pieces (`Policies`). Every field defaults to `None` meaning "current behaviour", so `Policies()` and passing nothing are the same run. The individual keyword arguments -- `task_sampler`, `staleness_policy`, `aggregator_factory` -- are shortcuts onto its fields and keep working; an explicit argument wins over a bundle default rather than being silently ignored. Fields whose implementations have not landed yet raise rather than being accepted and ignored: a caller who passes a custom acceptance rule and sees a finished run would reasonably conclude it ran. New capabilities go here rather than adding another parameter to a function that already has thirty-five. |

---

## The async orchestrator

The reference barrier-free runtime and its statistics. &nbsp;·&nbsp; `agentdescent.async_runtime` &nbsp;·&nbsp; [guide](async.md)

### `AsyncAgentDescent(...)`

Barrier-free reference runtime, on the general engine.

```python
AsyncAgentDescent(
    repo_path: str,
    universe: TaskUniverse,
    config: Optional[AsyncConfig] = None,
    agg_config: Optional[AggregatorConfig] = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    estimator: Optional[DurationEstimator] = None,
    skill_id: str = 'mol-router',
    aggregator_factory = None,
    rollout = None
) -> None
```

| method | what it does |
|---|---|
| `buffer_pending() -> int` | Cards waiting in the aggregator's buckets, or 0 before a run. |

### `AsyncConfig(...)`

```python
AsyncConfig(
    n_workers: int = 6,
    async_ratio: int = 3,
    resync_on_commit: bool = True,
    noise: float = 0.15,
    target_accuracy: float = 0.98,
    max_seconds: float = 20.0,
    oracle_budget: int = 400,
    stall_patience: int = 150,
    duration_timeout_factor: float = 3.0,
    seed: int = 0,
    self_verify: bool = True
) -> None
```

### `AsyncStats(...)`

```python
AsyncStats(
    rollouts: int = 0,
    proposals: int = 0,
    sweeps: int = 0,
    commits: int = 0,
    fused: int = 0,
    discarded_stale: int = 0,
    conflicts_dropped: int = 0,
    forced_refreshes: int = 0,
    stragglers_checkpointed: int = 0,
    retired_workers: int = 0,
    oracle_used: int = 0,
    final_dev_accuracy: float = 0.0,
    final_stable_accuracy: float = 0.0,
    wallclock: float = 0.0,
    error: Optional[str] = None,
    timeline: List[Tuple[int, float]] = <factory>
) -> None
```

---

## The reference orchestrator

The round loop the research results were measured with. &nbsp;·&nbsp; `agentdescent.orchestrator` &nbsp;·&nbsp; [guide](orchestrator.md)

### `AgentDescent(...)`

The merge-based parallel self-evolution system, on the general engine.

```python
AgentDescent(
    repo_path: str,
    universe: TaskUniverse,
    n_workers: int = 6,
    noise: float = 0.15,
    refresh_interval: int = 2,
    skill_id: str = 'mol-router',
    config: Optional[AggregatorConfig] = None,
    oracle_budget: int = 300,
    seed: int = 0,
    staleness_policy = None,
    self_verify: bool = True
) -> None
```

### `RoundStat(...)`

```python
RoundStat(
    round: int,
    dev_accuracy: float,
    stable_accuracy: float,
    committed: int,
    fused: int,
    discarded_stale: int,
    conflicts_dropped: int,
    oracle_used: int
) -> None
```

### `run_fork_baseline(...)`

DGM-style archive/fork control: parallel but never merged (RQ1).

```python
run_fork_baseline(
    universe: TaskUniverse,
    n_workers: int = 6,
    noise: float = 0.15,
    rounds: int = 40,
    seed: int = 0
) -> float
```

---

## Document backends

A tool-using agent over a document that is too big for a prompt. &nbsp;·&nbsp; `agentdescent.backends` &nbsp;·&nbsp; [guide](backends.md)

### `AgentBackend`

A base agent that answers a question about a document, possibly using tools.

### `document_agent(...)`

Turn **any** `Completion` into an `AgentBackend` for document questions.

```python
document_agent(
    completion: Completion,
    *,
    doc_filename: str = 'document.txt',
    inline_chars: int = 200000,
    skills_dir: str = '.claude/skills'
) -> AgentBackend
```

### `openhands(...)`

A **real OpenHands agent** (SDK v1.x) as a workspace-bindable Completion.

```python
openhands(
    model: str = 'openai/deepseek-v4-pro',
    *,
    base_url: str = 'https://api.deepseek.com',
    api_key_env: str = 'OPENAI_API_KEY',
    temperature: float = 0.0,
    max_iterations: int = 40
) -> '_OpenHandsAgent'
```

### `openhands_backend(...)`

`document_agent(openhands(...))` -- the document task on OpenHands.

```python
openhands_backend(
    model: str = 'openai/deepseek-v4-pro',
    *,
    base_url: str = 'https://api.deepseek.com',
    api_key_env: str = 'OPENAI_API_KEY',
    temperature: float = 0.0,
    max_iterations: int = 40,
    doc_filename: str = 'document.txt'
) -> AgentBackend
```

### `tool_loop_backend(complete: Completion, *, max_steps: int = 5, window: int = 3) -> AgentBackend`

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

### `scorer(score) -> Callable`

Resolve `score` -- a name from `SCORERS` or a `(task, output) -> float` callable -- into the reward `evolve()` takes.

---

## Equal-budget baselines

merge-of-N against best-of-N fork and serial, on one rollout budget. &nbsp;·&nbsp; `agentdescent.baselines` &nbsp;·&nbsp; [guide](results.md)

### `ArmResult(...)`

One arm, one seed, and the spend it actually incurred.

```python
ArmResult(
    arm: str,
    seed: int,
    width: int,
    rollouts: int,
    calls: int,
    prompt_tokens: int,
    completion_tokens: int,
    wallclock: float,
    wallclock_parallel: float,
    dev_reward: float,
    test_reward: Optional[float],
    test_oracle: Optional[float] = None,
    forks: Tuple[ForkOutcome, ...] = (),
    stop_reason: str = '',
    error: Optional[str] = None,
    fusion: Optional['FusionStats'] = None
) -> None
```

### `Budget(rollouts: int, calls: Optional[int] = None) -> None`

What every arm is allowed to spend.

| method | what it does |
|---|---|
| `split(ways: int) -> 'Budget'` | The share of this budget one of `ways` independent runs may spend. |

### `Comparison(...)`

Several seeds of several arms, and whether they are comparable at all.

```python
Comparison(
    arms: Dict[str, List[ArmResult]],
    fixed: str = 'rollouts',
    unequal: List[Tuple[str, str, float, float]] = <factory>,
    confounded: List[Tuple[str, str, float, float]] = <factory>,
    tolerance: float = 0.1
) -> None
```

| method | what it does |
|---|---|
| `scored(arm: str) -> int` | Seeds of `arm` that produced a test score at all. |
| `separates(a: str, b: str, *, min_seeds: int = 3) -> bool` | Whether `a`'s seeds are all above `b`'s, with no overlap. |
| `spread(arm: str) -> Optional[Tuple[float, float, float]]` | (min, median, max) test quality. Not a confidence interval. |
| `underpowered(*arms: str, min_seeds: int = 3) -> bool` | Whether any named arm has too few seeds to support a comparison. |

### `ForkOutcome(...)`

One member of a fork arm, kept so the selection step can be audited.

```python
ForkOutcome(
    seed: int,
    dev_reward: float,
    test_reward: Optional[float],
    rollouts: int,
    calls: int
) -> None
```

### `Workload(...)`

The half of the comparison that must not vary, in one object.

```python
Workload(
    tasks: Sequence[Task],
    reward: Reward,
    test_eval: Callable[[EvolutionResult], float],
    agent: Optional[Any] = None,
    run: Optional[Any] = None,
    propose: Optional[Any] = None,
    strategy: Optional[Any] = None,
    evolve_kwargs: Mapping[str, Any] = <factory>
) -> None
```

### `best_of_n_fork(...)`

N runs that never see each other, each on its share of the budget.

```python
best_of_n_fork(
    workload: Workload,
    n: int,
    *,
    budget: Budget,
    seed: int = 0,
    concurrency: int = 1
) -> ArmResult
```

### `compare(...)`

Group arm results by arm and check what the comparison actually held fixed.

```python
compare(
    results: Sequence[ArmResult],
    *,
    fixed: str = 'rollouts',
    tolerance: float = 0.1
) -> Comparison
```

### `merge_of_n(...)`

N workers proposing into one artifact, merged every round. The claim.

```python
merge_of_n(
    workload: Workload,
    n: int,
    *,
    budget: Budget,
    seed: int = 0,
    usage: Optional[Usage] = None
) -> ArmResult
```

### `serial(...)`

One worker, improving itself in sequence. The floor.

```python
serial(
    workload: Workload,
    *,
    budget: Budget,
    seed: int = 0,
    usage: Optional[Usage] = None
) -> ArmResult
```

### `to_markdown(comparison: Comparison) -> str`

A table whose caption cannot claim more than the numbers support.

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

### `FusionTrial`

One tournament: what the fused candidate scored against the best single.

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

### `PRIORITY_SEED`

str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str

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

### `SCORERS`

dict() -> new empty dictionary dict(mapping) -> new dictionary initialized from a mapping object's (key, value) pairs dict(iterable) -> new dictionary initialized as if via: d = {} for k, v in iterable: d[k] = v dict(**kwargs) -> new dictionary initialized with the name=value pairs in the keyword argument list. For example: dict(one=1, two=2)

### `SLOTS`

Built-in immutable sequence.

### `SLOT_PROTOCOLS`

dict() -> new empty dictionary dict(mapping) -> new dictionary initialized from a mapping object's (key, value) pairs dict(iterable) -> new dictionary initialized as if via: d = {} for k, v in iterable: d[k] = v dict(**kwargs) -> new dictionary initialized with the name=value pairs in the keyword argument list. For example: dict(one=1, two=2)

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

### `baselines`

The control every efficiency number in this repository is missing.

### `dataloader`

Dependency-free dataset loading -- the *data layer* for examples/experiments.

### `rule_id`

Content-address a proposal so identical proposals dedupe automatically.
