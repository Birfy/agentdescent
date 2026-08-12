"""Barrier-free asynchronous evolution -- the async twin of :func:`evolve`.

:func:`evolve` runs a round barrier: every worker steps, then a single
``aggregator.step()`` fires, then the next round begins. This module removes the
barrier while taking the **exact same plug-ins** as :func:`evolve`
(``run`` / ``reward`` / ``propose`` / ``strategy`` / ``aggregator_factory``), so
*any* task that runs under :func:`evolve` -- ACE, GEPA, EvoSkill, SkillOpt, ADAS,
DGM -- also runs here, asynchronously.

Two stages run as independent threads connected by a thread-safe intake buffer
(stage orchestration, FlashEvolve):

* **Workers** (``n_workers`` threads) hold a ledger snapshot and keep producing
  evidence cards against it -- rollout -> propose -> ``to_diff`` -> push. A worker
  refreshes its snapshot only once head has drifted more than ``async_ratio``
  versions ahead of it (ROLL Flash's *global lag budget*), so staleness (η > 0)
  genuinely arises: small ratio -> near-synchronous, large ratio -> highly async.
* **One merger** drains the buffer, runs each card through the active
  :class:`~agentdescent.staleness.StalenessPolicy` (ACCEPT η=0 / REBASE+re-verify /
  DISCARD), feeds the survivors to ``aggregator.ingest`` and calls
  ``aggregator.step()``. It is the **only** writer to the ledger, so there are no
  CAS conflicts, and every aggregator sees only rebased (η=0) cards -- which is
  why the custom optimizers work here unchanged.

The GIL means threads are not CPU-parallel, but the pipeline *overlap* (workers
producing while the merger merges) and the concurrency-control machinery (buffer
lock, CAS, per-diff staleness) are real, and the same shape drives a genuinely
parallel process/host pool. See ``examples/*`` (``--async``) and
:func:`~agentdescent.evolution.evolve` (``asynchronous=True``).
"""

from __future__ import annotations

import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .advantage import GroupAdvantage
from .agents import Usage
from .policies import Policies
from .evolution import (
    _publish_stable, _safe_log,
    Agent, EvolutionResult, Propose, Reward, RoundInfo, Run, Strategy, Task,
    _ASYNC_WIRED_POLICIES, _cost_fields, _fusion_trials, _resolve_policies,
    SOLVED, _build_engine, _checked_proposal, _checked_reward,
)
from .aggregator import Aggregator, AggregatorConfig, check_reports
from .evolvable import ContractError, EvidenceCard, vv_staleness
from .ledger import Ledger, LedgerFailure
from .sampling import RoundRobin, TaskSampler
from .scheduler import DurationEstimator
from .selection import SingleHead
from .pipeline import EarlyStop, FirstError, StallGuard, WorkerHealth
from .staleness import StaleAction, StalenessPolicy, get_policy


def async_evolve(
    tasks,
    reward: Reward,
    *,
    agent: Optional[Agent] = None,
    run: Optional[Run] = None,
    propose: Optional[Propose] = None,
    strategy: Optional[Strategy] = None,
    initial_state: Optional[Dict[str, str]] = None,
    blast_radius: float = 0.2,
    artifact_id: str = "artifact",
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
    agg_config=None,
    staleness_policy: Optional[StalenessPolicy] = None,
    aggregator_factory=None,
    oracle_budget: int = 200,
    cheap_eval_tasks: Optional[int] = None,
    fusion_tournament: Optional[bool] = None,
    solved_threshold: float = SOLVED,
    shuffle: bool = False,
    seed: int = 0,
    self_verify: bool = True,
    shutdown_grace: float = 2.0,
    stall_patience: int = 50,
    duration_estimator: Optional["DurationEstimator"] = None,
    straggler_factor: float = 3.0,
    task_sampler: Optional["TaskSampler"] = None,
    on_round: Optional[Callable[[RoundInfo], None]] = None,
    verbose: bool = False,
    #: Share one `Usage` with your model adapters (`claude(usage=u)`) and the
    #: result's token counts become real; without it only calls and seconds are
    #: known, because an opaque `run` cannot report tokens.
    usage: Optional[Usage] = None,
    policies: Optional["Policies"] = None,
) -> EvolutionResult:
    """Evolve an artifact **without a round barrier**.

    Same actor/strategy/aggregator plug-ins as :func:`~agentdescent.evolution.evolve`.
    ``async_ratio`` is the lag budget: a worker keeps proposing against its
    snapshot until head drifts past it, then refreshes -- so stale diffs (η > 0)
    arise and the ``staleness_policy`` (default ``guarded``) rebases or discards
    them. The run stops at ``max_seconds`` (or ``max_iters`` worker rollouts, or
    when held-out reward reaches ``target_reward``, or stalls for ``patience``
    sweeps).

    ``tasks``, ``reward``, ``agent``, ``run``, ``propose``, ``strategy``,
    ``initial_state``, ``blast_radius``, ``artifact_id``, ``held_out_frac``,
    ``repo_path``, ``agg_config``, ``staleness_policy``, ``aggregator_factory``
    and ``oracle_budget`` mean exactly what they do in
    :func:`~agentdescent.evolution.evolve`, which documents them. The parameters
    below are the ones specific to running without a barrier.

    Parameters
    ----------
    tasks, reward, agent, run, propose, strategy, initial_state, blast_radius, artifact_id, held_out_frac, repo_path, agg_config, staleness_policy, aggregator_factory, oracle_budget:
        Exactly as in :func:`~agentdescent.evolution.evolve`, which documents them.
        (Listed rather than left to the paragraph above: a completeness check that
        reads prose cannot tell a documented parameter from a mentioned one.)
    n_workers:
        Producer threads (``>= 1``). The train tasks are sharded round-robin
        across them; a worker with an empty shard is not started.
    async_ratio:
        The lag budget, in two senses: a worker refreshes its snapshot once head
        drifts more than this far ahead, **and** it stops producing while more
        than this many cards sit un-merged. The second bound matters at cold
        start, before any commit has moved head.
    resync_on_commit:
        Refresh every worker as soon as a sweep commits, whatever the ratio.
        On by default: a worker that starts a rollout against a version a
        finished sweep has already replaced is doing work the merger will
        discard, and no workload wants that.

        This does **not** remove staleness where a real workload gets it. A
        worker snapshots, then spends the rollout in ``run``, then pushes; a
        commit landing anywhere in that window makes the card stale no matter
        what the top of the loop does. What it removes is the other source:
        *starting* a rollout against a snapshot a finished sweep has already
        superseded. The two coincide only when rollouts are short relative to
        sweep cadence -- as they are in this repo's synthetic tests and bench
        workloads, where a rollout is a dictionary lookup and turning this on
        does collapse η to 0. Those are the cases that need ``False``: anything
        measuring what the lag budget alone does has to switch this off, or the
        budget is no longer the only resync trigger and the measurement is of
        something else.

        Turn it on when the artifact's *content* is what workers reason from,
        so an out-of-date copy makes the work void rather than merely stale.
        Evolving a skill library from empty is the case that motivated it: the
        lag budget fires at ``head_v - base_v > async_ratio``, so with the
        default 3 the first three commits leave every worker still proposing
        against no library at all, re-deriving what head already has for the
        merger to discard.
    max_seconds:
        Wall-clock budget for the **production phase only**. Two things still
        happen after it, so budget for them: a bounded shutdown
        (``shutdown_grace``, since an in-flight rollout cannot be cancelled) and
        **one held-out scoring pass** to compute ``final_reward``. That pass is
        memoised per (artifact, task), so it is free when the final head was
        already scored by a sweep and costs a full held-out sweep of the backend
        when it was not -- which is exactly the case when the budget was too
        short for any sweep to finish.
    max_iters:
        Stop after this many worker rollouts in total (a budget, not a barrier).
    max_calls:
        Stop after this many actor invocations (``run`` + ``propose``) in total.
        The second half of an equal-budget comparison: two configurations matched
        on rollouts still differ in model spend whenever one of them asks for
        more proposals per rollout, and the cheaper unit is the one a reader
        assumes was held fixed. Both bounds are checked as each rollout lands, so
        a run overshoots only by what was already in flight.
    eval_concurrency:
        How many held-out tasks the merger scores at once. ``1`` restores the old
        sequential behaviour.
    pipelined_gate:
        Run a merge's **measurement** phase on its own threads instead of on the
        merger. Off by default.

        The merger is one thread and it does three things per merge: drain and
        filter (cheap), score the base and the candidate (expensive), then
        accept, audit and commit (cheap). Measured on the stub workload, the
        middle phase is **94% of the merger's gate time** and the merger is
        ~90% busy, which leaves 4.5 of 8 workers blocked at the backpressure
        gate at any moment (``docs/efficiency.md``). This lets the merger go
        back to draining while the measurement runs, so the workers keep
        producing.

        **It changes no commit semantics.** At most one candidate per artifact
        is measured at a time, so every candidate is still committed against the
        head it was prepared and measured on -- there is no candidate-level
        staleness to have a policy about. Cards arriving meanwhile accumulate in
        the aggregator's buffer, which is what the buffer is for, so batches get
        larger rather than more numerous.

        Requires an aggregator with ``begin_step`` / ``measure`` / ``finish_step``
        (the shipped one has them). A custom one that predates the seam warns and
        keeps the inline path.
    gate_workers:
        Threads for the measurement phase when ``pipelined_gate`` is on. Bounded
        in practice by one candidate per artifact, so the default of 2 is enough
        for a single-artifact run with one measurement finishing as the next
        starts. This is a **third** pool -- ``n_workers`` rollouts,
        ``gate_workers`` measurements, each of which fans out over
        ``eval_concurrency`` tasks -- so the ceiling your provider sees is
        ``n_workers + gate_workers * eval_concurrency``.
    max_worker_errors:
        Consecutive failed rollouts before a worker that has *never* succeeded
        gives up. Workers that have succeeded at least once never retire; they
        back off and keep trying until the run's own budget ends it.
    patience:
        Stop after this many consecutive merge sweeps that fail to beat the best
        held-out reward seen so far. The async analogue of the synchronous knob:
        there are no round barriers here, so a *sweep* (one drain-and-merge by the
        merger) is the unit. ``None`` disables it.
    target_reward:
        Stop as soon as a sweep's held-out reward reaches this. Compared against
        the real reward, never against an acceptance probability.
    cheap_eval_tasks:
        As in :func:`evolve`: how many held-out tasks the cheap layer scores when
        ranking candidates. ``None`` is 8, or the whole held-out set when that is
        smaller.
    fusion_tournament:
        As in :func:`evolve`: rank the survivors against their fusion before
        putting one forward. ``None`` defers to ``agg_config``, which is off. The
        cost/benefit is identical on this path -- there is one merger thread here
        too, and it pays the ranking on the critical path of every commit.
    solved_threshold:
        As in :func:`evolve`: the reward at which a task counts as solved and no
        proposal is requested. Lower it for a graded scorer.
    shuffle, seed:
        As in :func:`evolve`: shuffle before the positional train/held-out split.
        Off by default.
    self_verify:
        As in :func:`evolve`. ``False`` skips the extra per-trajectory rollout,
        which is what ports that judge candidates only on held-out want.
    stall_patience:
        Merger sweeps that may pass with cards arriving and nothing committing
        before every worker is forced to resync, regardless of ``async_ratio``.
        Without it a lag budget larger than the staleness tolerance **livelocks**
        under the Guarded policy: workers propose against a snapshot too old for
        the policy to accept, every card is discarded, head never moves, and the
        lag budget therefore never triggers a refresh either.
    duration_estimator, straggler_factor:
        Pass a :class:`~agentdescent.scheduler.DurationEstimator` to fit
        ``seconds ~ intercept + slope * len(prompt)`` online and count rollouts
        that overran their own estimate by more than ``straggler_factor``
        (``result.stragglers``). This is the design's **L-traj** mechanism, which
        until now lived only in the reference runtime and so was unreachable from
        the API a real workload uses. Detection only: resuming a partial rollout
        would need it to expose its turns, and ``run(rendered, task) -> output``
        is opaque.
    shutdown_grace:
        Total seconds to wait for the worker and merger threads after the budget
        expires -- shared across all of them, not per thread. An in-flight
        rollout cannot be cancelled, so a slow backend can still overrun it; a
        warning says so and work already merged is kept.
    task_sampler:
        Which task a worker takes next from its shard.
    on_round:
        Called with each :class:`~agentdescent.evolution.RoundInfo` as a merger
        sweep completes -- progress for a long run. It runs on the merger thread
        and must be cheap and thread-safe; an exception is reported, not fatal.
    usage:
        Share one :class:`~agentdescent.agents.Usage` with your model adapters
        (``claude(usage=u)``, ``openai_compatible(usage=u)``) and the result's
        token counts become real. Without it the run still reports calls,
        seconds and failures -- ``run`` is ``(rendered, task) -> str``, so an
        opaque actor has no way to surface tokens, and inventing a number would
        be worse than reporting zero.
    verbose:
        Print one line per merger sweep.
    policies:
        Bundle of replaceable pieces (:class:`~agentdescent.policies.Policies`).
        Every field defaults to ``None`` meaning "current behaviour", so
        ``Policies()`` and passing nothing are the same run. The individual
        keyword arguments -- ``task_sampler``, ``staleness_policy``,
        ``aggregator_factory`` -- are shortcuts onto its fields and keep working;
        an explicit argument wins over a bundle default rather than being
        silently ignored. Fields whose implementations have not landed yet raise
        rather than being accepted and ignored: a caller who passes a custom
        acceptance rule and sees a finished run would reasonably conclude it ran.
        New capabilities go here rather than adding another parameter to a
        function that already has thirty-five.

    Returns
    -------
    EvolutionResult
        ``error`` is set only when the run **ended** because of a failure -- a
        transient error the workers retried past leaves it ``None``, so read
        ``stop_reason`` to tell ``"target_reward"`` from ``"max_seconds"`` /
        ``"max_iters"`` / ``"max_calls"`` / ``"patience"``.

        ``history`` holds one entry per **merger sweep** that had cards to merge,
        not per round: its length tracks how fast the workers produced and is not
        bounded by any argument (a 3s run with a fast reward produced 221).
        ``RoundInfo.round`` is the sweep index. Compare ``final_reward`` across the
        sync and async paths, not ``len(history)``.

        That "per sweep that had cards" rule is load-bearing, not descriptive: a
        sweep whose whole batch was discarded as stale **must** still be
        recorded, because the same bookkeeping feeds ``stall_patience`` -- the
        run's only livelock guard once head stops moving. Under
        ``pipelined_gate`` one more kind of entry appears: the sweep that
        *collects* a finished measurement records the merge it completes, so a
        single merge can contribute two entries (the sweep that ingested its
        cards, then the one that landed its decision).
    """
    _pol = _resolve_policies(policies, "async_evolve()",
                             supported=_ASYNC_WIRED_POLICIES,
                             task_sampler=task_sampler,
                             staleness=staleness_policy,
                             aggregator_factory=aggregator_factory)
    task_sampler, staleness_policy = _pol.task_sampler, _pol.staleness
    aggregator_factory = _pol.aggregator_factory
    # Passed through rather than consulted here: `_build_engine` turns a declared
    # policy into the population layer, and the aggregator asks it once per merge.
    selection = _pol.selection or SingleHead()
    eng = _build_engine(
        tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
        initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
        held_out_frac=held_out_frac, repo_path=repo_path, agg_config=agg_config,
        staleness_policy=staleness_policy, aggregator_factory=aggregator_factory,
        selection=selection,
        oracle_budget=oracle_budget, eval_concurrency=eval_concurrency,
        cheap_eval_tasks=cheap_eval_tasks, fusion_tournament=fusion_tournament,
        shuffle=shuffle, seed=seed,
        usage=usage, verifier=_pol.verifier, ledger_impl=_pol.ledger,
        policies_bundle=_pol)
    eng.meter.start()
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    policy = staleness_policy or get_policy("guarded")
    sampler = task_sampler or RoundRobin()
    # Staleness tolerance must come from the same config the aggregator uses --
    # hardcoding 5/1 here silently ignored agg_config.alpha_head/alpha_tail, so a
    # tightened tolerance was honoured by the aggregator but not by this gate.
    _cfg = agg_config or AggregatorConfig()
    from .governance import FAST_MAX
    alpha = _cfg.alpha_head if eng.blast_radius > FAST_MAX else _cfg.alpha_tail

    # data-parallel: shard the train tasks round-robin across workers.
    shards: List[List[Task]] = [[] for _ in range(n_workers)]
    for idx, key in enumerate(eng.train_ids):
        shards[idx % n_workers].append(eng.by_id[key])

    intake: List[EvidenceCard] = []
    intake_lock = threading.Lock()
    # The head every worker measures its drift against, published by the merger.
    #
    # Workers used to read it from the ledger on **every rollout**, and a ledger
    # read is a `git checkout` guarded by a process-wide file lock and an RLock
    # that every worker and the merger queue behind -- so the cost of asking "am I
    # far enough behind to resync?" grew with the concurrency it was there to
    # support. The merger is the only writer, so it can simply say. This is what
    # `AsyncAgentDescent._publish_head` has always done; the general engine reached
    # for git instead.
    #
    # A published head can lag by at most one sweep, which delays a refresh by one
    # rollout at worst. The refresh itself still takes a real `ledger.snapshot()`,
    # so nothing downstream reads a cached version.
    head_pub = [eng.ledger.head_version(Ledger.DEV).get(eng.artifact_id, 0)]
    head_lock = threading.Lock()

    def _publish_head(version: int) -> None:
        with head_lock:
            head_pub[0] = version

    def _published_head() -> int:
        with head_lock:
            return head_pub[0]
    stop = threading.Event()
    counter = [0]
    counter_lock = threading.Lock()
    history: List[RoundInfo] = []
    # [best held-out reward so far, sweeps since it last improved] -- a list so
    # the merger closure can mutate it without a `nonlocal` per field.
    # Shared with the barrier-free loop's sibling: one tracker, one epsilon.
    early = EarlyStop(target_reward=target_reward, patience=patience)
    errors: List[Optional[str]] = [None]      # first backend failure seen (diagnostic)
    # Most recent artifact read from the ledger, so a failing final read still
    # yields a result instead of an exception (same reasoning as the sync path).
    last_good: List[object] = [None]
    died = [False]                            # True only if the run ENDED on failure
    contract_error = FirstError()          # caller bug -> re-raise on this thread
    stop_reason = ["max_seconds"]             # overwritten by whichever bound fires
    # Group-relative reward, recorded on every card. Thread-safe by way of the
    # GIL on dict updates; the statistic is a running one, so a torn read costs
    # a slightly stale mean rather than a wrong decision -- and nothing reads it
    # by default anyway.
    advantage = GroupAdvantage()
    # Backpressure: bumped when the pipeline stalls (evidence keeps arriving and
    # nothing commits), which forces every worker to resync regardless of the ratio.
    epoch = [0]
    #: Bumped on every commit when ``resync_on_commit``. Separate from `epoch`
    #: because that one feeds `forced_refreshes`, which the docs define as quiet
    #: on a healthy run -- and a commit is the pipeline working, not a fault.
    commit_epoch = [0]
    forced_refreshes = [0]
    stragglers = [0]
    estimator = duration_estimator
    n_live = sum(1 for s in shards if s)      # workers that will actually start
    live = [n_live]                           # workers still running
    retired = [0]                             # workers that gave up (diagnostic)
    # Retirement and backpressure policy are shared with AsyncAgentDescent
    # (`pipeline.WorkerHealth` / `pipeline.StallGuard`), which is what keeps the
    # two barrier-free runtimes from drifting apart again.
    health = WorkerHealth(max_errors=max_worker_errors)
    stall = StallGuard(patience=stall_patience)
    max_merger_errors = max(3, max_worker_errors)   # sweeps it may fail in a row

    #: Set once the clock starts; the merger reads it to bound the budget-stop
    #: drain below. A list, because it is assigned after the closures are built.
    run_deadline = [float("inf")]
    worker_threads: List[threading.Thread] = []

    # -- the evaluation stage -------------------------------------------------
    #
    # `pipelined_gate` moves phase 2 of a merge (the four verifier calls, 94% of
    # the merger's gate time -- see `docs/efficiency.md`) onto its own threads.
    # Phase 1 (drain, staleness, fusion) and phase 3 (accept, audit, commit)
    # stay on the merger, which is still the only writer.
    #
    # `begin_step(skip_in_flight=True)` bounds it to one candidate per artifact,
    # so a candidate is always committed against the head it was measured on and
    # this needs no staleness rule of its own -- the cards that arrive meanwhile
    # simply accumulate in the aggregator's buffer, which is what a buffer is
    # for. A custom aggregator that predates the seam has no `begin_step`; it
    # keeps the inline path rather than being refused.
    #
    # Having the three methods is **not** enough, and the difference is not
    # academic: `PopulationAggregator` subclasses `Aggregator` -- so it inherits
    # all three -- and overrides `step()` to admit the pre-merge head into its
    # archive and consult its selection policy. Driving the phases directly there
    # would skip every line of that override and run a different algorithm while
    # reporting the requested one. So the test is that `step()` is still the base
    # implementation, which is the only case where the three phases are provably
    # what `step()` does.
    _has_phases = all(callable(getattr(eng.aggregator, m, None))
                      for m in ("begin_step", "measure", "finish_step"))
    _own_step = getattr(type(eng.aggregator), "step", None) is Aggregator.step
    _pipelined = pipelined_gate and _has_phases and _own_step
    if pipelined_gate and not _pipelined:
        why = ("it overrides step(), so the three phases are not what its step() "
               "does and running them directly would skip the override"
               if _has_phases else
               "it has no begin_step/measure/finish_step")
        warnings.warn(
            f"async_evolve(pipelined_gate=True) cannot pipeline "
            f"{type(eng.aggregator).__name__}: {why}. The gate stays on the "
            "merger thread as before. To pipeline a custom aggregator, express "
            "it as begin_step/measure/finish_step and leave step() inherited.",
            RuntimeWarning, stacklevel=2)
    gate_pool = (ThreadPoolExecutor(max(1, gate_workers),
                                    thread_name_prefix="agentdescent-gate")
                 if _pipelined else None)
    #: (future, items) for every batch out being measured.
    gate_inflight: List[Tuple[Any, List[Any]]] = []

    def _collect_gate(block: bool = False) -> List[Any]:
        """Finish every measured batch, and leave the rest in flight.

        `block=True` is the shutdown path: a candidate whose measurement has
        already been paid for must still get its decision, or the run throws
        away a full held-out sweep of real model calls it has already spent.
        """
        reports: List[Any] = []
        still: List[Tuple[Any, List[Any]]] = []
        for fut, items in gate_inflight:
            if not (block or fut.done()):
                still.append((fut, items))
                continue
            # Re-raised here on purpose: a measurement that failed is a backend
            # failure, and the merger's own error handling is what knows how to
            # tolerate one. Swallowing it would leave the artifact in flight
            # forever, and a merger that quietly stops merging looks idle.
            fut.result()
            reports.extend(eng.aggregator.finish_step(items))
        gate_inflight[:] = still
        return reports

    def _gated_step() -> List[Any]:
        """`aggregator.step()`, with phase 2 off this thread when asked."""
        if not _pipelined:
            return eng.aggregator.step()
        reports = _collect_gate()
        items = eng.aggregator.begin_step(skip_in_flight=True)
        if any(not hasattr(i, "committed_version") for i in items):
            gate_inflight.append(
                (gate_pool.submit(eng.aggregator.measure, items), items))
        elif items:                       # nothing to score: decide them now
            reports.extend(eng.aggregator.finish_step(items))
        return reports

    def _worker(wid: int, shard: List[Task]) -> None:
        snap = eng.ledger.snapshot(Ledger.DEV)
        base_v = snap.version.get(eng.artifact_id, 0)
        artifact = snap.get(eng.artifact_id)
        shard_ids = [t.id for t in shard]          # the sampler works on ids
        by_shard_id = {t.id: t for t in shard}
        i = 0
        local_epoch = epoch[0]
        local_commit = commit_epoch[0]
        consecutive = 0            # consecutive backend failures for this worker
        warned = False
        while not stop.is_set():
            # Lag budget bounds *un-merged* work too, not just committed drift.
            # Before head first advances (no commit yet) the version check can't
            # engage, so gate on pending intake: don't pile up more than
            # ``async_ratio`` candidates ahead of the merger (prevents the
            # cold-start flood where workers race while the merger is busy on the
            # first slow held-out eval).
            #
            # Timed, because this is where a slow gate becomes a slow *run*. The
            # merger's own occupancy says how busy it was; only this says whether
            # that cost anybody a rollout. `t_gate` stays None until the first
            # sleep so a worker that never waits records nothing at all -- an
            # unconditional timer would report the two `intake_lock` acquisitions
            # as starvation and put a floor under a counter whose whole value is
            # being zero when there is no problem.
            t_gate: Optional[float] = None
            while not stop.is_set():
                with intake_lock:
                    pending = len(intake)
                if pending <= async_ratio:
                    break
                if t_gate is None:
                    t_gate = time.time()
                time.sleep(0.05)
            if t_gate is not None:
                eng.meter.add("worker_starved_seconds", time.time() - t_gate)
            head_v = _published_head()
            # Refresh on the lag budget, or when the merger has asked everyone to
            # sync. `concepts.md` documents that backpressure guard as what keeps a
            # mismatched `async_ratio > alpha` from livelocking under Guarded --
            # workers keep proposing against a snapshot too old for the policy to
            # accept, every card is discarded, head never moves, so the lag budget
            # never triggers a refresh either. It existed only in the reference
            # runtime, which is not the one a real workload reaches.
            forced = epoch[0] != local_epoch
            committed_since = commit_epoch[0] != local_commit
            if head_v - base_v > async_ratio or forced or committed_since:
                snap = eng.ledger.snapshot(Ledger.DEV)
                base_v = snap.version.get(eng.artifact_id, 0)
                artifact = snap.get(eng.artifact_id)
                local_epoch = epoch[0]
                local_commit = commit_epoch[0]
                # Only the *forced* half is counted. Refreshing on one's own lag
                # budget is what an async worker does all run long -- counting it
                # made `forced_refreshes` non-zero on every healthy run, while the
                # field, `docs/async.md`, `docs/staleness.md` and `concepts.md` all
                # say a non-zero count means the lag budget and the staleness
                # tolerance disagree. It is a diagnostic, so it has to be quiet
                # when there is nothing to diagnose.
                if forced:
                    with counter_lock:
                        forced_refreshes[0] += 1
            task = by_shard_id[sampler.pick(shard_ids, i)]
            i += 1
            # Duration-aware straggler detection (design spec 5.1, L-traj). It
            # existed only in the reference runtime, which accepts nothing but the
            # synthetic router domain -- so the whole L-traj mechanism was
            # unreachable from the API every real workload uses. Cost is the task's
            # size, which is what correlates with agentic rollout time.
            cost = float(len(task.prompt))
            predicted = estimator.estimate(cost) if estimator is not None else 0.0
            t_start = time.time()
            try:
                output = eng.run(artifact.render(), task)
                score = _checked_reward(eng.reward(task, output), task)
                eng.meter.add("rollouts")
                eng.meter.add("rollout_seconds", time.time() - t_start)
                sampler.record(task.id, score)     # learn which tasks carry signal
                # Before the solved-task branch, for the same reason as on the
                # synchronous path: a group that only saw the failures has no
                # variance to standardise against.
                adv = advantage.observe(
                    advantage.key(base_v, str(task.meta.get("cluster", ""))), score)
                if score < solved_threshold:
                    proposal = _checked_proposal(
                        eng.propose(artifact.render(), task, output, score), task)
                    if proposal:
                        diff = eng.strategy.to_diff(artifact.state, proposal,
                                                    f"w{wid}", base_v, eng.artifact_id)
                        if diff is not None:
                            # Optional local self-verify: re-run the trajectory with the
                            # diff applied for a before/after signal. Faithful repos that
                            # only score the candidate on held-out (e.g. EvoSkill) pass
                            # self_verify=False to skip this extra rollout.
                            if self_verify:
                                after = _checked_reward(
                                    eng.reward(task, eng.run(artifact.apply(diff).render(), task)), task)
                                delta = after - score
                            else:
                                delta = 0.0
                            card = EvidenceCard(
                                diff=diff, base_version={eng.artifact_id: base_v},
                                touched=[eng.artifact_id], before_after_delta=delta,
                                trajectory_refs=[task],
                                # Same signal as the synchronous path, and the
                                # reason `GroupAdvantage` accumulates rather than
                                # batching at a barrier: there is no barrier here,
                                # so a batched version would silently record
                                # nothing on exactly the runtime the project is
                                # making claims about.
                                advantage=adv)
                            with intake_lock:
                                intake.append(card)
            except ContractError as e:
                # a caller-contract violation, not a flaky backend: stop at once.
                with counter_lock:
                    if errors[0] is None:
                        errors[0] = f"{type(e).__name__}: {e}"
                    died[0] = True
                contract_error.record(e)      # first one wins; this site overwrote
                stop.set()
                return
            except Exception as e:  # noqa: BLE001 - a backend failure (API error,
                # rate limit, credit exhaustion) must not silently kill the run.
                # Record it, tolerate transient ones, and retire only this worker
                # once they persist; the run ends when every worker has retired.
                with counter_lock:
                    if errors[0] is None:
                        errors[0] = f"{type(e).__name__}: {str(e)[:200]}"
                consecutive += 1
                if verbose:
                    print(f"worker {wid}  error {consecutive}/{max_worker_errors}: "
                          f"{type(e).__name__}: {str(e)[:100]}")
                # Two different situations wear the same exception, and they want
                # opposite responses. If NO worker has ever completed a rollout the
                # backend is almost certainly misconfigured -- wrong key, dead
                # endpoint -- so retire fast and let the run end loudly. Once any
                # worker has succeeded the backend demonstrably works, so this is a
                # transient: a rate limit, a blip. The test is deliberately global:
                # keyed on the worker's own history instead, an intermittent backend
                # retires whoever loses its first few rolls (at a 2-in-3 failure rate
                # that is ~30% of workers) even though nothing is wrong with them.
                # Retiring it there is actively wrong, because every worker shares
                # one backend, so shedding workers cannot relieve the throttling and
                # only guarantees the run dies. Measured: at a 1-in-3 call failure
                # rate (~56% per rollout, an ordinary 429 storm) the old blanket
                # rule retired all three workers in 22s with nothing learned.
                if health.should_retire(consecutive):
                    with counter_lock:
                        retired[0] += 1
                        live[0] -= 1
                        if live[0] <= 0:          # every worker retired -> end the run
                            died[0] = True
                            stop.set()
                    return
                if health.should_warn(consecutive) and not warned:
                    warned = True   # a backend that dies mid-run must not look idle
                    warnings.warn(
                        f"async_evolve: worker {wid} has failed {consecutive} "
                        f"rollouts in a row and is backing off, not retiring "
                        f"(it succeeded earlier, so this reads as transient). "
                        f"Last error: {type(e).__name__}: {str(e)[:120]}",
                        RuntimeWarning, stacklevel=2)
                # Exponential, capped: a throttled run waits the limit out instead
                # of hammering it, while staying short enough that a worker which
                # recovers has time left to produce something. (The paragraph that
                # used to sit here described the *merger's* backoff -- "unlike a
                # worker, the merger blocks the whole run" -- and had been copied
                # onto the worker path, where it contradicts both its own first
                # sentence and the code: the merger below waits up to 30s, this
                # waits up to 5.)
                stop.wait(min(0.25 * 2.0 ** consecutive, 5.0))     # backoff, then retry
                continue
            elapsed = time.time() - t_start
            if estimator is not None:
                estimator.observe(cost, elapsed)
                # A rollout that overran its own estimate is a straggler. Counted,
                # not resumed: `run(rendered, task) -> output` is opaque, so there
                # is no continuation state to check point (see concepts.md L-traj).
                if predicted > 0 and elapsed > straggler_factor * predicted:
                    with counter_lock:
                        stragglers[0] += 1
            consecutive, warned = 0, False                    # a clean rollout resets
            health.record_success()
            with counter_lock:
                counter[0] += 1
                if max_iters is not None and counter[0] >= max_iters:
                    stop_reason[0] = "max_iters"
                    stop.set()
                elif max_calls is not None and eng.meter.usage.calls >= max_calls:
                    # Read from the meter, not from a local tally: `calls` counts
                    # `propose` as well as `run`, and a rollout that solved its
                    # task never proposes -- so rollouts and calls are not a fixed
                    # ratio and the second budget cannot be derived from the first.
                    stop_reason[0] = "max_calls"
                    stop.set()

    def _drain_and_merge() -> None:
        with intake_lock:
            batch, intake[:] = intake[:], []
        # A sweep with no cards still has work when a measurement has *finished*:
        # its decision is the merger's, and nothing else will collect it. Without
        # this the pipeline would only ever finish a merge on the sweep that
        # happened to bring the next card in.
        #
        # `fut.done()`, not "anything in flight". Skipping the sleep while a
        # measurement is merely *running* turns the merger into a busy-wait, and
        # a busy-wait holds the GIL against the workers it was meant to free.
        # Measured that way first: 462 rollouts inline against 389 pipelined --
        # the change made the run slower, and the counters read as a fully
        # occupied merger because spinning is occupancy.
        if not batch and not any(f.done() for f, _ in gate_inflight):
            time.sleep(0.005)
            return
        # Timed from here, not from the top: a sweep that found nothing spent its
        # time in that poll sleep, and counting it would make merger occupancy a
        # measure of how idle the merger was.
        with eng.meter.timed("merge_seconds"):
            _merge_batch(batch)

    def _merge_batch(batch: List[EvidenceCard]) -> None:
        # Only when there is something to rebase *against*. A ledger read is a
        # `git checkout` behind a process-wide lock that every worker queues on,
        # and the collect-only sweeps above would otherwise take one every 5ms.
        if batch:
            snap = eng.ledger.snapshot(Ledger.DEV)
            head_vv, head_art = snap.version, snap.get(eng.artifact_id)

        def _discarded() -> None:
            """Record a card this gate is dropping, denominator included.

            The denominator is split between here and `Aggregator`, and it has to
            be: this gate sees every card but forwards only survivors, while the
            aggregator sees only survivors and counts them on the **same meter**.
            Counting the whole batch here as well counted every survivor twice --
            measured, 20 cards reported `stale_considered = 40`, so a true 50%
            stale rate came out as 33% and the "you are discarding most of your
            evidence" warning at the end of this function needed a *67%* true rate
            to fire at its documented 50% threshold.

            So each side counts what only it can see: the discards here, the
            survivors there. `D + (N - D) = N`.
            """
            eng.meter.add("stale_considered")
            eng.meter.add("stale_discarded")

        # staleness gate: hand the aggregator only rebased (η=0) cards.
        for card in batch:
            eta = vv_staleness(head_vv, card.base_version)
            action = policy.decide(eta, alpha, card.diff.contract_breaking)
            if action is StaleAction.ACCEPT:
                eng.aggregator.ingest(card if eta == 0 else card.rebased_onto(head_vv))
            elif action is StaleAction.REBASE:
                cand = head_art.apply(card.diff)         # cheap re-verify on current head
                # Gate work, and easy to forget it is: `evidence_eval` runs the
                # agent on the card's trajectories, so the "cheap" re-verify is
                # two rollout-priced measurements per rebased card.
                with eng.meter.timed("merge_gate_seconds"):
                    better = head_art.evidence_eval(card) <= cand.evidence_eval(card)
                if better:
                    eng.aggregator.ingest(card.rebased_onto(head_vv))
                else:
                    _discarded()
            else:
                _discarded()                             # DISCARD -> drop the card
        with eng.meter.timed("merge_gate_seconds"):
            reports = check_reports(_gated_step(), eng.aggregator)
        if not reports and not batch:
            # A pipelined poll whose candidate is still being measured has
            # nothing to report yet. Returning keeps the merger draining --
            # which is the point of the pipeline -- and keeps `history` free of
            # rounds no merge produced.
            #
            # `not batch` is load-bearing, and it was missing: a sweep that HAD
            # cards and reported nothing is not idle, it is the run's evidence
            # being rejected -- every card discarded at the staleness gate above,
            # or buffered below the batch trigger. Returning here skipped
            # `stall.note_sweep`, and that counter is the ONLY livelock guard:
            # with `async_ratio >> alpha`, head stops moving, so the lag budget
            # never forces a refresh, and `stall_patience` forcing one is what
            # breaks the cycle. Skipping the count disabled the guard in exactly
            # the situation it exists for -- CI caught it on all three Python
            # versions in `test_a_large_lag_budget_does_not_livelock`, the test
            # named for it, while faster local machines passed on the commits
            # that softened the race.
            return
        committed = sum(1 for x in reports if x.committed_version is not None)
        after = eng.ledger.snapshot(Ledger.DEV)
        dev = after.get(eng.artifact_id)
        _publish_head(after.version.get(eng.artifact_id, 0))
        last_good[0] = dev
        # Must be the real held-out reward: MergeReport.prob_improve is P(Δ>0)
        # from the Beta posterior, a *probability*, and reporting it here would
        # both corrupt `history` and make `target_reward` fire spuriously. This
        # is not a redundant eval -- `_Runtime.eval_one` memoises on
        # (artifact signature, task id), so re-scoring an unchanged head is free.
        with eng.meter.timed("merge_gate_seconds"):
            r = dev.score(eng.held_out)
        # Same three steps as the barrier loop; the round index is a merger sweep
        # here and a barrier there, which is the only part that differs.
        _info, early_stop = eng.record_round(
            index=len(history), reward=r, n_items=len(dev.state),
            reports=reports, history=history, early=early, on_round=on_round)
        # A stalled pipeline: cards keep arriving and none of them commits. Under
        # Guarded with async_ratio > alpha that is a livelock, not slow progress.
        # Counted per sweep **that had cards or reports** -- a poll with neither
        # returned above. A sweep whose whole batch was discarded at the
        # staleness gate MUST land here with committed=0: those discards are the
        # livelock's signature, and this counter is the only thing that breaks it.
        stall.note_sweep(committed)
        # A commit changes what the artifact *is*, so a worker about to start a
        # rollout on the pre-merge snapshot is working from a version that no
        # longer exists. The lag budget does not catch that on its own -- it
        # fires at ``head_v - base_v > async_ratio``, so with the default 3 the
        # first three commits leave the whole fleet on its start-of-run
        # snapshot, which on a run that starts from an empty artifact means
        # every worker keeps re-deriving what head already has.
        #
        # This does not make cards fresh: the snapshot is taken before `run` and
        # the card is pushed after it, so a commit during a long rollout still
        # arrives stale, which is where staleness comes from on any workload
        # whose rollouts outlast a sweep -- the module's premise survives. It
        # stays switchable because on *short* rollouts the two windows collapse
        # into one and η goes to 0, so anything measuring the lag budget in
        # isolation has to turn it off.
        if resync_on_commit and committed:
            commit_epoch[0] += 1
        if stall.should_force_refresh():
            epoch[0] += 1                      # every worker resyncs on its next loop
            stall.force()
        if verbose:
            print(f"sweep {len(history):>3}  reward={r:.3f}  merged={len(batch)}  "
                  f"+{committed}  pending={len(intake)}")
        if early_stop is not None:
            stop_reason[0] = early_stop
            stop.set()

    def _finish_pipeline() -> None:
        """Decide every candidate still out being measured, then let it go.

        On the merger thread, like every other decision. Failures are reported
        the way the rest of the merger reports them rather than raised: this runs
        after the budget has expired, and a run that produced results must not
        lose them to a measurement that failed on the way out.
        """
        if not gate_inflight:
            return
        try:
            reports = _collect_gate(block=True)
        except Exception as e:  # noqa: BLE001 - the run is over; keep what landed
            with counter_lock:
                if errors[0] is None:
                    errors[0] = f"{type(e).__name__}: {str(e)[:200]}"
            return
        if not reports:
            return
        try:
            after = eng.ledger.snapshot(Ledger.DEV)
            dev = after.get(eng.artifact_id)
            if dev is None:
                return
            _publish_head(after.version.get(eng.artifact_id, 0))
            last_good[0] = dev
            # Recorded, not just committed. A merge that lands here is a merge
            # like any other, and leaving it out of `history` would put a commit
            # in the ledger that no round accounts for -- visible only as a
            # `final_reward` that no entry in the history explains. The score is
            # memoised per (artifact, task), so this costs nothing when the
            # candidate was the one just measured.
            eng.record_round(index=len(history), reward=dev.score(eng.held_out),
                             n_items=len(dev.state), reports=reports,
                             history=history, early=early, on_round=on_round)
        except Exception as e:  # noqa: BLE001 - the commits are already in the ledger
            with counter_lock:
                if errors[0] is None:
                    errors[0] = f"{type(e).__name__}: {str(e)[:200]}"

    def _merger() -> None:
        # The merger is the only writer; if it dies the workers would keep filling a
        # buffer nobody drains and the run would spin to max_seconds with no reason
        # given. But it also *calls the backend* every sweep (it scores the held-out
        # set), so a single blanket try/except around the whole loop made it a single
        # point of failure that one transient could take out permanently -- measured:
        # against a backend refusing 1 call in 3, the run ended with 0 sweeps while
        # the workers were still healthy. It gets the same tolerance they have.
        consecutive = 0
        while True:
            try:
                if stop.is_set():
                    # A stop on a WORK budget (max_iters / max_calls) owes the run
                    # a full drain. The budget counts a rollout when it completes,
                    # so at the moment it trips, up to n_workers-1 rollouts are
                    # still in flight -- legitimately started, their model calls
                    # already made and billed. Returning after one drain abandoned
                    # their evidence, and not uniformly: a failing rollout runs
                    # propose + self-verify after solve, three sequential calls
                    # against a successful rollout's one, so the in-flight set is
                    # *enriched with exactly the rollouts that produce cards*.
                    # Measured on an 8-worker run: 8 cards produced, 7 abandoned,
                    # the pool never grew past the seed, and the arm looked fast
                    # because it had silently skipped its merging.
                    #
                    # A stop on the TIME budget keeps the old behaviour: the user
                    # bounded wall-clock, and waiting would overshoot the one
                    # thing they fixed. Bounded either way by `max_seconds`, the
                    # run's own outer limit, so a wedged rollout cannot turn the
                    # drain into a hang.
                    if stop_reason[0] in ("max_iters", "max_calls"):
                        while (any(t.is_alive() for t in worker_threads)
                               and time.time() < run_deadline[0]):
                            _drain_and_merge()
                    _drain_and_merge()       # final drain after stop
                    # A measurement already paid for is a full held-out sweep of
                    # real model calls; abandoning it at the buzzer throws away
                    # the most expensive thing the run bought, and the artifact
                    # would stay in flight with its decision never made. Blocking
                    # here is bounded by the measurement itself, which was
                    # already running before the budget expired.
                    _finish_pipeline()
                    return
                _drain_and_merge()
                consecutive = 0
            except ContractError as e:
                # A caller bug (a broken aggregator, a bad reward) must propagate,
                # not be absorbed and reported as if the provider had failed.
                contract_error.record(e)
                stop.set()
                return
            except Exception as e:  # noqa: BLE001 - surface, don't hang
                consecutive += 1
                with counter_lock:
                    if errors[0] is None:
                        errors[0] = f"{type(e).__name__}: {str(e)[:200]}"
                if verbose:
                    print(f"merger error {consecutive}/{max_merger_errors}: "
                          f"{type(e).__name__}: {str(e)[:120]}")
                # Deliberately no fail-fast of its own: killing the run here would
                # repeat the workers' old mistake, and the two cases that *should*
                # end a run are already covered. A truly dead backend retires every
                # worker, which ends it; a broken aggregator or reward raises
                # ContractError, handled above. What is left is a transient, and the
                # run's own budget (max_seconds / target_reward / patience) bounds
                # the wait. Say it out loud once so a merger that never recovers is
                # not mistaken for an idle one.
                if consecutive == max_merger_errors:
                    warnings.warn(
                        f"async_evolve: the merger has failed {consecutive} sweeps "
                        f"in a row and is retrying; nothing will merge until it "
                        f"recovers. Last error: {type(e).__name__}: {str(e)[:120]}",
                        RuntimeWarning, stacklevel=2)
                stop.wait(min(2.0 ** consecutive, 30.0))

    workers = [threading.Thread(target=_worker, args=(w, s), daemon=True)
               for w, s in enumerate(shards) if s]
    worker_threads.extend(workers)
    merger = threading.Thread(target=_merger, daemon=True)
    t0 = time.time()
    run_deadline[0] = t0 + max_seconds
    for t in workers:
        t.start()
    merger.start()
    while time.time() - t0 < max_seconds and not stop.is_set():
        time.sleep(0.02)
    stop.set()
    # Bounded shutdown. Joining each worker for 2s and the merger for 10s made the
    # overshoot scale with n_workers -- a 1s budget could return 16s later. The
    # threads are daemons and the merger does a final drain, so share one short
    # deadline across all the joins instead of paying it per thread.
    # Same rule as the merger's drain: a work-budget stop waits for the paid-for
    # rollouts (bounded by max_seconds); a time-budget stop keeps the short grace.
    if stop_reason[0] in ("max_iters", "max_calls"):
        shutdown_deadline = max(time.time() + shutdown_grace, run_deadline[0])
    else:
        shutdown_deadline = time.time() + shutdown_grace
    for t in workers:
        t.join(timeout=max(0.0, shutdown_deadline - time.time()))
    merger.join(timeout=max(0.0, shutdown_deadline - time.time()))
    if any(t.is_alive() for t in workers) or merger.is_alive():
        # A rollout in flight cannot be cancelled; say so rather than hang.
        warnings.warn(
            f"async_evolve: {sum(t.is_alive() for t in workers)} worker(s) still "
            f"running after the {shutdown_grace}s shutdown grace; their in-flight "
            "rollouts are abandoned (results already merged are kept)",
            RuntimeWarning, stacklevel=2)

    contract_error.raise_if_set()
    if not died[0]:
        # Same reason as the synchronous path: confirmation takes
        # `promote_after_k` sweeps and `target_reward` stops the run on the very
        # commit that reaches it, so publish the head this run produced rather
        # than leaving `stable` on the seed artifact.
        _publish_stable(eng.aggregator)
    try:
        final = eng.ledger.snapshot(Ledger.DEV).get(eng.artifact_id)
    except LedgerFailure as e:
        # The ledger is infrastructure: neither a caller bug nor a backend blip.
        # Its failure must still leave a result behind (see the sync path), so fall
        # back to whatever the merger last read.
        final = last_good[0]
        errors[0] = (f"the final ledger read failed, so the returned artifact is "
                     f"the last one successfully read: {type(e).__name__}: "
                     f"{str(e)[:160]}")
        died[0] = True
    if final is None:                  # nothing was ever read: hand back the seed
        from .evolution import EvolvingArtifact
        final = EvolvingArtifact(eng.artifact_id, dict(eng.strategy.initial()),
                                 blast_radius=eng.blast_radius,
                                 runtime=eng.runtime, strategy=eng.strategy)
    # Scoring the final artifact runs the agent too, so a dead backend must not
    # raise out of the driver -- that would discard the work already committed.
    # Retry it: scoring is memoised per (artifact, task), so a retry re-runs only
    # the tasks that actually failed. Without this a single transient on the last
    # measurement of an otherwise healthy run got reported as `error`, which is
    # documented to mean the run *ended* on a failure -- it did not.
    final_reward, score_error = None, None
    for attempt in range(3):
        try:
            final_reward = final.score(eng.held_out)
            break
        except Exception as e:  # noqa: BLE001 - report, keep the partial result
            score_error = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    if final_reward is None:
        # Say which of the two happened. The workers may have run fine all along
        # and only this last measurement failed, and the fixes differ.
        errors[0] = (f"final held-out scoring failed after 3 attempts, so "
                     f"final_reward falls back to the last measured round: "
                     f"{score_error}")
        died[0] = True
        final_reward = history[-1].held_out_reward if history else 0.0

    # `error` means "the run ended because of a failure", not "a failure happened":
    # transient errors that the workers retried past leave a clean result.
    run_error = errors[0] if died[0] else None
    if run_error:
        if verbose:
            print(f"async run ended with a backend failure: {run_error[:140]}")
        warnings.warn(f"async_evolve() ended with a backend failure: {run_error}",
                      RuntimeWarning, stacklevel=2)
    # A run that discarded most of its evidence is misconfigured, and every
    # number it reports is a number about the fraction that survived. `stale_rate`
    # makes that visible to anyone who looks; this says it to anyone who does not.
    #
    # Not a reason to lower the default: `async_ratio` is a lag budget in
    # *versions*, and how much wall-clock a version represents depends entirely on
    # how long a rollout takes. Three is sensible when a rollout is a model call
    # taking seconds; on a workload where rollouts are near-instant, a worker
    # drifts three versions behind almost immediately and stays there.
    _considered = eng.meter.snapshot().stale_considered
    _discarded = eng.meter.snapshot().stale_discarded
    if _considered >= 20 and _discarded / _considered > 0.5:
        warnings.warn(
            f"async_evolve discarded {_discarded}/{_considered} "
            f"({_discarded / _considered:.0%}) of its evidence as stale. "
            f"async_ratio={async_ratio} is a lag budget in artifact versions, so "
            "it is too high whenever a worker finishes several rollouts in the "
            "time the merger takes one sweep. Lower it (0 resyncs every rollout) "
            "or use a staleness policy that rebases rather than discards.",
            RuntimeWarning, stacklevel=2)

    result = EvolutionResult(state=dict(final.state), rendered=final.render(),
                             final_reward=final_reward, history=history,
                             ledger_log=_safe_log(eng.ledger),
                             error=run_error, retired_workers=retired[0],
                             stop_reason="error" if run_error else stop_reason[0],
                             forced_refreshes=forced_refreshes[0],
                             stragglers=stragglers[0],
                             fusion_trials=_fusion_trials(eng.aggregator),
                             **_cost_fields(eng.meter))
    if gate_pool is not None:
        # `wait=False`: the merger has already blocked on every measurement it
        # meant to keep (`_finish_pipeline`), so anything still running here is
        # work the run has decided to abandon. Waiting for it would put an
        # unbounded backend call after the budget the caller fixed.
        gate_pool.shutdown(wait=False)
    eng.cleanup()          # do not hold a scratch git repo for the whole process
    return result
