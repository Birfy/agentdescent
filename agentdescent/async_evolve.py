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
from typing import Callable, Dict, List, Optional

from .evolution import (
    _safe_log,
    Agent, EvolutionResult, Propose, Reward, RoundInfo, Run, Strategy, Task, _tally,
    SOLVED, _build_engine, _checked_proposal, _checked_reward,
)
from .aggregator import AggregatorConfig, check_reports
from .evolvable import ContractError, EvidenceCard, vv_staleness
from .ledger import Ledger, LedgerFailure
from .sampling import RoundRobin, TaskSampler
from .scheduler import DurationEstimator
from .pipeline import WorkerHealth
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
    max_seconds: float = 20.0,
    max_iters: Optional[int] = None,
    target_reward: Optional[float] = None,
    patience: Optional[int] = None,
    max_worker_errors: int = 3,
    eval_concurrency: int = 8,
    held_out_frac: float = 0.4,
    repo_path: Optional[str] = None,
    agg_config=None,
    staleness_policy: Optional[StalenessPolicy] = None,
    aggregator_factory=None,
    oracle_budget: int = 200,
    cheap_eval_tasks: Optional[int] = None,
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
    eval_concurrency:
        How many held-out tasks the merger scores at once. ``1`` restores the old
        sequential behaviour.
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
        ranking candidates. ``None`` scores them all.
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
    verbose:
        Print one line per merger sweep.

    Returns
    -------
    EvolutionResult
        ``error`` is set only when the run **ended** because of a failure -- a
        transient error the workers retried past leaves it ``None``, so read
        ``stop_reason`` to tell ``"target_reward"`` from ``"max_seconds"`` /
        ``"max_iters"`` / ``"patience"``.

        ``history`` holds one entry per **merger sweep** that had cards to merge,
        not per round: its length tracks how fast the workers produced and is not
        bounded by any argument (a 3s run with a fast reward produced 221).
        ``RoundInfo.round`` is the sweep index. Compare ``final_reward`` across the
        sync and async paths, not ``len(history)``.
    """
    eng = _build_engine(
        tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
        initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
        held_out_frac=held_out_frac, repo_path=repo_path, agg_config=agg_config,
        staleness_policy=staleness_policy, aggregator_factory=aggregator_factory,
        oracle_budget=oracle_budget, eval_concurrency=eval_concurrency,
        cheap_eval_tasks=cheap_eval_tasks, shuffle=shuffle, seed=seed)
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
    stop = threading.Event()
    counter = [0]
    counter_lock = threading.Lock()
    history: List[RoundInfo] = []
    # [best held-out reward so far, sweeps since it last improved] -- a list so
    # the merger closure can mutate it without a `nonlocal` per field.
    best: List[float] = [float('-inf'), 0]
    errors: List[Optional[str]] = [None]      # first backend failure seen (diagnostic)
    # Most recent artifact read from the ledger, so a failing final read still
    # yields a result instead of an exception (same reasoning as the sync path).
    last_good: List[object] = [None]
    died = [False]                            # True only if the run ENDED on failure
    contract_error: List[Optional[BaseException]] = [None]   # caller bug -> re-raise
    stop_reason = ["max_seconds"]             # overwritten by whichever bound fires
    # Backpressure: bumped when the pipeline stalls (evidence keeps arriving and
    # nothing commits), which forces every worker to resync regardless of the ratio.
    epoch = [0]
    forced_refreshes = [0]
    no_commit_sweeps = [0]
    stragglers = [0]
    estimator = duration_estimator
    n_live = sum(1 for s in shards if s)      # workers that will actually start
    live = [n_live]                           # workers still running
    retired = [0]                             # workers that gave up (diagnostic)
    # Retirement policy is shared with AsyncAgentDescent (`pipeline.WorkerHealth`),
    # which is what keeps the two barrier-free runtimes from drifting apart again.
    health = WorkerHealth(max_errors=max_worker_errors)
    max_merger_errors = max(3, max_worker_errors)   # sweeps it may fail in a row

    def _worker(wid: int, shard: List[Task]) -> None:
        snap = eng.ledger.snapshot(Ledger.DEV)
        base_v = snap.version.get(eng.artifact_id, 0)
        artifact = snap.get(eng.artifact_id)
        shard_ids = [t.id for t in shard]          # the sampler works on ids
        by_shard_id = {t.id: t for t in shard}
        i = 0
        local_epoch = epoch[0]
        consecutive = 0            # consecutive backend failures for this worker
        warned = False
        while not stop.is_set():
            # Lag budget bounds *un-merged* work too, not just committed drift.
            # Before head first advances (no commit yet) the version check can't
            # engage, so gate on pending intake: don't pile up more than
            # ``async_ratio`` candidates ahead of the merger (prevents the
            # cold-start flood where workers race while the merger is busy on the
            # first slow held-out eval).
            while not stop.is_set():
                with intake_lock:
                    pending = len(intake)
                if pending <= async_ratio:
                    break
                time.sleep(0.05)
            head_v = eng.ledger.head_version(Ledger.DEV).get(eng.artifact_id, 0)
            # Refresh on the lag budget, or when the merger has asked everyone to
            # sync. `concepts.md` documents that backpressure guard as what keeps a
            # mismatched `async_ratio > alpha` from livelocking under Guarded --
            # workers keep proposing against a snapshot too old for the policy to
            # accept, every card is discarded, head never moves, so the lag budget
            # never triggers a refresh either. It existed only in the reference
            # runtime, which is not the one a real workload reaches.
            if head_v - base_v > async_ratio or epoch[0] != local_epoch:
                snap = eng.ledger.snapshot(Ledger.DEV)
                base_v = snap.version.get(eng.artifact_id, 0)
                artifact = snap.get(eng.artifact_id)
                local_epoch = epoch[0]
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
                sampler.record(task.id, score)     # learn which tasks carry signal
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
                                trajectory_refs=[task])
                            with intake_lock:
                                intake.append(card)
            except ContractError as e:
                # a caller-contract violation, not a flaky backend: stop at once.
                with counter_lock:
                    if errors[0] is None:
                        errors[0] = f"{type(e).__name__}: {e}"
                    died[0] = True
                    contract_error[0] = e
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
                # Cap the backoff well above the retirement threshold so a throttled
                # run waits the limit out instead of hammering it.
                # A short backoff on purpose: unlike a worker, the merger blocks
                # the whole run while it waits, and its work is memoised, so a
                # retry is cheap. A worker-sized backoff spent an entire short run
                # asleep and produced no sweeps at all.
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

    def _drain_and_merge() -> None:
        with intake_lock:
            batch, intake[:] = intake[:], []
        if not batch:
            time.sleep(0.005)
            return
        head_vv = eng.ledger.head_version(Ledger.DEV)
        head_art = eng.ledger.snapshot(Ledger.DEV).get(eng.artifact_id)
        # staleness gate: hand the aggregator only rebased (η=0) cards.
        for card in batch:
            eta = vv_staleness(head_vv, card.base_version)
            action = policy.decide(eta, alpha, card.diff.contract_breaking)
            if action is StaleAction.ACCEPT:
                eng.aggregator.ingest(card if eta == 0 else card.rebased_onto(head_vv))
            elif action is StaleAction.REBASE:
                cand = head_art.apply(card.diff)         # cheap re-verify on current head
                if head_art.evidence_eval(card) <= cand.evidence_eval(card):
                    eng.aggregator.ingest(card.rebased_onto(head_vv))
            # DISCARD -> drop the card
        reports = check_reports(eng.aggregator.step(), eng.aggregator)
        committed = sum(1 for x in reports if x.committed_version is not None)
        dev = eng.ledger.snapshot(Ledger.DEV).get(eng.artifact_id)
        last_good[0] = dev
        # Must be the real held-out reward: MergeReport.prob_improve is P(Δ>0)
        # from the Beta posterior, a *probability*, and reporting it here would
        # both corrupt `history` and make `target_reward` fire spuriously. This
        # is not a redundant eval -- `_Runtime.eval_one` memoises on
        # (artifact signature, task id), so re-scoring an unchanged head is free.
        r = dev.score(eng.held_out)
        if r > best[0] + 1e-12:
            best[0], best[1] = r, 0
        else:
            best[1] += 1
        history.append(RoundInfo(len(history), r, len(dev.state), committed,
                                 len(reports) - committed, _tally(reports)))
        # A stalled pipeline: cards keep arriving and none of them commits. Under
        # Guarded with async_ratio > alpha that is a livelock, not slow progress.
        if committed:
            no_commit_sweeps[0] = 0
        elif batch:
            no_commit_sweeps[0] += 1
            if no_commit_sweeps[0] >= stall_patience:
                epoch[0] += 1                  # every worker resyncs on its next loop
                no_commit_sweeps[0] = 0
        if verbose:
            print(f"sweep {len(history):>3}  reward={r:.3f}  merged={len(batch)}  "
                  f"+{committed}  pending={len(intake)}")
        if on_round is not None:
            try:                       # a reporting callback must not kill the merger
                on_round(history[-1])
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"on_round callback raised: {type(e).__name__}: {e}",
                              RuntimeWarning, stacklevel=2)
        if target_reward is not None and r >= target_reward:
            stop_reason[0] = "target_reward"
            stop.set()
        elif patience is not None and best[1] >= patience:
            stop_reason[0] = "patience"
            stop.set()

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
                    _drain_and_merge()       # final drain after stop
                    return
                _drain_and_merge()
                consecutive = 0
            except ContractError as e:
                # A caller bug (a broken aggregator, a bad reward) must propagate,
                # not be absorbed and reported as if the provider had failed.
                with counter_lock:
                    if contract_error[0] is None:
                        contract_error[0] = e
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
    merger = threading.Thread(target=_merger, daemon=True)
    t0 = time.time()
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

    if contract_error[0] is not None:
        raise contract_error[0]
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
    result = EvolutionResult(state=dict(final.state), rendered=final.render(),
                             final_reward=final_reward, history=history,
                             ledger_log=_safe_log(eng.ledger),
                             error=run_error, retired_workers=retired[0],
                             stop_reason="error" if run_error else stop_reason[0],
                             forced_refreshes=forced_refreshes[0],
                             stragglers=stragglers[0])
    eng.cleanup()          # do not hold a scratch git repo for the whole process
    return result
