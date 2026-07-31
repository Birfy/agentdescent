"""Asynchronous stage-orchestration runtime (design doc, section 3.1;
FlashEvolve 2605.08520; ROLL Flash 2510.11345).

The synchronous :class:`~agentdescent.orchestrator.AgentDescent` runs a round barrier:
every worker steps, then a single ``aggregator.step()`` fires, then the next
round begins.  That is "synchronous DP".  This module removes the barrier.

**Stage orchestration (FlashEvolve).**  The stages -- *rollout+propose* (workers)
and *aggregate+commit* (aggregator) -- run as independent threads connected by
the thread-safe :class:`~agentdescent.aggregator.EvidenceBuffer`.  A worker keeps
producing evidence while the aggregator is still merging the previous batch, so
the pipeline overlaps instead of stalling on a barrier.

**Asynchronous ratio (ROLL Flash).**  ``async_ratio`` is a *global lag budget*:
a worker refreshes its ledger snapshot only once head has drifted more than
``async_ratio`` versions ahead of it.  Small ratio -> near-synchronous, few stale
diffs; large ratio -> highly asynchronous, many stale diffs that the active
:class:`~agentdescent.staleness.StalenessPolicy` (Full / Guarded / Reflective) must
rebase or discard.  This is the knob that trades throughput against staleness.

The GIL means Python threads do not give true CPU parallelism, but the *pipeline
overlap* and the concurrency-control machinery (CAS, buffer locks, per-diff
staleness) are real -- and the same code shape drives a genuinely parallel
process/host pool.
"""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .aggregator import Aggregator, AggregatorConfig
from .domains.router import (
    RouterSkill,
    TaskUniverse,
    deserialize_router,
    router_eval,
    serialize_router,
)
from .ledger import Ledger
from .scheduler import (
    AuditScheduler,
    DurationEstimator,
    ResumeItem,
    ResumeQueue,
    TaskCluster,
    TaskScheduler,
)
from .staleness import StalenessPolicy, get_policy
from .verifier import ThreeLayerVerifier, VerifierBudget
from .worker import Worker


@dataclass
class AsyncConfig:
    n_workers: int = 6
    async_ratio: int = 3          # ROLL Flash global lag budget (max head drift)
    noise: float = 0.15
    target_accuracy: float = 0.98
    max_seconds: float = 20.0     # wall-clock safety bound
    aggregator_interval: float = 0.002  # sleep between aggregator sweeps
    worker_pause: float = 0.001   # sleep between worker rollouts
    oracle_budget: int = 400
    stall_patience: int = 150     # no-commit sweeps before backpressure sync
    # duration-aware straggler control: a rollout that runs longer than
    # duration_timeout_factor x its *estimated* cost is checkpointed to the
    # ResumeQueue (partial rollout) instead of blocking a worker.
    duration_timeout_factor: float = 3.0
    seed: int = 0


@dataclass
class AsyncStats:
    rollouts: int = 0
    proposals: int = 0
    sweeps: int = 0
    commits: int = 0
    fused: int = 0
    discarded_stale: int = 0
    conflicts_dropped: int = 0
    forced_refreshes: int = 0
    stragglers_checkpointed: int = 0
    #: Workers that gave up after repeated failures *before any worker had ever
    #: succeeded*. A run can finish cleanly at a fraction of its requested
    #: concurrency, so `error` stays None while throughput quietly drops.
    retired_workers: int = 0
    oracle_used: int = 0
    final_dev_accuracy: float = 0.0
    final_stable_accuracy: float = 0.0
    wallclock: float = 0.0
    #: ``None`` on a clean run; otherwise the backend failure that ended it. A run
    #: whose workers all died used to return normal-looking zeros.
    error: Optional[str] = None
    timeline: List[Tuple[int, float]] = field(default_factory=list)


class AsyncAgentDescent:

    _MAX_WORKER_ERRORS = 3
    _MAX_MERGER_ERRORS = 3
    """Barrier-free, thread-per-worker parallel self-evolution runtime."""

    def __init__(
        self,
        repo_path: str,
        universe: TaskUniverse,
        config: Optional[AsyncConfig] = None,
        agg_config: Optional[AggregatorConfig] = None,
        staleness_policy: Optional[StalenessPolicy] = None,
        estimator: Optional[DurationEstimator] = None,
        skill_id: str = "mol-router",
    ) -> None:
        self.cfg = config or AsyncConfig()
        self.universe = universe
        self.skill_id = skill_id
        # duration-aware scheduling: predict rollout cost from task size and
        # checkpoint stragglers to the resume queue (design doc, section 5.1).
        self.estimator = estimator
        self.resume_queue = ResumeQueue()
        self.train, self.held_out = universe.split()

        self.ledger = Ledger(repo_path, serialize_router, deserialize_router)
        self.ledger.register(RouterSkill(skill_id, table={}), branch=Ledger.DEV)

        self.verifier = ThreeLayerVerifier(
            eval_fn=router_eval,
            held_out=self.held_out,
            budget=VerifierBudget(oracle_calls_remaining=self.cfg.oracle_budget),
            seed=self.cfg.seed,
        )
        self.audit = AuditScheduler()
        self.aggregator = Aggregator(
            self.ledger, self.verifier, self.audit,
            agg_config or AggregatorConfig(),
            staleness_policy=staleness_policy or get_policy("guarded"),
        )

        clusters = universe.clusters(n_clusters=max(2, self.cfg.n_workers))
        self.scheduler = TaskScheduler(
            [TaskCluster(id=f"c{i}", tasks=c) for i, c in enumerate(clusters)]
        )
        self.workers = [
            Worker(worker_id=f"w{i}", gold=universe.gold,
                   noise=self.cfg.noise if i % 3 == 0 else 0.0,
                   seed=self.cfg.seed + i)
            for i in range(self.cfg.n_workers)
        ]

        self.stats = AsyncStats()
        self._stop = threading.Event()
        # published head version (updated by the aggregator thread on commit) so
        # workers can measure drift without hitting git on every rollout.
        self._head_lock = threading.Lock()
        self._head: Dict[str, int] = dict(self.ledger.head_version(Ledger.DEV))
        # backpressure: bumped when the pipeline stalls (buffer keeps filling but
        # nothing commits), forcing every worker to sync regardless of the ratio.
        self._refresh_epoch = 0
        self._stats_lock = threading.Lock()
        # worker resilience: retire a worker only while NO worker has ever
        # succeeded (see _worker_loop); the run ends once every worker has retired.
        self._live_workers = len(self.workers)
        self._any_success = False

    # -- shared head bookkeeping (ROLL Flash async ratio) --------------------

    def _publish_head(self) -> None:
        hv = self.ledger.head_version(Ledger.DEV)
        with self._head_lock:
            self._head = dict(hv)

    def _head_version(self, artifact_id: str) -> int:
        with self._head_lock:
            return self._head.get(artifact_id, 0)

    def _epoch(self) -> int:
        with self._head_lock:
            return self._refresh_epoch

    def _bump_epoch(self) -> None:
        with self._head_lock:
            self._refresh_epoch += 1

    # -- worker thread -------------------------------------------------------

    def _worker_loop(self, worker: Worker) -> None:
        # each worker caches a snapshot and refreshes only when its drift from
        # head exceeds the async ratio (bounded staleness).
        snap = self.ledger.snapshot(Ledger.DEV)
        local = snap.get(self.skill_id)
        local_v = snap.version.get(self.skill_id, 0)
        local_epoch = self._epoch()
        consecutive = 0            # consecutive backend failures for this worker

        while not self._stop.is_set():
            drift = self._head_version(self.skill_id) - local_v
            epoch = self._epoch()
            # refresh when the ratio budget is exceeded, or when a pipeline stall
            # forced a global sync (backpressure).
            if drift > self.cfg.async_ratio or epoch != local_epoch:
                snap = self.ledger.snapshot(Ledger.DEV)
                local = snap.get(self.skill_id)
                local_v = snap.version.get(self.skill_id, 0)
                local_epoch = epoch
                with self._stats_lock:
                    self.stats.forced_refreshes += 1

            cluster = self.scheduler.lease_round_robin()
            # estimate this rollout's cost from task size, time it, and calibrate.
            cost = float(sum(len(t.text) for t in cluster.tasks))
            predicted = self.estimator.estimate(cost) if self.estimator else 0.0
            t0 = time.time()
            try:
                card = worker.run(local, {self.skill_id: local_v}, cluster.tasks)
            except Exception as e:  # noqa: BLE001 - a backend failure must not kill
                # the thread silently: an unhandled exception here used to print a
                # traceback and leave the run spinning out its whole budget with no
                # producers, returning zeros that looked like a normal result.
                consecutive += 1
                with self._stats_lock:
                    if self.stats.error is None:
                        self.stats.error = f"{type(e).__name__}: {str(e)[:200]}"
                # Two situations wear the same exception and want opposite
                # responses, and this used to treat them alike -- the blanket rule
                # `async_evolve` removed after measuring it. Keyed on a worker's own
                # history, an intermittent backend retires whoever loses its first
                # few rolls even though nothing is wrong with it; and since every
                # worker shares one backend, shedding workers cannot relieve the
                # throttling and only guarantees the run dies. Measured there at a
                # 1-in-3 call failure rate: all three workers retired in 22s with
                # nothing learned. The test is global -- while NO worker has ever
                # completed a rollout the backend is misconfigured, so give up fast;
                # once any has, it demonstrably works and this is a transient.
                if not self._any_success and consecutive >= self._MAX_WORKER_ERRORS:
                    with self._stats_lock:
                        self._live_workers -= 1
                        self.stats.retired_workers += 1
                        all_dead = self._live_workers <= 0
                    if all_dead:
                        self._stop.set()      # nothing can make progress any more
                    return
                if self._any_success and consecutive == self._MAX_WORKER_ERRORS:
                    warnings.warn(
                        f"{worker.worker_id} has failed {consecutive} rollouts in a "
                        f"row and is backing off, not retiring (it succeeded "
                        f"earlier, so this reads as transient). Last error: "
                        f"{type(e).__name__}: {str(e)[:120]}",
                        RuntimeWarning, stacklevel=2)
                self._stop.wait(min(2.0 ** consecutive, 5.0))
                continue
            consecutive = 0
            self._any_success = True     # the backend demonstrably works
            elapsed = time.time() - t0
            if self.estimator is not None:
                self.estimator.observe(cost, elapsed)
                # a rollout that overran its estimate is a straggler: checkpoint it
                # (partial rollout) rather than letting it define the wall-clock.
                if predicted > 0 and elapsed > self.cfg.duration_timeout_factor * predicted:
                    self.resume_queue.push(ResumeItem(
                        task_id=cluster.id, turn=0, conversation=[],
                        external_handle=None, version_at_checkpoint={self.skill_id: local_v}))
                    with self._stats_lock:
                        self.stats.stragglers_checkpointed += 1
            with self._stats_lock:
                self.stats.rollouts += 1
            if card is not None:
                self.aggregator.ingest(card)
                self.scheduler.record(
                    cluster.id, learning_value=max(0.0, card.before_after_delta),
                    passed=card.before_after_delta > 0,
                )
                with self._stats_lock:
                    self.stats.proposals += 1
            time.sleep(self.cfg.worker_pause)

    # -- aggregator thread ---------------------------------------------------

    def _aggregator_loop(self) -> None:
        no_commit_streak = 0
        consecutive = 0
        while not self._stop.is_set():
            try:
                reports = self.aggregator.step()
                consecutive = 0
            except Exception as e:  # noqa: BLE001 - the merger is the only writer,
                # so if it dies the workers fill a buffer nobody drains. But it also
                # *calls the backend* every sweep (it scores held-out), so ending
                # the run on the first exception made it a single point of failure
                # one transient could take out permanently -- the same mistake
                # `async_evolve` removed after measuring it end with 0 sweeps while
                # the workers were still healthy. Tolerate, and let the run's own
                # budget bound the wait: a truly dead backend retires every worker,
                # which ends the run anyway.
                consecutive += 1
                with self._stats_lock:
                    if self.stats.error is None:
                        self.stats.error = f"{type(e).__name__}: {str(e)[:200]}"
                if consecutive == self._MAX_MERGER_ERRORS:
                    warnings.warn(
                        f"the merger has failed {consecutive} sweeps in a row and "
                        f"is retrying; nothing will merge until it recovers. Last "
                        f"error: {type(e).__name__}: {str(e)[:120]}",
                        RuntimeWarning, stacklevel=2)
                self._stop.wait(min(2.0 ** consecutive, 30.0))
                continue
            committed = False
            with self._stats_lock:
                self.stats.sweeps += 1
                for rep in reports:
                    self.stats.discarded_stale += rep.discarded_stale
                    self.stats.conflicts_dropped += rep.conflicts_dropped
                    if rep.committed_version is not None:
                        self.stats.commits += 1
                        committed = True
                        if rep.fused:
                            self.stats.fused += 1
            if committed:
                no_commit_streak = 0
                self._publish_head()
                acc = self._dev_accuracy()
                with self._stats_lock:
                    self.stats.timeline.append((self.stats.sweeps, acc))
                if acc >= self.cfg.target_accuracy:
                    self._stop.set()
                    return
            else:
                # pipeline is producing evidence but nothing commits (all stale /
                # rejected) -> apply backpressure and force a global sync.
                if self.buffer_pending() > 0:
                    no_commit_streak += 1
                    if no_commit_streak >= self.cfg.stall_patience:
                        self._bump_epoch()
                        no_commit_streak = 0
            time.sleep(self.cfg.aggregator_interval)

    def buffer_pending(self) -> int:
        return self.aggregator.buffer.pending()

    # -- accuracy helpers ----------------------------------------------------

    def _dev_accuracy(self) -> float:
        skill = self.ledger.snapshot(Ledger.DEV).get(self.skill_id)
        return skill.accuracy(self.held_out) if skill else 0.0

    def _stable_accuracy(self) -> float:
        skill = self.ledger.snapshot(Ledger.STABLE).get(self.skill_id)
        return skill.accuracy(self.held_out) if skill else 0.0

    # -- driver --------------------------------------------------------------

    def run(self) -> AsyncStats:
        start = time.time()
        # daemon: a run that overruns max_seconds must not block interpreter exit.
        agg_thread = threading.Thread(target=self._aggregator_loop, name="aggregator",
                                      daemon=True)
        worker_threads = [
            threading.Thread(target=self._worker_loop, args=(w,), name=w.worker_id,
                             daemon=True)
            for w in self.workers
        ]
        agg_thread.start()
        for t in worker_threads:
            t.start()

        # main thread enforces the wall-clock safety bound.
        while not self._stop.is_set():
            if time.time() - start > self.cfg.max_seconds:
                self._stop.set()
                break
            time.sleep(0.01)

        for t in worker_threads:
            t.join(timeout=2.0)
        agg_thread.join(timeout=2.0)

        if self.stats.error is None:
            # Confirmation takes promote_after_k rounds and this loop can stop the
            # instant target_accuracy is hit, so publish the head the run produced.
            self.aggregator.finalize()
        self.stats.wallclock = time.time() - start
        self.stats.final_dev_accuracy = self._dev_accuracy()
        self.stats.final_stable_accuracy = self._stable_accuracy()
        self.stats.oracle_used = self.verifier.budget.oracle_calls_used
        return self.stats
