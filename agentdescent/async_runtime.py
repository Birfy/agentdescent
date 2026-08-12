"""The reference barrier-free runtime -- now an adapter over ``async_evolve()``.

Like :mod:`agentdescent.orchestrator`, this used to be a second implementation:
its own worker threads, its own merger thread, its own published head, its own
backpressure. Every one of those now exists in
:func:`~agentdescent.async_evolve.async_evolve`, in three cases *because* this
file had it first and the general engine had to be taught (see #93 and the
`stale_considered` / published-head fixes that came out of it).

So the threads are gone and this is the adapter that remains: it describes the
reference domain the way `async_evolve` wants it and reports the result as
:class:`AsyncStats`.

**Stage orchestration (FlashEvolve)** and the **asynchronous ratio (ROLL Flash)**
are unchanged in substance -- they are what `async_evolve` implements. The knob
names are preserved: ``AsyncConfig.async_ratio`` is the lag budget,
``stall_patience`` the backpressure guard, ``duration_timeout_factor`` the
straggler threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .aggregator import Aggregator, AggregatorConfig
from .async_evolve import async_evolve
from .domains.router import (
    RouterSkill,
    RouterStrategy,
    TaskUniverse,
    cluster_split_frac,
    cluster_tasks,
    router_propose,
    router_reward,
    router_run,
)
from .ledger import Ledger
from .orchestrator import _read_skill, _write_skill
from .scheduler import DurationEstimator, ResumeQueue
from .staleness import StalenessPolicy


@dataclass
class AsyncConfig:
    n_workers: int = 6
    async_ratio: int = 3          # ROLL Flash global lag budget (max head drift)
    #: Resync every worker the moment a sweep commits. Mirrors the engine's
    #: default. Switch off to study the lag budget in isolation: this domain's
    #: rollout is a dictionary lookup, so with it on a worker is never more than
    #: one lookup behind head and eta never leaves 0.
    resync_on_commit: bool = True
    noise: float = 0.15
    target_accuracy: float = 0.98
    max_seconds: float = 20.0     # wall-clock safety bound
    oracle_budget: int = 400
    stall_patience: int = 150     # no-commit sweeps before backpressure sync
    # duration-aware straggler control: a rollout that runs longer than
    # duration_timeout_factor x its *estimated* cost is counted as a straggler.
    duration_timeout_factor: float = 3.0
    seed: int = 0
    #: Re-run a proposal's own rollout with the diff applied, to record a local
    #: before/after delta. The reference loop got that delta **free** -- it
    #: measured accuracy directly, with no second rollout -- so a port that
    #: leaves this on doubles what a counted rollout costs. It stays on by
    #: default because the delta is real evidence and feeds acceptance; the
    #: throughput experiment turns it off, because a second sleep per rollout is
    #: not the dispatch rate it is trying to measure.
    self_verify: bool = True

    # `aggregator_interval` and `worker_pause` used to live here. They paced a
    # loop that no longer exists -- `async_evolve` owns its own sleeps -- and a
    # field that is accepted and then ignored is the one outcome this codebase
    # keeps deciding is worse than a missing feature. Nothing in the tests or
    # examples set them.


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
    #: Rollouts that overran their own predicted duration. **Detection only** --
    #: `run(rendered, task) -> output` is opaque, so there is no continuation
    #: state to check point. The name is kept because it is a published field.
    stragglers_checkpointed: int = 0
    #: Workers that gave up after repeated failures *before any worker had ever
    #: succeeded*. A run can finish cleanly at a fraction of its requested
    #: concurrency, so `error` stays None while throughput quietly drops.
    retired_workers: int = 0
    oracle_used: int = 0
    final_dev_accuracy: float = 0.0
    final_stable_accuracy: float = 0.0
    wallclock: float = 0.0
    #: ``None`` on a clean run; otherwise the backend failure that ended it.
    error: Optional[str] = None
    timeline: List[Tuple[int, float]] = field(default_factory=list)


class AsyncAgentDescent:
    """Barrier-free reference runtime, on the general engine."""

    def __init__(
        self,
        repo_path: str,
        universe: TaskUniverse,
        config: Optional[AsyncConfig] = None,
        agg_config: Optional[AggregatorConfig] = None,
        staleness_policy: Optional[StalenessPolicy] = None,
        estimator: Optional[DurationEstimator] = None,
        skill_id: str = "mol-router",
        aggregator_factory=None,
        rollout=None,
    ) -> None:
        self.cfg = config or AsyncConfig()
        self.universe = universe
        self.skill_id = skill_id
        self.estimator = estimator
        self.agg_config = agg_config or AggregatorConfig()
        self.staleness_policy = staleness_policy
        self._aggregator_factory = aggregator_factory
        #: What a rollout does, defaulting to the reference domain's classifier.
        #: This is the seam `Worker.run` used to be: the resilience tests replace
        #: it with something that raises, and there is no other way to ask "what
        #: does this runtime do when the backend is down".
        self._rollout = rollout or router_run
        #: Kept because it is a published attribute. Nothing pushes to it any
        #: more: the reference loop checkpointed stragglers here, and nothing
        #: ever popped one -- see the table in `docs/modules.md`.
        self.resume_queue = ResumeQueue()
        self.train, self.held_out = universe.split()

        self.strategy = RouterStrategy()
        self._train_tasks, self._held_tasks = cluster_tasks(
            universe, n_clusters=max(2, self.cfg.n_workers))

        self.ledger = Ledger(repo_path, _write_skill, _read_skill)
        self._repo_path = repo_path
        self.aggregator: Optional[Aggregator] = None
        self.verifier: Optional[Any] = None
        self.stats = AsyncStats()
        self.result = None

    # -- reading a branch -----------------------------------------------------

    def _accuracy_on(self, branch: str) -> float:
        try:
            skill = self.ledger.snapshot(branch).get(self.skill_id)
        except Exception:  # noqa: BLE001 -- a diagnostic read must not end a run
            return 0.0
        return skill.accuracy(self.held_out) if skill else 0.0

    def buffer_pending(self) -> int:
        """Cards waiting in the aggregator's buckets, or 0 before a run."""
        return self.aggregator.buffer.pending() if self.aggregator else 0

    # -- the run --------------------------------------------------------------

    def run(self) -> AsyncStats:
        timeline: List[Tuple[int, float]] = []

        def factory(ledger, verifier, audit, config, policy):
            self.verifier = verifier
            if self._aggregator_factory is not None:
                self.aggregator = self._aggregator_factory(
                    ledger, verifier, audit, config, policy)
            else:
                self.aggregator = Aggregator(ledger, verifier, audit, config,
                                             staleness_policy=policy)
            return self.aggregator

        def on_round(info) -> None:
            if info.committed:
                timeline.append((info.round, info.held_out_reward))

        self.result = async_evolve(
            self._train_tasks + self._held_tasks,
            router_reward,
            run=self._rollout,
            propose=router_propose(self.universe.gold, noise=self.cfg.noise,
                                   seed=self.cfg.seed),
            strategy=self.strategy,
            artifact_id=self.skill_id,
            n_workers=self.cfg.n_workers,
            async_ratio=self.cfg.async_ratio,
            resync_on_commit=self.cfg.resync_on_commit,
            max_seconds=self.cfg.max_seconds,
            target_reward=self.cfg.target_accuracy,
            held_out_frac=cluster_split_frac(self._train_tasks, self._held_tasks),
            repo_path=self._repo_path,
            agg_config=self.agg_config,
            staleness_policy=self.staleness_policy,
            aggregator_factory=factory,
            oracle_budget=self.cfg.oracle_budget,
            stall_patience=self.cfg.stall_patience,
            duration_estimator=self.estimator,
            straggler_factor=self.cfg.duration_timeout_factor,
            self_verify=self.cfg.self_verify,
            solved_threshold=0.999,
            seed=self.cfg.seed,
            on_round=on_round,
        )

        r = self.result
        self.stats = AsyncStats(
            rollouts=r.rollouts,
            # Cards produced. `stale_considered` counts every card the staleness
            # gate saw, which is one per proposal.
            proposals=r.stale_considered,
            sweeps=len(r.history),
            commits=sum(h.committed for h in r.history),
            fused=sum(h.fused for h in r.history),
            discarded_stale=r.stale_discarded,
            conflicts_dropped=sum(h.conflicts_dropped for h in r.history),
            forced_refreshes=r.forced_refreshes,
            stragglers_checkpointed=r.stragglers,
            retired_workers=r.retired_workers,
            oracle_used=(self.verifier.budget.oracle_calls_used
                         if self.verifier is not None else 0),
            final_dev_accuracy=self._accuracy_on(Ledger.DEV),
            final_stable_accuracy=self._accuracy_on(Ledger.STABLE),
            # The engine's own clock, which starts after the ledger, verifier
            # and aggregator are built. Timing from the top of this method
            # instead put that setup inside the window, and since it is a fixed
            # cost it depressed the low-worker rows hardest -- which reads as
            # superlinear speedup. Measured: 8 workers came out at 8.5-9.4x
            # against a true ~8.1x.
            wallclock=r.wallclock,
            error=r.error,
            timeline=timeline,
        )
        return self.stats
