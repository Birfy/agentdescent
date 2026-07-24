"""Orchestrator: the parallel RSI loop (design doc, section 3.1).

Ties the components together:

    TaskScheduler --lease tasks--> Workers --diff+evidence--> EvidenceBuffer
        --> Aggregator (staleness/conflict/fusion/accept) --CAS--> Ledger
        --broadcast--> Workers ;  AuditScheduler audits merges

The loop is deliberately in-process and deterministic so it can be run and
tested without external models.  Two entry points are provided:

* :meth:`Concordia.run` -- the merge-based system (this framework).
* :func:`run_fork_baseline` -- the archive/fork baseline (DGM-style: parallel but
  never merged), used as the RQ1 control.

Staleness arises naturally: workers refresh their ledger snapshot only every
``refresh_interval`` rounds, so between refreshes their ``base_version`` lags the
head and the aggregator must rebase or discard (design doc, section 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .aggregator import Aggregator, AggregatorConfig, MergeReport
from .domains.router import (
    RouterSkill,
    Task,
    TaskUniverse,
    deserialize_router,
    router_eval,
    serialize_router,
)
from .ledger import Ledger
from .scheduler import AuditScheduler, TaskCluster, TaskScheduler
from .verifier import ThreeLayerVerifier, VerifierBudget
from .worker import Worker


@dataclass
class RoundStat:
    round: int
    dev_accuracy: float
    stable_accuracy: float
    committed: int
    fused: int
    discarded_stale: int
    conflicts_dropped: int
    oracle_used: int


class Concordia:
    """The merge-based parallel self-evolution system."""

    def __init__(
        self,
        repo_path: str,
        universe: TaskUniverse,
        n_workers: int = 6,
        noise: float = 0.15,
        refresh_interval: int = 2,
        skill_id: str = "mol-router",
        config: Optional[AggregatorConfig] = None,
        oracle_budget: int = 300,
        seed: int = 0,
    ) -> None:
        self.universe = universe
        self.skill_id = skill_id
        self.refresh_interval = refresh_interval
        self.train, self.held_out = universe.split()

        self.ledger = Ledger(repo_path, serialize_router, deserialize_router)
        self.ledger.register(RouterSkill(skill_id, table={}), branch=Ledger.DEV)

        self.verifier = ThreeLayerVerifier(
            eval_fn=router_eval,
            held_out=self.held_out,
            budget=VerifierBudget(oracle_calls_remaining=oracle_budget),
            seed=seed,
        )
        self.audit = AuditScheduler()
        self.aggregator = Aggregator(self.ledger, self.verifier, self.audit,
                                     config or AggregatorConfig())

        clusters = universe.clusters(n_clusters=max(2, n_workers))
        self.task_scheduler = TaskScheduler(
            [TaskCluster(id=f"c{i}", tasks=c) for i, c in enumerate(clusters)]
        )
        # one worker per cluster keeps attribution clean; noisy minority injects
        # contradictions for the conflict-resolution path.
        self.workers = [
            Worker(
                worker_id=f"w{i}",
                gold=universe.gold,
                noise=noise if i % 3 == 0 else 0.0,
                seed=seed + i,
            )
            for i in range(n_workers)
        ]
        # each worker caches a snapshot; refreshed on a stagger to create staleness.
        self._snapshots: Dict[str, Tuple[RouterSkill, Dict[str, int]]] = {}

    def _snapshot_for(self, worker: Worker, round_idx: int) -> Tuple[RouterSkill, Dict[str, int]]:
        cached = self._snapshots.get(worker.worker_id)
        due = (round_idx % self.refresh_interval) == (hash(worker.worker_id) % self.refresh_interval)
        if cached is None or due:
            snap = self.ledger.snapshot(Ledger.DEV)
            skill = snap.get(self.skill_id)
            self._snapshots[worker.worker_id] = (skill, snap.version)
        return self._snapshots[worker.worker_id]

    def run(self, rounds: int = 40) -> List[RoundStat]:
        history: List[RoundStat] = []
        for r in range(rounds):
            leases = self.task_scheduler.select_batch(len(self.workers))
            for worker, cluster in zip(self.workers, leases):
                skill, base_version = self._snapshot_for(worker, r)
                card = worker.run(skill, base_version, cluster.tasks)
                if card is not None:
                    self.aggregator.ingest(card)
                    self.task_scheduler.record(
                        cluster.id, learning_value=max(0.0, card.before_after_delta),
                        passed=card.before_after_delta > 0,
                    )
            reports = self.aggregator.step()
            history.append(self._collect(r, reports))
        return history

    def _collect(self, r: int, reports: Sequence[MergeReport]) -> RoundStat:
        dev = self.ledger.snapshot(Ledger.DEV).get(self.skill_id)
        stable = self.ledger.snapshot(Ledger.STABLE).get(self.skill_id)
        return RoundStat(
            round=r,
            dev_accuracy=dev.accuracy(self.held_out) if dev else 0.0,
            stable_accuracy=stable.accuracy(self.held_out) if stable else 0.0,
            committed=sum(1 for x in reports if x.committed_version is not None),
            fused=sum(1 for x in reports if x.fused and x.committed_version is not None),
            discarded_stale=sum(x.discarded_stale for x in reports),
            conflicts_dropped=sum(x.conflicts_dropped for x in reports),
            oracle_used=self.verifier.budget.oracle_calls_used,
        )

    def final_accuracy(self, branch: str = Ledger.DEV) -> float:
        skill = self.ledger.snapshot(branch).get(self.skill_id)
        return skill.accuracy(self.held_out) if skill else 0.0


def run_fork_baseline(
    universe: TaskUniverse,
    n_workers: int = 6,
    noise: float = 0.15,
    rounds: int = 40,
    seed: int = 0,
) -> float:
    """DGM-style archive/fork control: parallel but never merged (RQ1).

    Each worker grows its *own* skill from the tasks it happens to see and the
    forks are never combined.  Because a production system cannot ship a random
    fork per user, we score the best single fork -- which still cannot exceed
    what merging all complementary fixes achieves."""
    train, held_out = universe.split()
    clusters = universe.clusters(n_clusters=max(2, n_workers))
    forks: List[RouterSkill] = []
    for i in range(n_workers):
        worker = Worker(worker_id=f"f{i}", gold=universe.gold,
                        noise=noise if i % 3 == 0 else 0.0, seed=seed + i)
        skill = RouterSkill(f"fork{i}", table={})
        cluster = clusters[i % len(clusters)]
        for _ in range(rounds):
            card = worker.run(skill, {skill.id: skill.version}, cluster)
            if card is None:
                break
            # a fork accepts its own proposals without cross-worker merging.
            skill = skill.apply(card.diff)
        forks.append(skill)
    return max(f.accuracy(held_out) for f in forks)
