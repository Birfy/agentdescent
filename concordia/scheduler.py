"""The three-in-one scheduler (design doc, sections 3.1 and 5).

Concordia treats the long tail as *three* distinct problems, each with its own
mechanism:

* :class:`TaskScheduler` -- L-task (data layer): UCB over (task-cluster x
  artifact) so evidence-starved tail artifacts get an exploration bonus and
  converged head artifacts are down-weighted (design doc, section 5.2).
* :class:`AuditScheduler` -- L-value (signal layer): allocates the scarce oracle
  budget to high-blast-radius, high-uncertainty, low-trust diffs, and audits the
  aggregator's own merges to prevent self-pollution (design doc, section 5.3).
* :class:`ResumeQueue` -- L-traj (system layer): turn-level checkpoints of
  timed-out rollouts, resumed against the *latest* ledger, yielding a free A/B
  signal across versions (design doc, section 5.1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .stats import ucb_score


@dataclass
class TaskCluster:
    id: str
    tasks: List[Any]
    # rolling estimate of recent learning value (delta) for this cluster.
    recent_value: float = 0.5
    n_evidence: float = 0.0
    # difficulty filter: clusters that are all-pass or all-fail carry no signal.
    pass_rate: float = 0.5


class TaskScheduler:
    """UCB over task clusters, with a difficulty (zero-advantage) filter."""

    def __init__(self, clusters: List[TaskCluster], c: float = 1.4) -> None:
        self.clusters: Dict[str, TaskCluster] = {cl.id: cl for cl in clusters}
        self.c = c
        self._t = 0

    def _difficulty_weight(self, cl: TaskCluster) -> float:
        """Down-weight clusters with no learning signal (GRPO zero-advantage)."""
        # peaks at pass_rate ~0.5, ~0 at the all-pass / all-fail extremes.
        return 4.0 * cl.pass_rate * (1.0 - cl.pass_rate) + 1e-3

    def _ranked(self) -> List[TaskCluster]:
        self._t += 1
        total = float(self._t)
        scored = []
        for cl in self.clusters.values():
            value = cl.recent_value * self._difficulty_weight(cl)
            scored.append((ucb_score(value, cl.n_evidence, total, self.c), cl.id, cl))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cl for _, _, cl in scored]

    def select(self) -> TaskCluster:
        return self._ranked()[0]

    def select_batch(self, k: int) -> List[TaskCluster]:
        """Lease up to ``k`` *distinct* clusters to workers, UCB-ordered.

        Data-parallel sharding: distinct leases guarantee coverage across the
        task long tail while UCB still front-loads the highest-value clusters
        (design doc, sections 3.1 and 5.2)."""
        ranked = self._ranked()
        if not ranked:
            return []
        # cycle through the ranked list so k > n_clusters still fills every slot.
        return [ranked[i % len(ranked)] for i in range(k)]

    def record(self, cluster_id: str, learning_value: float, passed: bool) -> None:
        cl = self.clusters[cluster_id]
        cl.n_evidence += 1
        # exponential moving estimate of learning value.
        cl.recent_value = 0.8 * cl.recent_value + 0.2 * learning_value
        cl.pass_rate = 0.9 * cl.pass_rate + 0.1 * (1.0 if passed else 0.0)


@dataclass(order=True)
class _AuditItem:
    priority: float
    diff_id: str = field(compare=False)
    payload: Any = field(compare=False)


class AuditScheduler:
    """Allocates oracle budget by estimated value G-hat (design doc, 5.3).

        priority(d) ~ blast_radius(d) * uncertainty(d) / trust(artifact)

    The aggregator's own merge decisions are enqueued here too, closing the
    audit loop over the optimizer itself.
    """

    def __init__(self) -> None:
        self._items: List[_AuditItem] = []
        self._trust: Dict[str, float] = defaultdict(lambda: 1.0)

    def trust(self, artifact_id: str) -> float:
        return self._trust[artifact_id]

    def update_trust(self, artifact_id: str, oracle_agreed: bool) -> None:
        """Raise trust when cheap eval agreed with the oracle, lower it when not."""
        t = self._trust[artifact_id]
        if oracle_agreed:
            self._trust[artifact_id] = min(4.0, t + 0.25)
        else:
            self._trust[artifact_id] = max(0.25, t * 0.5)

    def submit(
        self,
        diff_id: str,
        artifact_id: str,
        blast_radius: float,
        uncertainty: float,
        payload: Any = None,
    ) -> float:
        priority = blast_radius * uncertainty / max(0.25, self._trust[artifact_id])
        # heapq is a min-heap; negate so the highest priority pops first.
        self._items.append(_AuditItem(-priority, diff_id, payload))
        self._items.sort()
        return priority

    def force_oracle(self, blast_radius: float, artifact_id: str) -> bool:
        """High-impact or low-trust changes are forced through the oracle."""
        return blast_radius >= 0.5 or self._trust[artifact_id] < 0.75

    def pop(self) -> Optional[_AuditItem]:
        if not self._items:
            return None
        return self._items.pop(0)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class ResumeItem:
    task_id: str
    turn: int
    conversation: List[Any]
    external_handle: Optional[str]  # e.g. an HPC job id
    version_at_checkpoint: Dict[str, int]


class ResumeQueue:
    """Turn-level checkpoints of timed-out rollouts (partial rollout)."""

    def __init__(self, p90_multiplier: float = 2.0) -> None:
        self.p90_multiplier = p90_multiplier
        self._items: List[ResumeItem] = []

    def should_checkpoint(self, elapsed: float, p90: float) -> bool:
        return elapsed > self.p90_multiplier * p90

    def push(self, item: ResumeItem) -> None:
        self._items.append(item)

    def pop(self) -> Optional[ResumeItem]:
        return self._items.pop(0) if self._items else None

    def __len__(self) -> int:
        return len(self._items)
