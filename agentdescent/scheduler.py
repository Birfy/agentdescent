"""The three-in-one scheduler (design doc, sections 3.1 and 5).

AgentDescent treats the long tail as *three* distinct problems, each with its own
mechanism:

* :class:`TaskScheduler` -- L-task (data layer): UCB over **task clusters**, so
  evidence-starved clusters get an exploration bonus and clusters that carry no
  learning signal are down-weighted (design doc, section 5.2). The design frames
  L-task as an *artifact* problem -- head skills flooded, tail skills starved --
  and the cross-product ``(task-cluster x artifact)`` it calls for is **not**
  implemented: :class:`TaskCluster` has no artifact dimension, and both reference
  runtimes register exactly one artifact, as does ``evolve()``, so there is
  nowhere for the second axis to live yet.
* :class:`AuditScheduler` -- L-value (signal layer): allocates the scarce oracle
  budget to high-blast-radius, high-uncertainty, low-trust diffs, and audits the
  aggregator's own merges to prevent self-pollution (design doc, section 5.3).
* :class:`ResumeQueue` -- L-traj (system layer): turn-level checkpoints of
  timed-out rollouts, resumed against the *latest* ledger, yielding a free A/B
  signal across versions (design doc, section 5.1).
"""

from __future__ import annotations

import heapq
import threading
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
        self._cursor = 0
        # guards select/select_batch/record against concurrent worker threads.
        self._lock = threading.Lock()

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
        with self._lock:
            return self._ranked()[0]

    def select_batch(self, k: int) -> List[TaskCluster]:
        """Lease ``k`` clusters to workers, UCB-ordered, cycling if ``k`` exceeds
        the number of clusters.

        Data-parallel sharding: leases spread coverage across the task long tail
        while UCB front-loads the highest-value clusters (design doc, sections 3.1
        and 5.2).

        **Not distinct beyond ``len(clusters)``.** The docstring used to promise
        distinctness and "guaranteed coverage", which holds only while there are at
        least ``k`` clusters -- and :meth:`TaskUniverse.clusters` drops empty hash
        buckets, so on the default 24-keyword universe that stops being true at 12
        workers, and by 24 workers 7 of them (29%) are handed a cluster another
        worker already has. Those workers roll out the same deterministic tasks and
        produce diffs that content-address to duplicates: real budget, no extra
        evidence. :class:`~agentdescent.orchestrator.AgentDescent` now warns when it
        has fewer clusters than workers."""
        with self._lock:
            ranked = self._ranked()
            if not ranked:
                return []
            # cycle through the ranked list so k > n_clusters fills every slot.
            return [ranked[i % len(ranked)] for i in range(k)]

    def lease_one(self) -> TaskCluster:
        """Atomically pick the single highest-UCB cluster (async worker pull)."""
        with self._lock:
            return self._ranked()[0]

    def lease_round_robin(self) -> TaskCluster:
        """Async worker pull that spreads concurrent workers across clusters.

        Workers pull independently; a rotating cursor over the UCB-ranked list
        keeps them from all hammering the single top cluster while still biasing
        toward high-value ones."""
        with self._lock:
            ranked = self._ranked()
            cl = ranked[self._cursor % len(ranked)]
            self._cursor += 1
            return cl

    def record(self, cluster_id: str, learning_value: float, passed: bool) -> None:
        with self._lock:
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

    #: Cap on queued audits. Nothing in the engines drains this queue, so an
    #: unbounded list would grow for the whole run; keep the highest-priority
    #: items and drop the tail, which is the lowest-value work anyway.
    MAX_QUEUED = 4096

    def __init__(self, max_queued: int = MAX_QUEUED) -> None:
        self._items: List[_AuditItem] = []          # a heap (see submit)
        self._trust: Dict[str, float] = defaultdict(lambda: 1.0)
        self._max_queued = max_queued
        self._dropped = 0
        self._lock = threading.Lock()               # submitted from worker threads

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
        with self._lock:
            priority = blast_radius * uncertainty / max(0.25, self._trust[artifact_id])
            # heapq is a min-heap; negate so the highest priority pops first.
            # heappush is O(log n) -- a full sort per submit was O(n log n), which
            # made a long run quadratic overall.
            heapq.heappush(self._items, _AuditItem(-priority, diff_id, payload))
            # The lowest-priority items sit at the heap's leaves, so shedding them
            # costs a rebuild. Amortise it: let the queue grow to 2x the cap, then
            # rebuild once, keeping the best half -- O(1) per submit on average
            # instead of a rebuild on every submit past the cap.
            if len(self._items) > 2 * self._max_queued:
                kept = heapq.nsmallest(self._max_queued, self._items)
                self._dropped += len(self._items) - len(kept)
                self._items = kept
                heapq.heapify(self._items)
            return priority

    def force_oracle(self, blast_radius: float, artifact_id: str) -> bool:
        """High-impact or low-trust changes are forced through the oracle.

        "High-impact" means L1, and the boundary belongs to
        :func:`~agentdescent.governance.classify` -- a third hand-written
        threshold here (``>= 0.5``) meant an artifact at 0.4 was L1 by governance
        and audited like an L2 skill: never."""
        from .governance import FAST_MAX
        return blast_radius > FAST_MAX or self._trust[artifact_id] < 0.75

    def pop(self) -> Optional[_AuditItem]:
        with self._lock:
            if not self._items:
                return None
            return heapq.heappop(self._items)

    @property
    def dropped(self) -> int:
        """How many low-priority audits were shed to stay under the cap."""
        return self._dropped

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
        # every async worker thread pushes here; list.append happens to be
        # GIL-atomic but pop() + the emptiness check are not.
        self._lock = threading.Lock()

    def should_checkpoint(self, elapsed: float, p90: float) -> bool:
        return elapsed > self.p90_multiplier * p90

    def push(self, item: ResumeItem) -> None:
        with self._lock:
            self._items.append(item)

    def pop(self) -> Optional[ResumeItem]:
        with self._lock:
            return self._items.pop(0) if self._items else None

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# Duration-aware scheduling (L-traj): estimate rollout cost from task size,
# then schedule asynchronously on that estimate.
# ---------------------------------------------------------------------------


@dataclass
class DurationEstimator:
    """Predicts a rollout's wall-clock cost from a task's *size* (e.g. prompt
    length), calibrated online from observed rollouts.

    Agentic rollout time correlates with input size / complexity, but the
    constant isn't known a priori -- so this fits ``seconds ≈ intercept +
    slope * cost`` incrementally by least squares as real durations come in.
    Falls back to a running mean (then a prior) until it has enough samples."""

    prior: float = 0.05      # seconds, before any data
    min_samples: int = 3
    _n: int = 0
    _sx: float = 0.0
    _sy: float = 0.0
    _sxx: float = 0.0
    _sxy: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, cost: float, seconds: float) -> None:
        with self._lock:
            self._n += 1
            self._sx += cost
            self._sy += seconds
            self._sxx += cost * cost
            self._sxy += cost * seconds

    def estimate(self, cost: float) -> float:
        with self._lock:
            if self._n == 0:
                return self.prior
            if self._n < self.min_samples:
                return self._sy / self._n           # running mean
            denom = self._n * self._sxx - self._sx * self._sx
            if denom <= 1e-12:
                return self._sy / self._n           # no spread in cost -> mean
            slope = (self._n * self._sxy - self._sx * self._sy) / denom
            intercept = (self._sy - slope * self._sx) / self._n
            return max(0.0, intercept + slope * cost)

    @property
    def params(self) -> Tuple[float, float]:
        """Return (intercept, slope) of the current fit, or (mean, 0)."""
        with self._lock:
            if self._n < self.min_samples:
                return (self._sy / self._n if self._n else self.prior, 0.0)
            denom = self._n * self._sxx - self._sx * self._sx
            if denom <= 1e-12:
                return (self._sy / self._n, 0.0)
            slope = (self._n * self._sxy - self._sx * self._sy) / denom
            return ((self._sy - slope * self._sx) / self._n, slope)


def lpt_schedule(weights: List[float], n_workers: int) -> Tuple[List[int], float]:
    """Longest-Processing-Time-first assignment of items to workers.

    Assigns each item (heaviest estimated duration first) to the currently
    least-loaded worker -- the classic greedy that minimizes makespan (within
    4/3 of optimal). Returns ``(assignment, makespan)`` where ``assignment[i]``
    is the worker index for item ``i`` and ``makespan`` is the max worker load.
    Dispatching the tail *early* is what keeps one long rollout from defining
    the whole batch's wall-clock."""
    n = max(1, n_workers)
    loads = [0.0] * n
    assignment = [0] * len(weights)
    for i in sorted(range(len(weights)), key=lambda k: weights[k], reverse=True):
        j = min(range(n), key=lambda k: loads[k])
        assignment[i] = j
        loads[j] += weights[i]
    return assignment, max(loads) if loads else 0.0


def fifo_makespan(weights: List[float], n_workers: int) -> float:
    """Makespan of naive round-robin dispatch (the baseline LPT improves on)."""
    n = max(1, n_workers)
    loads = [0.0] * n
    for i, w in enumerate(weights):
        loads[i % n] += w
    return max(loads) if loads else 0.0
