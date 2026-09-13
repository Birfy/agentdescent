"""Where ground truth comes from, and how long it takes to arrive.

Two shapes, and the difference between them is the whole reason this module is
not a function call:

* **:class:`GoldAnswer`** -- truth is a computation. An exact match against a
  reference, a unit test, a checker, a simulator. It returns now, and it is
  cheap enough per unit that the only reason to sample rather than score
  everything is that "everything" is a lot of units.
* **:class:`DeferredOracle`** -- truth is an *event*. A wet-lab experiment, a
  backtest that runs overnight, a human reviewer, a batch job on someone else's
  queue. It returns in hours or days, or never.

The evolution loop cannot wait for the second kind, so neither source is allowed
to block it: :meth:`submit` is called on the evaluation path and must return
promptly. A :class:`GoldAnswer` answers there and then; a
:class:`DeferredOracle` writes down the question and returns ``None``. Whoever
runs the experiment resolves it later, in another process if they like, against
the record id -- which is why the record carries the output rather than a
reference to a live object.

A source that *sometimes* blocks is the dangerous case, and the reason
:class:`GoldAnswer` exists as a distinct class rather than a callable being
sniffed at runtime: declaring which kind you have is cheap, and discovering it
by having a merge thread hang for four hours is not.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from .records import AuditRecord


@runtime_checkable
class OracleSource(Protocol):
    """Ground truth for one ``(task, output)`` pair.

    One method, and its contract is a promise about *time*, not about accuracy:
    it is called from the evaluation path and must not block on anything slow.
    Return a score when truth is already known, ``None`` when it will be
    supplied later through :meth:`AuditStore.resolve`.
    """

    def submit(self, record: AuditRecord, task: object, output: str) -> Optional[float]:
        ...


class GoldAnswer:
    """Synchronous truth: a gold answer, an exact match, a checker, a simulator.

    Wraps a plain ``(task, output) -> float`` -- the same
    :data:`~agentdescent.evolution.Reward` signature the loop already uses,
    which is the point. The cheap agent judge and the truth it is a proxy for
    have identical types, so pairing them is a matter of calling both on the
    same output rather than of building a parallel evaluation stack.

    Failures are swallowed to ``None`` rather than raised. An oracle is a
    *side* channel: it must never be able to fail a rollout that the loop would
    otherwise have scored fine, because that would make turning the audit on
    change the run's outcome, which is the one thing the audit layer promises
    not to do. The exception is recorded in :attr:`errors` so a silent oracle
    is discoverable instead of merely quiet.
    """

    def __init__(self, fn: Callable[[object, str], float]) -> None:
        self.fn = fn
        #: ``(record_id, exception)`` for every submit that raised.
        self.errors: List[tuple] = []
        self._lock = threading.Lock()

    def submit(self, record: AuditRecord, task: object, output: str) -> Optional[float]:
        try:
            return float(self.fn(task, output))
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            with self._lock:
                self.errors.append((record.record_id, exc))
            return None


class DeferredOracle:
    """Truth that arrives later: an experiment, a human, an overnight job.

    :meth:`submit` records the question and returns ``None`` immediately, so the
    evaluation path pays a list append and nothing else. The queue is a *view*
    for whoever answers them; the durable copy is the store's JSONL, which
    carries the output and therefore survives this process exiting.

    Nothing here polls or calls back. Resolution is somebody else's turn:

        for rec in store.pending():
            print(rec.record_id, rec.output)      # run the experiment
        store.resolve(rec.record_id, measured_value)

    That asymmetry is deliberate. A callback would put an unbounded external
    latency on a thread the merge loop is waiting for, and the whole reason this
    class is separate from :class:`GoldAnswer` is to make that impossible rather
    than merely discouraged.
    """

    def __init__(self, *, max_queued: Optional[int] = None) -> None:
        #: Cap on the in-memory view. ``None`` is unbounded. The store is not
        #: capped by this -- dropping a *record* would lose an audit; dropping a
        #: queue entry only loses a reminder, and :meth:`AuditStore.pending`
        #: reconstructs the list from disk anyway.
        self.max_queued = max_queued
        self._queued: List[str] = []
        self._dropped = 0
        self._lock = threading.Lock()

    def submit(self, record: AuditRecord, task: object, output: str) -> None:
        with self._lock:
            if self.max_queued is not None and len(self._queued) >= self.max_queued:
                self._dropped += 1
                return None
            self._queued.append(record.record_id)
        return None

    def queued(self) -> List[str]:
        """Record ids awaiting an answer, oldest first."""
        with self._lock:
            return list(self._queued)

    @property
    def dropped(self) -> int:
        """Queue entries shed by :attr:`max_queued`. The records still exist."""
        with self._lock:
            return self._dropped

    def forget(self, record_id: str) -> None:
        """Drop a resolved id from the queue view."""
        with self._lock:
            if record_id in self._queued:
                self._queued.remove(record_id)


class NullOracle:
    """An oracle that never answers. The default, and it is not a no-op.

    With this installed the tap still samples, still records ``(output,
    verifier_score, inclusion_prob, verifier_version)``, and still writes the
    JSONL -- it simply never fills in a truth. That is the useful state for a
    run whose oracle is a person who has not been asked yet: the questions are
    accumulating in a form somebody can answer next month, on units drawn by a
    known probability rule rather than by whoever happened to look.
    """

    def submit(self, record: AuditRecord, task: object, output: str) -> None:
        return None


@dataclass(frozen=True)
class BatchResult(int):
    """How a batch of answers landed.

    Subclasses `int` and equals ``landed`` so every existing caller that reads
    the return value as a count, compares it or sums it keeps working -- this
    used to be a bare `int` and the extra field is the new information, not a
    new contract.
    """

    landed: int = 0
    already_resolved: int = 0

    def __new__(cls, landed: int = 0, already_resolved: int = 0):
        obj = super().__new__(cls, landed)
        return obj


def resolve_from_mapping(store, answers: Dict[str, float], *, at: Optional[float] = None) -> int:
    """Fill in truth for many pending records at once. Returns how many landed.

    The shape a batch answer actually arrives in -- a CSV from an instrument, a
    dict from a job runner, a spreadsheet a reviewer filled in -- keyed by the
    record id printed from :meth:`AuditStore.pending`.

    **A record that is already resolved is skipped, not raised on.** Replaying a
    batch file is exactly what people do when a job half-failed, and this is the
    function they replay it through; letting `AuditStore.resolve`'s refusal
    escape meant one duplicate id aborted the batch and silently discarded every
    answer after it -- verified, and the answers were not recoverable except by
    hand-diffing the file. The refusal itself is right and stays where it is:
    overwriting a resolved record would throw away an oracle's answer. Skipping
    is the batch-level reading of the same rule.

    ``already`` on the returned report says how many were skipped, because a
    replay that lands nothing and a replay that lands everything both look like
    success from a bare count.
    """
    landed = already = 0
    for record_id, score in answers.items():
        try:
            if store.resolve(record_id, float(score), at=at):
                landed += 1
        except ValueError:
            already += 1
    return BatchResult(landed=landed, already_resolved=already)
