"""Draining the merge path's ranking into the queue a person actually works from.

:class:`~agentdescent.scheduler.AuditScheduler` has ranked every merge decision
since the beginning -- ``blast_radius * uncertainty / trust`` -- and with
``collect=True`` it keeps them in a heap. Nothing in the shipped runtimes has
ever called :meth:`~agentdescent.scheduler.AuditScheduler.pop`.

That is not an oversight, and it is worth saying why before wiring it up.
``force_oracle`` decides whether to audit by a *threshold*, and on the shipped
verifier an audit costs nothing: ``full_eval`` measures the same held-out set the
acceptance test just measured, so the aggregator reuses a number it has already
paid for. When the audit is free, every merge past the threshold gets one and
there is nothing for a ranking to do.

The ranking starts mattering exactly where this package lives: an oracle that is
a wet-lab run, a human reviewer, or a model call that costs money. There the
budget is smaller than the number of merges that clear the threshold, and the
question stops being "which merges qualify" and becomes "which of the qualifying
ones does the experimentalist do first". That is a ranking, and it is the one the
merge path has been computing all along.

So the drain does not fetch anything or spend anything. It turns the heap into
``artifact_signature -> priority``, the store persists it, and
:func:`~agentdescent.audit.service.audit_pending` can hand a person the pending
units in the order the merge path would have chosen -- in another process, next
week, from nothing but the JSONL.

The one join that needs the caller: the scheduler ranks **diffs** and the tap
records **artifact signatures**. Nothing in either knows about the other, so
``signature_of`` is supplied by whoever owns both. Without it the drain still
empties the queue and reports what it could not place, rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .records import AuditRecord

__all__ = ["DrainReport", "drain", "prioritise"]


@dataclass(frozen=True)
class DrainReport:
    """What came off the queue, and what could not be placed.

    ``unplaced`` is not a failure to report quietly: a diff whose signature the
    caller cannot resolve is a merge the audit will never prioritise, and the
    count is the size of that blind spot.
    """

    #: artifact signature -> the highest priority seen for it.
    priorities: Dict[str, float] = field(default_factory=dict)
    popped: int = 0
    unplaced: int = 0
    #: Diff ids that had no signature, up to a handful, for a log.
    examples: List[str] = field(default_factory=list)

    @property
    def placed(self) -> int:
        return self.popped - self.unplaced


def drain(scheduler: Any, *,
          signature_of: Optional[Callable[[Any], Optional[str]]] = None,
          limit: Optional[int] = None) -> DrainReport:
    """Empty the scheduler's queue into ``signature -> priority``.

    ``signature_of(item)`` receives the queued
    :class:`~agentdescent.scheduler._AuditItem` -- it carries ``diff_id`` and the
    ``payload`` the aggregator submitted, which is the ``Diff`` itself -- and
    returns the artifact signature the tap would have recorded, or ``None``.

    **The highest priority wins per signature**, not the latest. An artifact
    audited once at high priority and ten times at low priority is still the one
    to look at first; averaging would let a run of routine merges bury a single
    alarming one.

    Priorities come off the heap negated (it is a min-heap used as a max-heap),
    and this un-negates them, so what comes back reads the way
    :meth:`~agentdescent.scheduler.AuditScheduler.submit` returned it.
    """
    priorities: Dict[str, float] = {}
    popped = unplaced = 0
    examples: List[str] = []
    while limit is None or popped < limit:
        item = scheduler.pop()
        if item is None:
            break
        popped += 1
        signature = None
        if signature_of is not None:
            signature = signature_of(item)
        if not signature:
            unplaced += 1
            if len(examples) < 5:
                examples.append(getattr(item, "diff_id", "?"))
            continue
        # `_AuditItem.priority` is stored negated for the heap.
        value = abs(float(getattr(item, "priority", 0.0)))
        if value > priorities.get(signature, float("-inf")):
            priorities[signature] = value
    return DrainReport(priorities=priorities, popped=popped, unplaced=unplaced,
                       examples=examples)


def prioritise(records: Iterable[AuditRecord],
               priorities: Dict[str, float], *,
               default: float = 0.0) -> List[AuditRecord]:
    """Order pending records by what the merge path thought was risky.

    Stable within a priority: units of equally-ranked artifacts keep the order
    they were dispatched in, so a queue does not reshuffle under someone working
    through it when a new merge lands.

    ``default`` is what a signature the drain never saw is worth. Zero, so an
    unranked unit sorts *after* every ranked one rather than in the middle of
    them -- the ranking is information the queue has, and burying it under
    records nobody ranked would waste it. Raise it to push unranked units up when
    the ranking covers only part of the run.
    """
    rows = list(records)
    order = {rec.record_id: i for i, rec in enumerate(rows)}
    return sorted(
        rows,
        key=lambda r: (-priorities.get(r.artifact_signature, default),
                       order[r.record_id]))
