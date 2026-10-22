"""Evidence index: one read model over every capability's measured return.

Every rung of the framework already records evidence -- the consolidation
archive holds adopted rules with their self-exam stamps, the harness leaderboard
holds port outcomes per domain, and the portfolio view ranks spending layers by
return-per-cost. But a caller that wants to decide *what to improve next* has to
know which of those readers to call and how to reconcile their shapes. This
module is that reconciliation: :func:`snapshot` composes them into one
:class:`EvidenceSnapshot`, and :func:`rank_slots` turns it into a ranked
shortlist of decision slots for :mod:`agentdescent.self_improve`.

It is deliberately **read-only and deterministic**: the same store yields the
same snapshot, nothing is written, and missing evidence is an empty row rather
than an exception. The archive stays the single source of truth; this module is
a view, not a second store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .consolidation import ConsolidationStore
from .harness import harness_leaderboard
from .portfolio import layers_from_archive

__all__ = ["SlotEvidence", "HarnessEvidence", "EvidenceSnapshot", "snapshot",
           "rank_slots"]


@dataclass(frozen=True)
class SlotEvidence:
    """What the archive says about one decision slot's evolution."""

    name: str
    #: Return-per-cost (``return_`` over mean spend per rule), the portfolio's
    #: signal -- higher means the slot's evolution is paying for itself.
    rpc: float
    #: The slot's spend line this window (total rounds across its rules).
    budget: float
    #: Mean spend per rule (the matching aggregate the rpc divides by).
    spent: float
    #: Number of archived rules for the slot.
    samples: int
    #: Mean measured reward across those rules.
    return_: float
    #: Exam stamps. ``exam is None`` counts as ``trusted`` -- the framework's
    #: provisional-adoption default (seedable until examined).
    trusted: int = 0
    flagged: int = 0
    demoted: int = 0

    @property
    def examined(self) -> int:
        return self.flagged + self.demoted

    @property
    def trusted_ratio(self) -> float:
        """Fraction of the slot's rules neither flagged nor demoted (1.0 empty)."""
        if self.samples <= 0:
            return 1.0
        return self.trusted / self.samples


@dataclass(frozen=True)
class HarnessEvidence:
    """What the archive says about one algorithm port (harness) on a domain."""

    name: str
    rewards: List[float]
    best: Optional[float]

    @property
    def mean(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0


@dataclass(frozen=True)
class EvidenceSnapshot:
    """The reconciled view: slots, ports, and the archive's health counters."""

    slots: Dict[str, SlotEvidence] = field(default_factory=dict)
    harnesses: Dict[str, HarnessEvidence] = field(default_factory=dict)
    total_rules: int = 0
    demoted_rules: int = 0
    flagged_rules: int = 0

    def slot_names(self) -> List[str]:
        return sorted(self.slots)

    @property
    def trusted_ratio(self) -> float:
        if self.total_rules <= 0:
            return 1.0
        return (self.total_rules - self.demoted_rules) / self.total_rules


def _slot_of(record: Any) -> Optional[str]:
    problem = getattr(record, "problem", None) or {}
    slot = problem.get("slot") if isinstance(problem, Mapping) else None
    return slot or None


def snapshot(
    store: ConsolidationStore,
    *,
    config: Optional[Dict[str, Any]] = None,
    artifact_id: str = "meta-rule",
) -> EvidenceSnapshot:
    """Compose the archive's per-slot, per-port and health evidence.

    ``config`` restricts the harness leaderboard to one domain fingerprint (pass
    the same config the slots were evolved on); ``None`` aggregates across every
    domain. Slot rows come from :func:`~agentdescent.portfolio.layers_from_archive`
    (so the return-per-cost is the portfolio's own number) and their exam counts
    from the same records. Deterministic and side-effect free.
    """
    base: Dict[str, Dict[str, Any]] = {}
    for layer in layers_from_archive(store, artifact_id=artifact_id):
        name = layer.name[len("meta:"):] if layer.name.startswith("meta:") else layer.name
        base[name] = {
            "rpc": (layer.return_ / layer.spent) if layer.spent > 0 else 0.0,
            "budget": layer.budget, "spent": layer.spent,
            "samples": layer.samples, "return_": layer.return_,
            "trusted": layer.samples, "flagged": 0, "demoted": 0,
        }

    total = demoted = flagged = 0
    for record in store.iter_archive():
        if getattr(record, "artifact_id", None) != artifact_id:
            continue
        total += 1
        exam = getattr(record, "exam", None)
        if exam not in ("demoted", "flagged"):
            continue
        if exam == "demoted":
            demoted += 1
        else:
            flagged += 1
        slot = _slot_of(record)
        row = base.get(slot) if slot is not None else None
        if row is None:
            continue
        row["trusted"] = max(0, row["trusted"] - 1)
        row[exam] += 1

    slots = {
        name: SlotEvidence(name=name, rpc=row["rpc"], budget=row["budget"],
                           spent=row["spent"], samples=row["samples"],
                           return_=row["return_"], trusted=row["trusted"],
                           flagged=row["flagged"], demoted=row["demoted"])
        for name, row in base.items()
    }
    harnesses = {
        name: HarnessEvidence(name=name, rewards=list(info.get("rewards", [])),
                              best=info.get("best"))
        for name, info in harness_leaderboard(store, config).items()
    }
    return EvidenceSnapshot(slots=slots, harnesses=harnesses, total_rules=total,
                            demoted_rules=demoted, flagged_rules=flagged)


def _rotate(items: Sequence[str], rotate: int) -> List[str]:
    if not items:
        return []
    r = rotate % len(items)
    return list(items[r:]) + list(items[:r])


def rank_slots(
    snap: EvidenceSnapshot,
    *,
    k: int = 2,
    floor: float = 0.05,
    min_samples: int = 1,
    allow: Optional[Sequence[str]] = None,
    rotate: int = 0,
    explore: int = 1,
) -> List[str]:
    """The ``k`` slots to evolve next: paying slots first, plus exploration.

    A slot **pays** when its return-per-cost is at least ``floor``, it has
    ``min_samples`` rules, and none of them are demoted -- a slot that keeps
    producing demoted rules is not a target even if its mean reward is high.
    Paying slots are ranked by (rpc, samples, name).

    Two things keep the loop from starving slots:

    * **exploration** -- up to ``explore`` of the ``k`` slots are reserved for
      slots with fewer than ``min_samples`` rules, so an unexamined slot is
      tried at least once (a paying slot fills the remaining places). Set
      ``explore=0`` to target only paying slots.
    * **rotation** -- ``allow`` (e.g. the loop's declared slots) is rotated by
      ``rotate`` before the under-sampled and cold-start fallbacks are chosen,
      so a tie or a cold archive rotates across ticks instead of always picking
      the same prefix. Callers pass ``rotate=<tick index>``.

    Demoted slots are never targeted; below-floor slots with enough samples are
    not either. An empty ``allow``/snapshot yields ``[]``.
    """
    allowed = list(allow) if allow is not None else None
    space = allowed if allowed is not None else snap.slot_names()
    order = _rotate(space, rotate)

    def row_for(name: str) -> SlotEvidence:
        # a declared slot with no records is under-sampled, not absent -- it
        # must be explorable, even when the archive has never seen it
        return snap.slots.get(name) or SlotEvidence(
            name=name, rpc=0.0, budget=0.0, spent=0.0, samples=0, return_=0.0)

    candidates = [row_for(name) for name in order]
    qualified = [
        row for row in candidates
        if row.samples >= min_samples and row.demoted == 0 and row.rpc >= floor
    ]
    qualified.sort(key=lambda r: (r.rpc, r.samples, r.name), reverse=True)
    chosen: List[str] = []
    # Exploration share: under-sampled slots (including never-archived ones),
    # in rotated order, so an unexamined slot is tried at least once.
    if explore > 0:
        for row in candidates:
            if len(chosen) >= min(explore, k):
                break
            if row.samples < min_samples:
                chosen.append(row.name)
    for row in qualified:
        if len(chosen) >= k:
            break
        if row.name not in chosen:
            chosen.append(row.name)
    # Cold start (nothing paying yet): fill the remaining places from the
    # rotated order, so a fresh loop still tries k slots and moves across ticks.
    if not qualified:
        for name in order:
            if len(chosen) >= k:
                break
            row = row_for(name)
            if name not in chosen and row.demoted == 0:
                chosen.append(name)
    return chosen[:k]
