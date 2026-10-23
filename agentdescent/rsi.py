"""Longitudinal RSI report: is the archive still getting better?

Every other read model answers a question about *one* run or *one* archive
window. A recursive self-improvement system needs the cross-session question:
across the rules it has adopted, is capability still rising, or has the search
stagnated? `evidence.snapshot` composes the current portfolio health;
this module adds the time axis -- the per-slot sequence of adoptions, the
record that set the current best, and how many adoptions have happened since.

It is read-only and side-effect free: no model calls, no writes.
The verdict is deliberately conservative -- a slot improves only when its best
recorded reward is above its first eligible adoption by more than
``tolerance``, and stagnates after ``stagnation_window`` adoptions without a new
best -- a plateau after an early win is `stagnant`, not `improving`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .consolidation import ConsolidationStore
from .evidence import EvidenceSnapshot, snapshot
from .portfolio import layers_from_archive, portfolio_report

__all__ = ["RSIReport", "SlotRSI", "rsi_report"]


@dataclass(frozen=True)
class SlotRSI:
    """One decision slot's adoption history and whether it is still rising."""

    slot: str
    records: int
    first_reward: float
    best_reward: float
    last_reward: float
    best_run_id: str
    best_index: int
    best_created_at: float
    #: Adoptions since the one that set the current best.
    records_since_best: int
    gain: float
    trusted: int
    unexamined: int
    flagged: int
    demoted: int
    verdict: str
    reason: str
    #: The last few adoptions, oldest first: time, reward, run id, exam.
    recent: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RSIReport:
    """The archive's longitudinal view, plus the current portfolio health."""

    slots: Dict[str, SlotRSI] = field(default_factory=dict)
    total_rules: int = 0
    improving_slots: Tuple[str, ...] = ()
    stagnant_slots: Tuple[str, ...] = ()
    verdict: str = "insufficient"
    reason: str = ""
    evidence: Optional[EvidenceSnapshot] = None
    reallocated: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "total_rules": self.total_rules,
            "improving_slots": list(self.improving_slots),
            "stagnant_slots": list(self.stagnant_slots),
            "reallocated": dict(self.reallocated),
            "evidence": asdict(self.evidence) if self.evidence is not None else None,
            "slots": {
                name: {
                    "records": row.records,
                    "first_reward": row.first_reward,
                    "best_reward": row.best_reward,
                    "last_reward": row.last_reward,
                    "best_run_id": row.best_run_id,
                    "best_index": row.best_index,
                    "best_created_at": row.best_created_at,
                    "records_since_best": row.records_since_best,
                    "gain": row.gain,
                    "trusted": row.trusted,
                    "unexamined": row.unexamined,
                    "flagged": row.flagged,
                    "demoted": row.demoted,
                    "verdict": row.verdict,
                    "reason": row.reason,
                    "recent": list(row.recent),
                }
                for name, row in self.slots.items()
            },
        }


def _slot_of(record: Any) -> Optional[str]:
    problem = getattr(record, "problem", None) or {}
    slot = problem.get("slot") if isinstance(problem, dict) else None
    return slot or None


def _slot_report(slot: str, rows: List[Any], *,
                 tolerance: float, stagnation_window: int) -> SlotRSI:
    rows = sorted(rows, key=lambda r: float(getattr(r, "created_at", 0.0)))
    trusted = sum(1 for r in rows if getattr(r, "exam", None) == "passed")
    unexamined = sum(1 for r in rows if getattr(r, "exam", None) is None)
    flagged = sum(1 for r in rows if getattr(r, "exam", None) == "flagged")
    demoted = sum(1 for r in rows if getattr(r, "exam", None) == "demoted")
    # A trend built on rules the oracle refused (or has not certified) is not
    # evidence of progress. Demoted and flagged rules are excluded from the
    # rewarded trend; unexamined rules remain the provisional-adoption default.
    eligible = [r for r in rows
                if getattr(r, "exam", None) not in ("demoted", "flagged")]
    if not eligible:
        return SlotRSI(
            slot=slot, records=len(rows), first_reward=0.0, best_reward=0.0,
            last_reward=0.0, best_run_id="", best_index=-1,
            best_created_at=0.0, records_since_best=len(rows), gain=0.0,
            trusted=trusted, unexamined=unexamined, flagged=flagged,
            demoted=demoted, verdict="untrusted",
            reason="every adoption is flagged or demoted",
            recent=_recent(rows))
    rewards = [float(getattr(r, "final_reward", 0.0)) for r in eligible]
    best_index = max(range(len(rewards)), key=lambda i: (rewards[i], -i))
    best = rewards[best_index]
    first, last = rewards[0], rewards[-1]
    gain = best - first
    since = len(rewards) - 1 - best_index
    # A plateau after an early win is the case this report exists to catch, so
    # staleness takes precedence over lifetime gain.
    if since >= stagnation_window:
        verdict = "stagnant"
        reason = (f"no new best in the last {since} adoption(s); "
                  f"best {best:.3f}")
    elif gain > tolerance:
        verdict = "improving"
        reason = f"best {first:.3f} -> {best:.3f} (+{gain:.3f})"
    elif last < first - tolerance:
        verdict = "regressing"
        reason = f"latest {last:.3f} is below the first adoption {first:.3f}"
    else:
        verdict = "flat"
        reason = f"no gain beyond {tolerance:.3f} yet"
    return SlotRSI(
        slot=slot, records=len(rows), first_reward=first, best_reward=best,
        last_reward=last,
        best_run_id=str(getattr(eligible[best_index], "run_id", "")),
        best_index=best_index,
        best_created_at=float(getattr(eligible[best_index], "created_at", 0.0)),
        records_since_best=since, gain=gain,
        trusted=trusted, unexamined=unexamined, flagged=flagged,
        demoted=demoted, verdict=verdict, reason=reason,
        recent=_recent(rows))


def _recent(rows: List[Any]) -> Tuple[Dict[str, Any], ...]:
    return tuple({
        "run_id": str(getattr(r, "run_id", "")),
        "reward": float(getattr(r, "final_reward", 0.0)),
        "exam": getattr(r, "exam", None),
        "created_at": float(getattr(r, "created_at", 0.0)),
    } for r in rows[-5:])


def rsi_report(store: Any, *,
               config: Optional[Dict[str, Any]] = None,
               artifact_id: str = "meta-rule",
               slot: Optional[str] = None,
               tolerance: float = 0.02,
               stagnation_window: int = 3,
               floor: float = 0.05,
               min_samples: int = 1) -> RSIReport:
    """The archive's per-slot adoption history and a longitudinal verdict.

    ``store`` is a :class:`~agentdescent.consolidation.ConsolidationStore` or a
    path. ``slot`` restricts the report; ``tolerance`` is the gain that counts
    as improvement and ``stagnation_window`` the number of adoptions without a
    new best before a slot is called stagnant. The evidence snapshot and the
    portfolio reallocation use the same ``config``/``artifact_id``.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if stagnation_window < 1:
        raise ValueError("stagnation_window must be >= 1")
    if isinstance(store, str):
        path = os.path.abspath(os.path.expanduser(store))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"archive {path!r} does not exist")
        store = ConsolidationStore(path)
    grouped: Dict[str, List[Any]] = {}
    for record in store.iter_archive():
        if getattr(record, "artifact_id", None) != artifact_id:
            continue
        name = _slot_of(record)
        if not name or (slot is not None and name != slot):
            continue
        grouped.setdefault(name, []).append(record)
    rows = {name: _slot_report(name, records, tolerance=tolerance,
                               stagnation_window=stagnation_window)
            for name, records in sorted(grouped.items())}
    improving = tuple(n for n, r in rows.items() if r.verdict == "improving")
    stagnant = tuple(n for n, r in rows.items() if r.verdict == "stagnant")
    total = sum(r.records for r in rows.values())
    if total < 2:
        verdict, reason = "insufficient", (
            "fewer than two adopted rules; run more sessions before judging "
            "the trend")
    elif improving and not stagnant:
        verdict, reason = "improving", (
            f"{len(improving)} slot(s) improved: {', '.join(improving)}")
    elif stagnant and len(stagnant) == len(rows):
        verdict, reason = "stagnant", (
            f"no new best in {', '.join(stagnant)} for at least "
            f"{stagnation_window} adoption(s)")
    else:
        verdict = "mixed"
        reason = ", ".join(f"{name}: {row.verdict}"
                           for name, row in rows.items())
    ev = snapshot(store, config=config, artifact_id=artifact_id)
    layers = layers_from_archive(store, artifact_id=artifact_id)
    reallocated = (portfolio_report(layers, floor=floor,
                                    min_samples=min_samples).reallocated
                   if layers else {})
    return RSIReport(slots=rows, total_rules=total,
                     improving_slots=improving, stagnant_slots=stagnant,
                     verdict=verdict, reason=reason, evidence=ev,
                     reallocated=dict(reallocated))
