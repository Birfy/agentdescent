"""Admit: one deterministic verdict over an adoption's evidence rings.

The framework has several trust mechanisms -- meta-exam (oracle re-validation),
meta-audit (probability-sampled truth), rule-tournament (self-play promotion),
red-team (OOD gap) and chain-audit (evidence rings) -- each producing its own
status. An adoption path that consults one but not the others is only as
trustworthy as the weakest it skipped. :func:`admit` composes the ring statuses
into **one** verdict, so every adoption is held to the same standard -- the
generalisation of #388's "isolating untrusted rules" from the joint rung to
every adoption path.

The rings are *statuses*, not mechanisms: ``admit`` does **not** re-run an exam
or an attack. Callers pass what each ring already decided (a status string, or
``None`` for "not run"), so the gate is a pure, deterministic, side-effect-free
policy -- cheap enough to run on every adoption, and testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["AdmitVerdict", "admit", "admit_record", "RINGS"]

#: The rings this gate understands, in the order they appear in a report.
RINGS = ("exam", "audit", "tournament", "red_team", "chain")

_PASS = frozenset({"passed", "pass", "verified", "ok", "adopt", "adopted",
                   "trusted"})
_FLAG = frozenset({"flagged", "flag", "warn", "warning", "unverified",
                   "provisional"})
_DEMOTE = frozenset({"demoted", "demote", "fail", "failed", "reject",
                     "rejected", "untrusted"})


@dataclass(frozen=True)
class AdmitVerdict:
    """The composed decision: ``pass`` / ``flag`` / ``demote`` and why.

    ``rings`` holds the normalised status of every ring that was consulted (a
    ring left ``None`` is not run and does not appear). ``reason`` names the
    rings that drove the verdict, so a caller can log *which* piece of evidence
    rejected an adoption rather than just that it did.
    """

    verdict: str
    rings: Dict[str, str]
    reason: str


def _normalise(status: Any) -> str:
    """Map a ring's status to ``pass`` / ``flag`` / ``demote``.

    Unknown strings are treated as ``flag`` -- a status the gate does not
    understand is never allowed to pass silently.
    """
    text = str(status).strip().lower()
    if text in _PASS:
        return "pass"
    if text in _DEMOTE:
        return "demote"
    if text in _FLAG:
        return "flag"
    return "flag"


def admit(
    *,
    exam: Optional[str] = None,
    audit: Optional[str] = None,
    tournament: Optional[str] = None,
    red_team: Optional[str] = None,
    chain: Optional[str] = None,
) -> AdmitVerdict:
    """Compose the ring statuses into one verdict.

    Fail-closed precedence, so one bad ring is never outvoted by good ones:
    any ``demote`` ring rejects the adoption; otherwise any ``flag`` ring marks
    it provisional; otherwise it passes. Rings passed as ``None`` were not run
    and do not participate. Pass an unknown status and the gate flags it rather
    than guessing.
    """
    raw = {"exam": exam, "audit": audit, "tournament": tournament,
           "red_team": red_team, "chain": chain}
    rings = {name: _normalise(status) for name, status in raw.items()
             if status is not None}
    demoted = [name for name, verdict in rings.items() if verdict == "demote"]
    flagged = [name for name, verdict in rings.items() if verdict == "flag"]
    if demoted:
        return AdmitVerdict("demote", rings,
                            f"demoted by {', '.join(demoted)} ring(s)")
    if flagged:
        return AdmitVerdict("flag", rings,
                            f"flagged by {', '.join(flagged)} ring(s)")
    consulted = ", ".join(rings) if rings else "no rings (nothing examined)"
    return AdmitVerdict("pass", rings, f"passed ({consulted})")


def admit_record(
    record: Any,
    *,
    audit: Optional[str] = None,
    tournament: Optional[str] = None,
    red_team: Optional[str] = None,
    chain: Optional[str] = None,
) -> AdmitVerdict:
    """Admit an archived rule using its own ``exam`` stamp plus any ring statuses.

    The record's ``exam`` field is the meta-exam's verdict (``None`` until it is
    examined); the other rings are whatever the caller has already computed for
    this rule. A missing stamp is not a pass by this function's making -- it is
    the framework's documented provisional-adoption default, and
    :meth:`AdmitVerdict.reason` says "nothing examined" so a caller can tell the
    two apart.
    """
    exam = getattr(record, "exam", None) if record is not None else None
    return admit(exam=exam, audit=audit, tournament=tournament,
                 red_team=red_team, chain=chain)
