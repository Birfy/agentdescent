"""A safety case for an archive a recursive loop is about to trust.

`verified_rules` gates seeding, `meta_audit` measures verifier bias, and
`meta_exam` stamps whether a rule transferred. Each answers one question. A
recursive system needs the combined one before it keeps learning from an
archive: *may this archive be trusted to seed the next session?*

`safety_report` is read-only. It checks exam coverage, whether any slot's best
eligible rule is itself unexamined, and (when an audit store is supplied) the
worst oracle-vs-verifier bias in the archive. The result is a small list of
named checks and one of ``safe`` / ``caution`` / ``unsafe`` -- no model calls, no
writes, so it can run before every resume or campaign.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .consolidation import ConsolidationStore, config_fingerprint

__all__ = ["SafetyCheck", "SafetyReport", "safety_report"]


@dataclass(frozen=True)
class SafetyCheck:
    """One named check and why it passed, warned or failed."""

    name: str
    status: str          # "pass" / "warn" / "fail"
    detail: str


@dataclass(frozen=True)
class SafetyReport:
    """The archive's trust posture, with every check that produced it."""

    verdict: str = "caution"       # "safe" / "caution" / "unsafe"
    checks: Tuple[SafetyCheck, ...] = ()
    rules: int = 0
    unexamined: int = 0
    flagged: int = 0
    demoted: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "rules": self.rules,
            "unexamined": self.unexamined,
            "flagged": self.flagged,
            "demoted": self.demoted,
            "checks": [{"name": c.name, "status": c.status,
                        "detail": c.detail} for c in self.checks],
        }


def _slot_of(record: Any) -> Optional[str]:
    problem = getattr(record, "problem", None) or {}
    slot = problem.get("slot") if isinstance(problem, dict) else None
    return slot or None


def safety_report(store: Any, *,
                  config: Optional[Dict[str, Any]] = None,
                  artifact_id: str = "meta-rule",
                  audit: Optional[Any] = None,
                  bias_threshold: float = 0.1,
                  min_exam: Optional[str] = "passed") -> SafetyReport:
    """Check whether the archive is safe to keep seeding from.

    ``audit`` is an optional :class:`~agentdescent.meta_audit.MetaAuditStore`; a
    seedable run's recorded gain bias above ``bias_threshold`` is a hard
    failure. A missing audit is a warning, not a pass: no evidence is not good
    evidence. ``min_exam`` is the same floor as
    :func:`~agentdescent.verified_seed.verified_rules` -- ``"passed"`` by
    default, ``None`` to keep every non-demoted rule, ``"flagged"`` to include
    flagged rules. ``config`` ranks config-matching rules first, exactly as the
    seeder does.
    """
    from .verified_seed import _BAR_RANK, _EXAM_RANK

    if bias_threshold < 0:
        raise ValueError("bias_threshold must be >= 0")
    if min_exam not in _BAR_RANK:
        raise ValueError(
            "min_exam must be None, 'flagged' or 'passed'; "
            f"got {min_exam!r}")
    if isinstance(store, str):
        path = os.path.abspath(os.path.expanduser(store))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"archive {path!r} does not exist")
        store = ConsolidationStore(path)
    records = [r for r in store.iter_archive()
               if getattr(r, "artifact_id", None) == artifact_id
               and _slot_of(r) is not None]
    unexamined = sum(1 for r in records if getattr(r, "exam", None) is None)
    flagged = sum(1 for r in records if getattr(r, "exam", None) == "flagged")
    demoted = sum(1 for r in records if getattr(r, "exam", None) == "demoted")
    checks: List[SafetyCheck] = []
    unknown = sorted({str(getattr(r, "exam", None)) for r in records
                      if getattr(r, "exam", None) not in _EXAM_RANK})
    if unknown:
        checks.append(SafetyCheck(
            "exam_status", "fail",
            f"unknown exam status(es) {unknown}; verified_rules would raise"))
    else:
        checks.append(SafetyCheck(
            "exam_status", "pass", "every exam status is recognised"))

    if not records:
        checks.append(SafetyCheck(
            "archive", "warn",
            "no adopted rule records with a slot descriptor; nothing to trust "
            "yet"))
    else:
        checks.append(SafetyCheck(
            "archive", "pass", f"{len(records)} adopted rule(s)"))

    if records:
        if flagged or demoted:
            checks.append(SafetyCheck(
                "exam", "warn",
                f"{unexamined} unexamined, {flagged} flagged and {demoted} "
                "demoted rule(s) are in the archive"))
        elif unexamined:
            checks.append(SafetyCheck(
                "exam", "warn",
                f"{unexamined} rule(s) have not been examined; "
                f"min_exam={min_exam!r}"))
        else:
            checks.append(SafetyCheck(
                "exam", "pass", "every rule has an exam stamp"))

    # The seed set and its best are the seeder's, not a second interpretation:
    # demoted are dropped, the exam floor is `min_exam`, and config-matching
    # rules rank first exactly as `_archived_rules` sorts them.
    by_slot: Dict[str, List[Any]] = {}
    for record in records:
        by_slot.setdefault(_slot_of(record), []).append(record)
    want = config_fingerprint(config)
    bar = _BAR_RANK[min_exam]
    bad_slots = []
    provisional = []
    seedable: List[Any] = []
    for slot, rows in sorted(by_slot.items()):
        ranked = sorted(
            [r for r in rows if getattr(r, "exam", None) != "demoted"],
            key=lambda r: (getattr(r, "config_hash", "") == want,
                           float(getattr(r, "final_reward", 0.0))),
            reverse=True)
        eligible = [r for r in ranked
                    if getattr(r, "exam", None) in _EXAM_RANK
                    and _EXAM_RANK[getattr(r, "exam", None)] >= bar]
        if not eligible:
            bad_slots.append(slot)
            continue
        seedable.extend(eligible)
        if getattr(eligible[0], "exam", None) is None:
            provisional.append(slot)
    if bad_slots:
        checks.append(SafetyCheck(
            "seedable_best", "fail",
            f"no rule in slot(s) {bad_slots} clears min_exam={min_exam!r}"))
    elif provisional:
        checks.append(SafetyCheck(
            "seedable_best", "warn",
            f"slot(s) {provisional} would seed from an unexamined best rule"))
    elif records:
        checks.append(SafetyCheck(
            "seedable_best", "pass",
            f"every slot has a rule clearing min_exam={min_exam!r}"))

    if audit is None:
        checks.append(SafetyCheck(
            "audit_bias", "warn",
            "no MetaAuditStore supplied; verifier bias was not checked"))
    else:
        worst = None
        worst_run = None
        for record in seedable:
            bias = audit.gain_bias(run_id=getattr(record, "run_id", ""))
            if bias is not None and (worst is None or abs(bias) > abs(worst)):
                worst, worst_run = float(bias), getattr(record, "run_id", "")
        if worst is None:
            checks.append(SafetyCheck(
                "audit_bias", "warn",
                "the audit store has no resolved labels for these runs"))
        elif abs(worst) > bias_threshold:
            checks.append(SafetyCheck(
                "audit_bias", "fail",
                f"{worst_run} has gain bias {worst:+.3f}, above "
                f"{bias_threshold:.3f}"))
        else:
            checks.append(SafetyCheck(
                "audit_bias", "pass",
                f"worst gain bias {worst:+.3f} is within {bias_threshold:.3f}"))

    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    verdict = "unsafe" if fails else ("caution" if warns else "safe")
    reason = "; ".join(c.detail for c in fails or warns) or "all checks passed"
    return SafetyReport(verdict=verdict, checks=tuple(checks),
                        rules=len(records), unexamined=unexamined,
                        flagged=flagged, demoted=demoted, reason=reason)
