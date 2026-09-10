"""The audit's verbs as JSON, for the MCP surface and anything else out of process.

Every function here takes a **path** to the audit JSONL rather than a store
object, and that is the whole design rather than a convenience. Constraint 2 of
this package is that truth may take days: the process that dispatched a record
is gone by the time a wet-lab result or a human review comes back, so the
resolving caller has a file and nothing else. These functions are what it calls.

They never raise. A tool call that throws gives the model a stack trace and no
way to act on it, so every failure comes back as ``{"error": ...}`` with a
sentence a model can relay -- the same convention :mod:`agentdescent.mcp`
already uses for run-store errors.

``version=None`` means "the one with the most records", and the reply always
names which version it picked. A store can hold several, and silently answering
about the wrong one is the failure this package is built to prevent, committed
by its own reporting layer.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibrator import Calibrator, Rectification
from .diagnose import residual_stats
from .drift import DriftMonitor
from .scorecard import Cost, rescan, scorecard
from .store import AuditStore

__all__ = ["audit_drift", "audit_pending", "audit_recompute", "audit_rescan",
           "audit_resolve", "audit_scorecard", "audit_status", "open_store",
           "pick_version"]

#: Characters of a stored output to return in a listing. A pending queue of a
#: few hundred rollouts is easily a megabyte, and a tool result is read into a
#: model's context.
OUTPUT_PREVIEW = 400


def open_store(path: str) -> AuditStore:
    """Load the store at ``path``.

    Raises :class:`FileNotFoundError` when the file is not there, which
    :class:`~agentdescent.audit.store.AuditStore` itself does not: an absent
    store is a legitimate starting state for a run about to write one. It is
    never a legitimate answer to a *question* about a store, and reading a typo
    as "this audit has no records yet" is how a caller ends up telling a user
    their verifier is unbiased.
    """
    if not os.path.exists(os.path.expanduser(path)):
        raise FileNotFoundError(
            f"{path} does not exist. An audit store is written by "
            "AuditedReward(store=AuditStore(path)); this is the file it "
            "appends to, not a run directory.")
    return AuditStore(path)


def pick_version(store: AuditStore, version: Optional[str] = None
                 ) -> Tuple[Optional[str], List[str]]:
    """``(chosen, all_versions)``. The busiest version when none is named."""
    counts = Counter(rec.verifier_version for rec in store)
    versions = [v for v, _ in counts.most_common()]
    if version is not None:
        return (version if version in counts else None), versions
    return (versions[0] if versions else None), versions


def _versions_error(version: Optional[str], versions: Sequence[str]) -> Dict[str, Any]:
    if not versions:
        return {"error": "this store holds no records yet"}
    return {"error": f"no records for verifier_version {version!r}",
            "versions": list(versions)}


def _guard(path: str):
    """``(store, None)`` or ``(None, error_payload)``."""
    try:
        return open_store(path), None
    except OSError as e:
        return None, {"error": f"cannot read {path!r}: {e}"}


def _rect_payload(r: Rectification) -> Dict[str, Any]:
    return {
        "verifier_version": r.verifier_version,
        "delta_hat": r.delta_hat, "se": r.se,
        # The term the acceptance gate's variance is mostly made of, and the one
        # the plan's Phase 4 formula omits. Reported beside delta_hat rather
        # than below it for that reason.
        "resid_sd": r.resid_sd,
        "theta": r.theta, "theta_ci": list(r.theta_ci),
        "n": r.n, "n_unlab": r.n_unlab, "gain_factor": r.gain_factor,
        "is_stale": r.is_stale, "stale_reason": r.stale_reason,
        "warnings": list(r.warnings), "covers": list(r.covers),
    }


# -- read-only ---------------------------------------------------------------

def audit_status(path: str, version: Optional[str] = None) -> Dict[str, Any]:
    """The rectifier in force, how much it rests on, and what is outstanding."""
    store, err = _guard(path)
    if err:
        return err
    chosen, versions = pick_version(store, version)
    if chosen is None:
        return _versions_error(version, versions)

    pending = store.pending()
    labelled = store.for_calibration(chosen)
    moments = store.unlabelled_moments(chosen)
    rect = Calibrator(store).current(chosen)
    payload = {
        "path": path, "version": chosen, "versions": versions,
        "records": len(store), "corrupt_lines": store.corrupt,
        "pending": len(pending),
        "calibration_labels": len(labelled),
        "improvement_labels": len(store.for_improvement(chosen)),
        "unlabelled": int(sum(m["n"] for m in moments.values())),
        "strata": sorted(moments) or sorted({r.stratum for r in labelled}),
        "rectification": _rect_payload(rect),
    }
    if labelled:
        payload["observed"] = residual_stats(labelled)
    return payload


def audit_pending(path: str, limit: int = 50, older_than: Optional[float] = None,
                  version: Optional[str] = None) -> Dict[str, Any]:
    """The records waiting on an oracle -- for a person or an experiment system.

    ``output`` is truncated to :data:`OUTPUT_PREVIEW` characters and the reply
    says when it was. A queue of a few hundred rollouts is a megabyte, and a
    tool result lands in a model's context window.
    """
    store, err = _guard(path)
    if err:
        return err
    cutoff = None if older_than is None else time.time() - float(older_than)
    rows = store.pending(older_than=cutoff)
    if version is not None:
        rows = [r for r in rows if r.verifier_version == version]
    total = len(rows)
    rows = rows[:max(0, int(limit))]
    return {
        "path": path, "pending": total, "returned": len(rows),
        "truncated_outputs_at": OUTPUT_PREVIEW,
        "records": [{
            "record_id": r.record_id, "task_id": r.task_id,
            "verifier_version": r.verifier_version,
            "verifier_score": r.verifier_score, "stratum": r.stratum,
            "purpose": r.purpose.value, "inclusion_prob": r.inclusion_prob,
            "dispatched_at": r.dispatched_at,
            "output": r.output[:OUTPUT_PREVIEW],
            "output_truncated": len(r.output) > OUTPUT_PREVIEW,
        } for r in rows],
    }


# -- writes ------------------------------------------------------------------

def audit_resolve(path: str, record_id: str, oracle_score: float
                  ) -> Dict[str, Any]:
    """Attach ground truth to one pending record.

    Three outcomes, and they must not read alike: the record does not exist, it
    exists and is already resolved, or it is now resolved. The second is
    **refused**, because replaying a batch file is what people do when a job
    half-failed, and a silent overwrite would make the calibration set depend on
    how many times that happened.
    """
    store, err = _guard(path)
    if err:
        return err
    existing = store.get(record_id)
    if existing is None:
        return {"ok": False, "error": f"no record {record_id!r} in {path!r}",
                "pending": len(store.pending())}
    if existing.resolved:
        return {"ok": False, "error": (
            f"{record_id!r} is already resolved with oracle_score="
            f"{existing.oracle_score!r}; refusing to overwrite. A second result "
            f"for the same unit is either a duplicate submission or a "
            f"correction, and only a person can say which -- reopen it "
            f"deliberately if it is a correction."),
            "oracle_score": existing.oracle_score,
            "resolved_at": existing.resolved_at}
    try:
        store.resolve(record_id, float(oracle_score))
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "record_id": record_id,
            "oracle_score": float(oracle_score),
            "verifier_score": existing.verifier_score,
            "residual": existing.verifier_score - float(oracle_score),
            "pending": len(store.pending()),
            "note": "the rectifier is not recomputed by this call; "
                    "audit_recompute when a batch is in"}


def audit_recompute(path: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Re-read the store and re-estimate. Returns the new rectification."""
    store, err = _guard(path)
    if err:
        return err
    chosen, versions = pick_version(store, version)
    if chosen is None:
        return _versions_error(version, versions)
    return {"path": path, "version": chosen,
            "rectification": _rect_payload(Calibrator(store).recompute(chosen))}


# -- the two reports ---------------------------------------------------------

def audit_scorecard(path: str, version: Optional[str] = None,
                    previous: Optional[str] = None,
                    max_false_negative: float = 0.05,
                    verifier_seconds: Optional[float] = None,
                    oracle_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Fill the card for ``version``, against ``previous`` if one is named."""
    store, err = _guard(path)
    if err:
        return err
    chosen, versions = pick_version(store, version)
    if chosen is None:
        return _versions_error(version, versions)
    if previous is not None and previous not in versions:
        return {"error": f"no records for previous version {previous!r}",
                "versions": versions}

    cal = Calibrator(store)
    cost = (Cost(verifier_seconds,
                 float("nan") if oracle_seconds is None else oracle_seconds)
            if verifier_seconds is not None else None)
    card = scorecard(
        cal.current(chosen), store.for_calibration(chosen),
        previous=cal.current(previous) if previous else None,
        previous_records=store.for_calibration(previous) if previous else (),
        cost=cost, max_false_negative=max_false_negative)
    return {
        "path": path, "version": chosen, "previous_version": previous,
        "ship": card.ship, "blockers": list(card.blockers),
        "metrics": [{"name": m.name, "value": m.value, "previous": m.previous,
                     "goal": m.goal.name, "verdict": m.verdict,
                     "blocking": m.blocking, "triggered": m.triggered}
                    for m in card.metrics],
        "notes": list(card.notes),
        "markdown": card.to_markdown(),
    }


def audit_rescan(path: str, verifier: str, version: Optional[str] = None,
                 allow: Optional[Sequence[str]] = None,
                 pairs: Optional[Sequence[Sequence[str]]] = None
                 ) -> Dict[str, Any]:
    """Re-score the stored outputs with another verifier and see what moves.

    ``verifier`` is a ``"module:attribute"`` reference resolving to a callable
    ``(record, context) -> float``; ``context`` is ``None`` here, because the
    store holds outputs and not the gold answers a scorer might want. A scorer
    that needs them loads them the way the run did -- it is the caller's own
    code.

    Resolution is bounded by ``allow``, defaulting to this package only.
    Resolving a reference **runs whatever it imports**, so widening it is a
    decision the person operating the server makes, never one a calling model
    can make for them by naming a module.
    """
    from ..workspec import DEFAULT_ALLOWED_PREFIXES, Ref, RefError

    store, err = _guard(path)
    if err:
        return err
    chosen, versions = pick_version(store, version)
    if chosen is None:
        return _versions_error(version, versions)
    try:
        fn = Ref(verifier, call=False).resolve(
            tuple(DEFAULT_ALLOWED_PREFIXES) + tuple(allow or ()))
    except RefError as e:
        return {"error": str(e)}

    records = [r for r in store if r.verifier_version == chosen]
    try:
        report = rescan(records, fn, None,
                        pairs=[tuple(p) for p in pairs] if pairs else None)
    except Exception as e:                    # the caller's code, not ours
        return {"error": f"{verifier} raised {type(e).__name__}: {e}"}
    return {
        "path": path, "version": chosen, "verifier": verifier,
        "n": report.n, "n_artifacts": report.n_artifacts,
        "agreement": report.agreement, "mean_shift": report.mean_shift,
        "sigma_shift": report.sigma_shift,
        "sigma_before": report.sigma_before, "sigma_after": report.sigma_after,
        "n_pairs": report.n_pairs, "flip_rate": report.flip_rate,
        "flipped": [list(f) for f in report.flipped],
        "alarming": report.alarming,
        "markdown": report.to_markdown(),
        "note": "pairs default to every combination of audited artifacts, which "
                "over-counts comparisons no merge ever made; pass the run's own "
                "(base, candidate) signatures for a flip rate about the run",
    }


def audit_drift(path: str, versions: Optional[Sequence[str]] = None
                ) -> Dict[str, Any]:
    """Chart one rectification per verifier version, oldest first.

    A weak chart by construction, and it says so: a version is not a generation,
    and :class:`~agentdescent.audit.drift.DriftMonitor` wants one rectification
    per generation computed on that generation's own labels. What this can show
    is a verifier that got more generous across its own revisions, which is the
    same signature at a coarser grain.
    """
    store, err = _guard(path)
    if err:
        return err
    order = list(versions) if versions else _versions_by_first_seen(store)
    if not order:
        return {"error": "this store holds no records yet"}
    cal = Calibrator(store)
    monitor = DriftMonitor()
    for name in order:
        monitor.observe(cal.current(name), label=name)
    report = monitor.report
    return {
        "path": path, "versions": order,
        "charted": len(report.points),
        "overlapping": report.overlapping,
        "alarming": report.alarming,
        "signals": [{"kind": s.kind.value, "label": s.label, "value": s.value,
                     "z": s.z, "limit": s.limit, "detail": s.detail}
                    for s in report.signals],
        "markdown": report.to_markdown(),
        "note": "one point per verifier version, not per generation; a proper "
                "chart needs a rectification computed on each generation's own "
                "labels",
    }


def _versions_by_first_seen(store: AuditStore) -> List[str]:
    seen: Dict[str, float] = {}
    for rec in store:
        seen.setdefault(rec.verifier_version, rec.dispatched_at)
    return sorted(seen, key=lambda v: seen[v])
