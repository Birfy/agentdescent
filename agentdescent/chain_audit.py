"""Chain audit: one adoption, traced through every evidence ring.

The framework's promise is that no number in the loop is unverified: the inner
reward is audited (#273), the meta gain is audited (#426), adoptions are
provisional until the self-exam says so (the meta-exam rung), and routing
evidence can exist for the harness that won. `chain_audit` is that promise made
inspectable — given one archived rule, it walks the rings its adoption passed
through and shows, per ring, whether the evidence is present, within tolerance,
or missing entirely:

* **governance** — the slot's layer (decision plane vs machinery): always
  reported, statically known.
* **decision** — the recorded verdict and transfer, when a decision trail
  exists (e.g. `meta_run_audited`'s verdict lines).
* **meta-gain** — the oracle-vs-verifier gain bias for this run from a
  `MetaAuditStore`; a bias above `bias_threshold` flags the ring.
* **self-exam** — the rule's `exam` stamp (passed / flagged / demoted).
* **inner-reward** — the inner verifier's residual bias, from an `AuditStore`.
* **harness** — routing evidence (`harness_evidence(config)`), when the caller
  has any.

A ring is ``verified`` when its evidence is present and within tolerance,
``flagged`` when present and outside it, and ``unverified`` when the evidence
simply is not there — and a missing ring is reported, never silently assumed.
``ChainAuditReport.verified`` is true only when no ring is flagged or
unverified; the reason names the rings that keep it false, so "this adoption is
not fully verified" is a list, not a shrug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .consolidation import ConsolidationStore
from .meta import SLOTS
from .meta_audit import MetaAuditStore

__all__ = ["ChainLink", "ChainAuditReport", "chain_audit"]


@dataclass(frozen=True)
class ChainLink:
    """One ring of an adoption's evidence chain."""

    layer: str
    status: str               # "verified" / "flagged" / "unverified"
    evidence: Dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ChainAuditReport:
    """What a :func:`chain_audit` pass found, ring by ring."""

    run_id: str
    slot: str
    links: Tuple[ChainLink, ...]
    #: True only when every ring is verified (evidence present and within
    #: tolerance). A flagged or unverified ring makes it false.
    verified: bool
    reason: str


def chain_audit(
    archive: ConsolidationStore,
    run_id: str,
    *,
    meta_audit: Optional[MetaAuditStore] = None,
    inner_audit: Optional[Any] = None,
    harness_evidence: Optional[Callable[[Optional[Dict[str, Any]]],
                                        Optional[Tuple[str, float]]]] = None,
    bias_threshold: float = 0.1,
) -> ChainAuditReport:
    """Trace one archived adoption through its evidence rings.

    Parameters
    ----------
    archive:
        The :class:`~agentdescent.consolidation.ConsolidationStore` holding the
        rule record. ``KeyError`` when ``run_id`` is not archived.
    run_id:
        The archived run to audit.
    meta_audit:
        A :class:`~agentdescent.meta_audit.MetaAuditStore`; the ``meta-gain``
        ring reads ``gain_bias(run_id=...)`` from its resolved sample.
    inner_audit:
        An ``agentdescent.audit.AuditStore``; the ``inner-reward`` ring reads
        the mean residual over its resolved calibration records.
    harness_evidence:
        ``config -> Optional[(harness_name, best_score)]``; the ``harness``
        ring reports the routing evidence for the rule's config, when any.
    bias_threshold:
        A meta-gain (or inner) bias whose magnitude exceeds this flags the ring.
    """
    record = archive.find(run_id)
    if record is None:
        raise KeyError(f"no archived run {run_id!r} to audit")
    slot = (record.problem or {}).get("slot", "")
    links: list = []

    governance = ("decision-plane (L1 harness)" if slot in SLOTS
                  else "machinery (L0/L2)")
    links.append(ChainLink(
        layer="governance", status="verified",
        evidence={"slot": slot, "class": governance},
        reason=f"slot {slot!r} is {governance}"))

    links.append(_decision_link(record, meta_audit, run_id))
    links.append(_meta_gain_link(meta_audit, run_id, bias_threshold))
    links.append(_exam_link(record))
    links.append(_inner_link(inner_audit, bias_threshold))
    links.append(_harness_link(harness_evidence, record.config))

    bad = [l for l in links if l.status != "verified"]
    reason = (f"all {len(links)} rings verified"
              if not bad else
              f"rings not verified: {', '.join(l.layer for l in bad)}")
    return ChainAuditReport(run_id=run_id, slot=slot, links=tuple(links),
                            verified=not bad, reason=reason)


def _decision_link(record, meta_audit, run_id) -> ChainLink:
    verdict = None
    transfer = None
    if meta_audit is not None:
        for line in meta_audit.verdicts():
            if line.get("run_id") == run_id:
                verdict = line.get("verdict")
                transfer = line.get("transfer")
                break
    evidence = {"final_reward": float(record.final_reward),
                "verdict": verdict, "transfer": transfer}
    if verdict is None:
        return ChainLink(
            layer="decision", status="unverified", evidence=evidence,
            reason="no recorded decision trail for this run")
    return ChainLink(
        layer="decision", status="verified", evidence=evidence,
        reason=f"decision {verdict} recorded")


def _meta_gain_link(meta_audit, run_id, threshold) -> ChainLink:
    if meta_audit is None:
        return ChainLink(layer="meta-gain", status="unverified",
                         evidence={}, reason="no meta-audit store supplied")
    bias = meta_audit.gain_bias(run_id=run_id)
    n = sum(1 for _ in meta_audit.iter_records(run_id=run_id, resolved_only=True))
    evidence = {"gain_bias": bias, "resolved": n}
    if bias is None:
        return ChainLink(
            layer="meta-gain", status="unverified", evidence=evidence,
            reason="no resolved oracle sample for this run")
    if abs(bias) > threshold:
        return ChainLink(
            layer="meta-gain", status="flagged", evidence=evidence,
            reason=f"gain bias {bias:+.3f} exceeds threshold {threshold:.3f}")
    return ChainLink(
        layer="meta-gain", status="verified", evidence=evidence,
        reason=f"gain bias {bias:+.3f} within threshold {threshold:.3f}")


def _exam_link(record) -> ChainLink:
    status = record.exam
    evidence = {"exam": status, "exam_flags": record.exam_flags,
                "note": record.exam_note}
    if status is None:
        return ChainLink(layer="self-exam", status="unverified",
                         evidence=evidence,
                         reason="rule has not been examined yet")
    if status == "demoted":
        return ChainLink(layer="self-exam", status="flagged", evidence=evidence,
                         reason="rule was demoted by the self-exam")
    return ChainLink(layer="self-exam", status="verified", evidence=evidence,
                     reason=f"self-exam: {status}")


def _inner_link(inner_audit, threshold) -> ChainLink:
    if inner_audit is None:
        return ChainLink(layer="inner-reward", status="unverified",
                         evidence={}, reason="no inner audit store supplied")
    residuals = [r.residual for r in inner_audit.all()
                 if getattr(r, "resolved", False) and r.residual is not None]
    evidence = {"n_resolved": len(residuals)}
    if not residuals:
        return ChainLink(layer="inner-reward", status="unverified",
                         evidence=evidence,
                         reason="no resolved inner audit records")
    bias = sum(residuals) / len(residuals)
    evidence["residual_bias"] = bias
    if abs(bias) > threshold:
        return ChainLink(layer="inner-reward", status="flagged",
                         evidence=evidence,
                         reason=f"residual bias {bias:+.3f} exceeds "
                                f"threshold {threshold:.3f}")
    return ChainLink(layer="inner-reward", status="verified", evidence=evidence,
                     reason=f"residual bias {bias:+.3f} within threshold")


def _harness_link(harness_evidence, config) -> ChainLink:
    if harness_evidence is None:
        return ChainLink(layer="harness", status="unverified",
                         evidence={}, reason="no harness evidence supplied")
    try:
        outcome = harness_evidence(config)
    except Exception as error:  # noqa: BLE001 - a bad harness must not crash
        return ChainLink(layer="harness", status="flagged", evidence={},
                         reason=f"harness_evidence raised: {error}")
    if outcome is None:
        return ChainLink(layer="harness", status="unverified",
                         evidence={}, reason="no routing evidence for this config")
    if not (isinstance(outcome, (tuple, list)) and len(outcome) == 2
            and isinstance(outcome[0], str)):
        return ChainLink(
            layer="harness", status="flagged", evidence={},
            reason="harness_evidence must return a (name, score) pair")
    name, score = outcome
    return ChainLink(layer="harness", status="verified",
                     evidence={"harness": name, "best": float(score)},
                     reason=f"routed by evidence to {name} at {float(score):.3f}")
