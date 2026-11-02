"""One governed path for a harness or code self-modification.

The individual safety seams are deliberately separate: `safe_selfmod` reviews
syntax/capabilities and knows how to roll a change back, `proofs` carries an
invariant certificate, `approval` queues a human decision, `trust_evidence`
assembles the canary and evaluation reports, and `promotion` judges the whole
bundle. A caller that wires them by hand can easily skip one. This module is
that wiring: every harness change goes through the gate, the prover, the human
approval queue, the evidence builder and the promotion decision, in that order.

The path fails closed. A denied capability, a failed proof, a pending or
rejected human request all return a report without evidence and never call the
promotion seam. A canary that loses rolls back. The returned
`GovernedHarnessReport` keeps every decision and report it saw, plus a
non-destructive `rollback_plan` when the caller supplies the current and
previous artifact contents.

    report = govern_harness_change(
        "harness-1", spec, before_scores, after_scores,
        gate=SafeSelfModifier({"files:write"}),
        prover=ProofRunner([...]),
        approval=ApprovalStore("/var/lib/approvals.jsonl"),
        contaminated=False,
        rollback_current={"harness.py": "new"},
        rollback_previous={"harness.py": "old"},
    )
    report.promoted, report.verdict, report.rollback

This is a composition seam, not another policy engine: it does not re-scan
source, re-run invariants, re-run the canary, or decide approval itself.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from .approval import ApprovalRequest, ApprovalStore
from .promotion import (
    PromotionDecision,
    PromotionRules,
    ProofEvidence,
    promote,
)
from .proofs import ProofCertificate, ProofRunner
from .safe_selfmod import ModificationSpec, SafeSelfModifier, SafetyDecision
from .safety_controller import rollback_plan
from .trust_evidence import TrustEvidence, build_trust_evidence

__all__ = ["GovernedHarnessReport", "govern_harness_change"]


@dataclass(frozen=True)
class GovernedHarnessReport:
    """Everything the governed harness path decided about one change."""

    artifact_id: str
    safety: SafetyDecision
    approval: Optional[ApprovalRequest]
    proof: Optional[ProofCertificate]
    evidence: Optional[TrustEvidence]
    decision: PromotionDecision
    rollback: Mapping[str, Any]
    verdict: str
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.rollback, MappingABC):
            raise TypeError("rollback must be a mapping")
        object.__setattr__(self, "rollback", dict(self.rollback))

    @property
    def promoted(self) -> bool:
        return self.decision.action == "promote" and self.decision.accepted

    def to_dict(self) -> dict:
        return _json_safe({
            "artifact_id": self.artifact_id,
            "safety": asdict(self.safety),
            "approval": (None if self.approval is None
                         else asdict(self.approval)),
            "proof": None if self.proof is None else self.proof.to_dict(),
            "evidence": (None if self.evidence is None
                         else self.evidence.to_dict()),
            "decision": self.decision.to_dict(),
            "rollback": dict(self.rollback),
            "verdict": self.verdict,
            "detail": self.detail,
            "promoted": self.promoted,
        })


def _json_safe(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _detail(decision: PromotionDecision) -> str:
    parts = list(decision.reasons)
    if decision.missing:
        parts.append("missing: " + ", ".join(decision.missing))
    return "; ".join(parts) if parts else decision.action


def _report(artifact_id: str, safety: SafetyDecision, *,
            decision: PromotionDecision,
            approval: Optional[ApprovalRequest] = None,
            proof: Optional[ProofCertificate] = None,
            evidence: Optional[TrustEvidence] = None,
            rollback: Optional[Mapping[str, Any]] = None,
            verdict: Optional[str] = None,
            detail: Optional[str] = None) -> GovernedHarnessReport:
    return GovernedHarnessReport(
        artifact_id=artifact_id,
        safety=safety,
        approval=approval,
        proof=proof,
        evidence=evidence,
        decision=decision,
        rollback={} if rollback is None else rollback,
        verdict=decision.action if verdict is None else verdict,
        detail=_detail(decision) if detail is None else detail,
    )


def _synthetic(artifact_id: str, action: str, reason: str, *,
               risk: Optional[float] = None) -> PromotionDecision:
    return PromotionDecision(
        artifact_id=artifact_id,
        action=action,
        accepted=False,
        reasons=(reason,),
        risk=risk,
    )


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _check_scores(mapping: Any, name: str) -> Mapping[str, float]:
    if not isinstance(mapping, MappingABC):
        raise TypeError(f"{name} must be a mapping")
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    return mapping


def _check_risk(risk: Any) -> Optional[float]:
    if risk is None:
        return None
    if isinstance(risk, bool) or not isinstance(risk, (int, float)):
        raise TypeError("risk must be a number or None")
    number = float(risk)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("risk must be in [0, 1]")
    return number


def _check_rollback(current: Any, previous: Any) -> None:
    if (current is None) != (previous is None):
        raise ValueError("rollback_current and rollback_previous must be "
                         "supplied together")
    for name, mapping in (("rollback_current", current),
                          ("rollback_previous", previous)):
        if mapping is None:
            continue
        if not isinstance(mapping, MappingABC):
            raise TypeError(f"{name} must be a mapping")
        for path, content in mapping.items():
            if not isinstance(path, str) or not path:
                raise ValueError(f"{name} keys must be non-empty strings")
            if not isinstance(content, str):
                raise TypeError(f"{name} values must be strings")


def govern_harness_change(
        artifact_id: str,
        modification: ModificationSpec,
        before: Mapping[str, float],
        after: Mapping[str, float],
        *,
        gate: SafeSelfModifier,
        rules: Optional[PromotionRules] = None,
        prover: Optional[ProofRunner] = None,
        approval: Optional[ApprovalStore] = None,
        approval_key: Optional[str] = None,
        approval_reason: str = "high-risk harness change",
        rollback_current: Optional[Mapping[str, str]] = None,
        rollback_previous: Optional[Mapping[str, str]] = None,
        kind: str = "harness",
        contaminated: bool = False,
        risk: Optional[float] = None,
        **evidence_kwargs: Any) -> GovernedHarnessReport:
    """Run one harness change through safety, proof, approval and promotion."""
    artifact_id = _require_text(artifact_id, "artifact_id")
    if not isinstance(modification, ModificationSpec):
        raise TypeError("modification must be a ModificationSpec")
    if not isinstance(gate, SafeSelfModifier):
        raise TypeError("gate must be a SafeSelfModifier")
    if rules is not None and not isinstance(rules, PromotionRules):
        raise TypeError("rules must be a PromotionRules or None")
    if prover is not None and not isinstance(prover, ProofRunner):
        raise TypeError("prover must be a ProofRunner or None")
    if approval is not None and not isinstance(approval, ApprovalStore):
        raise TypeError("approval must be an ApprovalStore or None")
    if approval_key is not None:
        approval_key = _require_text(approval_key, "approval_key")
    approval_reason = _require_text(approval_reason, "approval_reason")
    kind = _require_text(kind, "kind")
    if not isinstance(contaminated, bool):
        raise TypeError("contaminated must be a bool")
    risk = _check_risk(risk)
    before = _check_scores(before, "before")
    after = _check_scores(after, "after")
    _check_rollback(rollback_current, rollback_previous)
    if "approved" in evidence_kwargs:
        raise TypeError("approved is decided by the approval gate")

    safety = gate.review(modification)
    if not safety.allowed:
        decision = _synthetic(artifact_id, "reject", safety.reason,
                              risk=modification.risk)
        return _report(artifact_id, safety, decision=decision)

    certificate: Optional[ProofCertificate] = None
    proof_evidence: Optional[ProofEvidence] = None
    if prover is not None:
        certificate = prover.prove(modification, artifact_id=artifact_id)
        verification = prover.verify(certificate, modification)
        proof_evidence = ProofEvidence(certificate, verification)
        if not certificate.passed or not verification.valid:
            failed = [result.name for result in certificate.results
                      if not result.passed]
            reason = "proof failed"
            if failed:
                reason += ": " + ",".join(failed)
            if not verification.valid:
                reason += f"; verification: {verification.reason}"
            decision = _synthetic(artifact_id, "reject", reason,
                                  risk=modification.risk)
            return _report(artifact_id, safety, decision=decision,
                           proof=certificate)

    approval_request: Optional[ApprovalRequest] = None
    approved: Optional[bool] = None
    if approval is not None:
        key = approval_key
        if key is None:
            digest = hashlib.sha256(modification.source.encode()).hexdigest()
            key = "harness:" + artifact_id + ":" + digest[:16]
        capabilities = ",".join(modification.requested_capabilities) or "-"
        summary = (f"{kind}:{modification.target} "
                   f"risk={modification.risk:.3f} caps={capabilities} "
                   f"tasks={len(before)}->{len(after)}")
        approval_request = approval.request(
            key, artifact_id, approval_reason, summary=summary)
        if approval_request.status == "rejected":
            decision = _synthetic(
                artifact_id, "reject",
                f"human approval rejected for {key}", risk=modification.risk)
            return _report(artifact_id, safety, decision=decision,
                           approval=approval_request, proof=certificate)
        if approval_request.status == "pending":
            decision = _synthetic(
                artifact_id, "defer",
                f"human approval pending for {key}", risk=modification.risk)
            return _report(artifact_id, safety, decision=decision,
                           approval=approval_request, proof=certificate,
                           verdict="needs-approval")
        approved = True

    effective_risk = modification.risk if risk is None else risk
    evidence = build_trust_evidence(
        before, after, approved=approved, risk=effective_risk,
        contaminated=contaminated, **evidence_kwargs)

    request = evidence.to_promotion_request(
        artifact_id, artifact_hash=None, proof=proof_evidence)
    if rules is None:
        rules = PromotionRules(require_proof=prover is not None,
                               require_approval=approval is not None)
    decision = promote(request, rules)

    rollback: Mapping[str, Any] = {}
    if rollback_current is not None and rollback_previous is not None:
        rollback = rollback_plan(rollback_current, rollback_previous)

    return _report(artifact_id, safety, decision=decision,
                   approval=approval_request, proof=certificate,
                   evidence=evidence, rollback=rollback)
