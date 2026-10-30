"""One fail-closed promotion decision over the evidence we already produce.

Evaluation, statistics, proofs, canary checks and policy admission exist as
separate modules. A release still needs one answer: *may this candidate be
promoted, and if not, what exactly is missing or failed?* This module collects
those evidence objects into one decision and refuses to guess:

* missing required evidence -> ``defer`` (not a silent pass);
* a failed gate -> ``reject``;
* a canary that says roll back -> ``rollback``;
* everything required is present and green -> ``promote``.

The decision carries every input it saw and a stable ``evidence_hash`` so the
same bundle can be attached to a manifest / lineage record. It deliberately
does not re-run the gates: callers pass the reports produced by
`evalops`, `conformal`, `sequential`, `safety_controller`, `proofs` and
`policy_release`. The kernel of trust is the fail-closed assembly, not another
statistical test.

    decision = promote(PromotionRequest(
        artifact_id="skill-1",
        eval_result=eval_result,
        contaminated=False,
        conformal=conformal_report,
        sequential=sequential_report,
        canary=canary_result,
        proof=ProofEvidence(certificate, verification),
        admission=admission,
        risk=0.2,
        approved=True,
    ))
    decision.action, decision.reasons, decision.evidence_hash
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .conformal import ConformalReport
from .evalops import EvalResult
from .policy_release import AdmissionResult
from .proofs import ProofCertificate, ProofVerification
from .safety_controller import CanaryResult
from .sequential import SequentialReport

__all__ = [
    "PromotionAction",
    "PromotionRules",
    "ProofEvidence",
    "PromotionRequest",
    "PromotionDecision",
    "promote",
]


class PromotionAction:
    """The closed vocabulary a promotion decision may emit."""

    PROMOTE = "promote"
    DEFER = "defer"
    REJECT = "reject"
    ROLLBACK = "rollback"

    ALL = (PROMOTE, DEFER, REJECT, ROLLBACK)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False, default=str)


@dataclass(frozen=True)
class PromotionRules:
    """What evidence a promotion must present, and the bars it must clear."""

    require_eval: bool = True
    require_contamination: bool = True
    require_risk: bool = True
    require_conformal: bool = False
    require_sequential: bool = False
    require_canary: bool = True
    require_proof: bool = False
    require_admission: bool = False
    require_approval: bool = False
    allow_contaminated: bool = False
    min_effect: float = 0.0
    min_tasks: int = 2
    max_risk: float = 1.0

    def __post_init__(self) -> None:
        for name in ("require_eval", "require_contamination", "require_risk",
                     "require_conformal", "require_sequential",
                     "require_canary", "require_proof", "require_admission",
                     "require_approval", "allow_contaminated"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if _finite(self.min_effect, "min_effect") < 0:
            raise ValueError("min_effect must be >= 0")
        if isinstance(self.min_tasks, bool) or not isinstance(
                self.min_tasks, int) or self.min_tasks < 1:
            raise ValueError("min_tasks must be an int >= 1")
        risk = _finite(self.max_risk, "max_risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("max_risk must be in [0, 1]")


@dataclass(frozen=True)
class ProofEvidence:
    """A proof certificate plus the verification that actually checked it."""

    certificate: ProofCertificate
    verification: ProofVerification

    def __post_init__(self) -> None:
        if not isinstance(self.certificate, ProofCertificate):
            raise TypeError("certificate must be a ProofCertificate")
        if not isinstance(self.verification, ProofVerification):
            raise TypeError("verification must be a ProofVerification")


@dataclass(frozen=True)
class PromotionRequest:
    """All evidence a caller has for one candidate."""

    artifact_id: str
    artifact_hash: Optional[str] = None
    eval_result: Optional[EvalResult] = None
    contaminated: Optional[bool] = None
    conformal: Optional[ConformalReport] = None
    sequential: Optional[SequentialReport] = None
    canary: Optional[CanaryResult] = None
    proof: Optional[ProofEvidence] = None
    admission: Optional[AdmissionResult] = None
    risk: Optional[float] = None
    approved: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id is required")
        if self.artifact_hash is not None and (
                not isinstance(self.artifact_hash, str)
                or not self.artifact_hash.strip()):
            raise ValueError("artifact_hash must be a non-empty string or None")
        if self.contaminated is not None and not isinstance(self.contaminated, bool):
            raise TypeError("contaminated must be a bool or None")
        if self.risk is not None:
            risk = _finite(self.risk, "risk")
            if not 0.0 <= risk <= 1.0:
                raise ValueError("risk must be in [0, 1]")
        if self.approved is not None and not isinstance(self.approved, bool):
            raise TypeError("approved must be a bool or None")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class PromotionDecision:
    """The verdict, the evidence it saw and a stable hash of that evidence."""

    artifact_id: str
    action: str
    accepted: bool
    reasons: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    evidence_hash: str = ""
    decision_hash: str = ""
    risk: Optional[float] = None

    def __post_init__(self) -> None:
        if self.action not in PromotionAction.ALL:
            raise ValueError(f"unknown promotion action {self.action!r}")
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "action": self.action,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "missing": list(self.missing),
            "evidence": dict(self.evidence),
            "evidence_hash": self.evidence_hash,
            "decision_hash": self.decision_hash,
            "risk": self.risk,
        }


def _evidence(request: PromotionRequest) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "artifact_id": request.artifact_id,
        "artifact_hash": request.artifact_hash,
        "contaminated": request.contaminated,
        "risk": request.risk,
        "approved": request.approved,
        "metadata": dict(request.metadata),
    }
    if request.eval_result is not None:
        evidence["eval"] = request.eval_result.to_dict()
    if request.conformal is not None:
        evidence["conformal"] = asdict(request.conformal)
    if request.sequential is not None:
        evidence["sequential"] = asdict(request.sequential)
    if request.canary is not None:
        evidence["canary"] = asdict(request.canary)
    if request.proof is not None:
        evidence["proof"] = {
            "certificate": request.proof.certificate.to_dict(),
            "verification": asdict(request.proof.verification),
        }
    if request.admission is not None:
        evidence["admission"] = request.admission.to_dict()
    return _safe(evidence)


def promote(request: PromotionRequest,
            policy: Optional[PromotionRules] = None) -> PromotionDecision:
    """Assemble and judge one promotion; missing or failed evidence never passes."""
    if not isinstance(request, PromotionRequest):
        raise TypeError("request must be a PromotionRequest")
    policy = policy or PromotionRules()
    if not isinstance(policy, PromotionRules):
        raise TypeError("policy must be a PromotionRules")

    reasons: list[str] = []
    missing: list[str] = []
    deferred: list[str] = []
    hard_fail = False
    rollback = False

    if request.contaminated is None:
        if policy.require_contamination:
            missing.append("contamination")
    elif request.contaminated and not policy.allow_contaminated:
        hard_fail = True
        reasons.append("dataset is contaminated")

    result = request.eval_result
    if result is None:
        if policy.require_eval:
            missing.append("eval")
    else:
        if result.n < policy.min_tasks:
            deferred.append(
                f"eval has {result.n} task(s), needs {policy.min_tasks}")
        if not result.passed:
            hard_fail = True
            reasons.append("eval gate did not pass")
        if result.delta < policy.min_effect:
            hard_fail = True
            reasons.append(
                f"eval delta {result.delta:.6g} < min_effect "
                f"{policy.min_effect:.6g}")

    report = request.conformal
    if report is None:
        if policy.require_conformal:
            missing.append("conformal")
    elif report.decision == "insufficient":
        deferred.append("conformal evidence is insufficient")
    elif report.decision != "accept":
        hard_fail = True
        reasons.append(f"conformal {report.decision}: {report.detail}")

    report = request.sequential
    if report is None:
        if policy.require_sequential:
            missing.append("sequential")
    elif report.decision == "continue":
        deferred.append("sequential evidence has not reached a decision")
    elif report.decision != "accept":
        hard_fail = True
        reasons.append(f"sequential {report.decision}: {report.detail}")

    report = request.canary
    if report is None:
        if policy.require_canary:
            missing.append("canary")
    elif report.verdict == "rollback":
        rollback = True
        reasons.append(f"canary rolled back: {report.reason}")
    elif report.verdict == "hold":
        note = (f"; missing {len(report.missing)} task(s)"
                if report.missing else "")
        deferred.append(f"canary is still holding: {report.reason}{note}")
    elif report.verdict != "promote":
        hard_fail = True
        reasons.append(f"canary verdict {report.verdict!r} is not promote")

    proof = request.proof
    if proof is None:
        if policy.require_proof:
            missing.append("proof")
    else:
        if proof.certificate.artifact_id != request.artifact_id:
            hard_fail = True
            reasons.append(
                f"proof artifact {proof.certificate.artifact_id!r} does not "
                f"match {request.artifact_id!r}")
        if request.artifact_hash is not None and (
                proof.certificate.artifact_hash != request.artifact_hash):
            hard_fail = True
            reasons.append("proof artifact hash does not match the request")
        if not proof.certificate.passed:
            hard_fail = True
            reasons.append("proof certificate is marked failed")
        if not proof.verification.valid:
            hard_fail = True
            reasons.append(
                f"proof verification failed: {proof.verification.reason}")

    admission = request.admission
    if admission is None:
        if policy.require_admission:
            missing.append("admission")
    else:
        if not admission.signature.valid:
            hard_fail = True
            reasons.append(
                f"admission signature failed: {admission.signature.reason}")
        elif admission.needs_approval:
            deferred.append(f"admission requires approval: {admission.detail}")
        elif not admission.allowed:
            hard_fail = True
            reasons.append(f"admission denied: {admission.detail}")
        if request.artifact_hash is not None and (
                admission.manifest_hash != request.artifact_hash):
            hard_fail = True
            reasons.append("admission manifest hash does not match the request")

    if policy.require_risk and request.risk is None:
        missing.append("risk")
    elif request.risk is not None and request.risk > policy.max_risk:
        hard_fail = True
        reasons.append(
            f"risk {request.risk:.3f} > max_risk {policy.max_risk:.3f}")

    if request.approved is False:
        # An explicit "no" is a hard failure even when approval is not required:
        # a caller that supplies it is not asking for the default behaviour.
        hard_fail = True
        reasons.append("human approval was explicitly denied")
    elif policy.require_approval and request.approved is not True:
        deferred.append("human approval is required")

    if rollback:
        action = PromotionAction.ROLLBACK
    elif hard_fail:
        action = PromotionAction.REJECT
    elif missing or deferred:
        action = PromotionAction.DEFER
    else:
        action = PromotionAction.PROMOTE

    evidence = _evidence(request)
    evidence_hash = hashlib.sha256(_canonical(evidence).encode()).hexdigest()
    policy_dict = asdict(policy)
    decision_hash = hashlib.sha256(_canonical({
        "evidence_hash": evidence_hash, "policy": policy_dict,
        "action": action}).encode()).hexdigest()
    accepted = action == PromotionAction.PROMOTE
    if not reasons and not deferred and not missing:
        reasons.append("all required evidence is green")
    reasons.extend(deferred)
    return PromotionDecision(
        artifact_id=request.artifact_id, action=action, accepted=accepted,
        reasons=tuple(reasons), missing=tuple(missing), evidence=evidence,
        evidence_hash=evidence_hash, decision_hash=decision_hash,
        risk=request.risk)
