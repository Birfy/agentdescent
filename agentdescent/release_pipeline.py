"""One trusted release pipeline from evidence to a signed ledger receipt.

The pieces already exist, but a caller has to connect them by hand:

    evidence = build_trust_evidence(before, after, ...)
    decision = promote(evidence.to_promotion_request("skill-1"), rules)
    release = PromotionLedger(LineageGraph()).record(
        decision, artifact_id="skill-1", content_hash=..., key_id="ci",
        secret=key, rules=rules, parents=("skill-0",))

This module is that missing seam. `release_evidence` turns trust evidence into
a promotion request, judges it with the same rules, and only when the decision
is an accepted promotion signs a manifest and writes the lineage receipt.
`release_candidate` builds the evidence from paired task scores first.

A reject or defer is a normal result: the report carries the verdict and the
ledger is untouched. Missing signing material for an otherwise accepted
promotion is an input error, not a silent unsigned release.

    report = release_candidate(
        before, after, artifact_id="skill-1", contaminated=False, risk=0.1,
        ledger=ledger, key_id="ci", secret=key, parents=("skill-0",))
    report.released, report.release.manifest.manifest_hash()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .policy_release import AdmissionResult
from .promotion import (
    ProofEvidence,
    PromotionDecision,
    PromotionRules,
    promote,
)
from .promotion_ledger import PromotionLedger, ReleaseRecord
from .trust_evidence import TrustEvidence, build_trust_evidence

__all__ = ["ReleaseReport", "release_evidence", "release_candidate"]


@dataclass(frozen=True)
class ReleaseReport:
    """The evidence, the promotion verdict and the signed release, if any."""

    artifact_id: str
    evidence: TrustEvidence
    decision: PromotionDecision
    release: Optional[ReleaseRecord]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        if not isinstance(self.evidence, TrustEvidence):
            raise TypeError("evidence must be a TrustEvidence")
        if not isinstance(self.decision, PromotionDecision):
            raise TypeError("decision must be a PromotionDecision")
        if self.release is not None and not isinstance(self.release, ReleaseRecord):
            raise TypeError("release must be a ReleaseRecord or None")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a string")

    @property
    def released(self) -> bool:
        return self.release is not None

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "released": self.released,
            "evidence": self.evidence.to_dict(),
            "decision": self.decision.to_dict(),
            "release": None if self.release is None else self.release.to_dict(),
            "detail": self.detail,
        }


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _required_text(value, name)


def _check_text(value: Any, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")


def _check_bytes(value: Any, name: str) -> None:
    if value is not None and not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes or None")


def _parent_ids(parents: Iterable[str]) -> tuple:
    if isinstance(parents, str):
        raise TypeError("parents must be an iterable of ids, not a bare string")
    try:
        values = tuple(parents)
    except TypeError:
        raise TypeError("parents must be an iterable of artifact ids") from None
    for parent in values:
        if not isinstance(parent, str):
            raise TypeError("every parent id must be a string")
        if not parent.strip():
            raise ValueError("every parent id must be non-empty")
    return values


def _not_released(decision: PromotionDecision) -> str:
    reasons = list(decision.reasons)
    if decision.missing:
        reasons.append("missing " + ", ".join(decision.missing))
    detail = f"not released: {decision.action}"
    if reasons:
        detail += ": " + "; ".join(reasons)
    return detail


def release_evidence(
        evidence: TrustEvidence, *,
        artifact_id: str,
        artifact_hash: Optional[str] = None,
        content_hash: Optional[str] = None,
        key_id: Optional[str] = None,
        secret: Optional[bytes] = None,
        rules: Optional[PromotionRules] = None,
        ledger: Optional[PromotionLedger] = None,
        proof: Optional[ProofEvidence] = None,
        admission: Optional[AdmissionResult] = None,
        parents: Iterable[str] = (),
        run_id: str = "",
        kind: str = "skill",
        metadata: Optional[Mapping[str, Any]] = None,
) -> ReleaseReport:
    """Judge prebuilt evidence and, if accepted, sign and record the release."""
    if not isinstance(evidence, TrustEvidence):
        raise TypeError("evidence must be a TrustEvidence")
    artifact_id = _required_text(artifact_id, "artifact_id")
    artifact_hash = _optional_text(artifact_hash, "artifact_hash")
    content_hash = _optional_text(content_hash, "content_hash")
    _check_text(key_id, "key_id")
    _check_bytes(secret, "secret")
    if rules is not None and not isinstance(rules, PromotionRules):
        raise TypeError("rules must be a PromotionRules or None")
    if ledger is not None and not isinstance(ledger, PromotionLedger):
        raise TypeError("ledger must be a PromotionLedger or None")
    if proof is not None and not isinstance(proof, ProofEvidence):
        raise TypeError("proof must be a ProofEvidence or None")
    if admission is not None and not isinstance(admission, AdmissionResult):
        raise TypeError("admission must be an AdmissionResult or None")
    parent_ids = _parent_ids(parents)
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    kind = _required_text(kind, "kind")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    release_metadata = dict(metadata or {})

    policy = rules if rules is not None else PromotionRules()
    request = evidence.to_promotion_request(
        artifact_id,
        artifact_hash=artifact_hash or content_hash,
        proof=proof,
        admission=admission,
    )
    decision = promote(request, policy)
    if not (decision.action == "promote" and decision.accepted):
        return ReleaseReport(
            artifact_id=artifact_id, evidence=evidence, decision=decision,
            release=None, detail=_not_released(decision))

    if ledger is None:
        raise ValueError(
            "an accepted promotion needs a PromotionLedger to record the release")
    if key_id is None or not key_id.strip():
        raise ValueError(
            "an accepted promotion needs a non-empty key_id to sign the release")
    if not isinstance(secret, (bytes, bytearray)) or not secret:
        raise ValueError(
            "an accepted promotion needs non-empty bytes secret to sign the release")

    release = ledger.record(
        decision,
        artifact_id=artifact_id,
        content_hash=content_hash or evidence.eval_result.content_hash,
        key_id=key_id,
        secret=bytes(secret),
        rules=policy,
        parents=parent_ids,
        run_id=run_id,
        kind=kind,
        metadata=release_metadata,
    )
    detail = (f"released {artifact_id} via {key_id}; "
              f"manifest {release.manifest.manifest_hash()}")
    return ReleaseReport(
        artifact_id=artifact_id, evidence=evidence, decision=decision,
        release=release, detail=detail)


def release_candidate(
        before: Mapping[str, float], after: Mapping[str, float], *,
        artifact_id: str,
        artifact_hash: Optional[str] = None,
        content_hash: Optional[str] = None,
        key_id: Optional[str] = None,
        secret: Optional[bytes] = None,
        rules: Optional[PromotionRules] = None,
        ledger: Optional[PromotionLedger] = None,
        proof: Optional[ProofEvidence] = None,
        admission: Optional[AdmissionResult] = None,
        parents: Iterable[str] = (),
        run_id: str = "",
        kind: str = "skill",
        metadata: Optional[Mapping[str, Any]] = None,
        **evidence_kwargs: Any,
) -> ReleaseReport:
    """Build trust evidence from paired scores, then run the release pipeline."""
    evidence = build_trust_evidence(before, after, **evidence_kwargs)
    return release_evidence(
        evidence,
        artifact_id=artifact_id,
        artifact_hash=artifact_hash,
        content_hash=content_hash,
        key_id=key_id,
        secret=secret,
        rules=rules,
        ledger=ledger,
        proof=proof,
        admission=admission,
        parents=parents,
        run_id=run_id,
        kind=kind,
        metadata=metadata,
    )
