"""Turn an accepted promotion into a signed, lineage-linked release record.

`promote()` answers whether a candidate may ship; `policy_release` can sign an
artifact manifest; `lineage` records where it came from. Nothing wrote the
promotion decision into that release path. `PromotionLedger` does one thing:

    decision (must be promote)
      -> ArtifactManifest carrying evidence_hash / decision_hash / reasons
      -> HMAC-signed SignedEnvelope
      -> verified
      -> LineageNode with parents and run id

A refused promotion never produces a manifest. The ledger is not another policy
engine; it is the durable receipt that makes an accepted promotion auditable.

    ledger = PromotionLedger(LineageGraph())
    release = ledger.record(decision, artifact_id="skill-1",
                            content_hash="abc", key_id="ci", secret=key,
                            parents=("skill-0",), run_id="run-8")
    release.verification.valid, release.manifest.manifest_id
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from .lineage import LineageGraph, LineageNode
from .policy_release import (
    ArtifactManifest,
    SignedEnvelope,
    VerificationResult,
    sign_manifest,
    verify_manifest,
)
from .promotion import (
    PromotionAction,
    PromotionDecision,
    PromotionRules,
    _canonical,
)

__all__ = ["ReleaseRecord", "PromotionLedger"]


@dataclass(frozen=True)
class ReleaseRecord:
    """The signed manifest, its verification and the lineage node."""

    manifest: ArtifactManifest
    envelope: SignedEnvelope
    verification: VerificationResult
    lineage: LineageNode
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest.manifest_hash(),
            "decision_hash": self.manifest.evidence.get("decision_hash", ""),
            "evidence_hash": self.manifest.evidence.get("evidence_hash", ""),
            "kind": self.manifest.kind,
            "created_at": self.manifest.created_at,
            "envelope": {
                "manifest": self.envelope.manifest.to_dict(),
                "key_id": self.envelope.key_id,
                "signature": self.envelope.signature,
                "algorithm": self.envelope.algorithm,
                "signed_at": self.envelope.signed_at,
            },
            "verification": {
                "valid": self.verification.valid,
                "reason": self.verification.reason,
                "key_id": self.verification.key_id,
            },
            "lineage": {
                "artifact_id": self.lineage.artifact_id,
                "parents": list(self.lineage.parent_ids),
                "run_id": self.lineage.run_id,
            },
            "detail": self.detail,
        }


class PromotionLedger:
    """Sign accepted promotions and write their lineage receipt."""

    def __init__(self, lineage: Optional[LineageGraph] = None, *,
                 issuer: str = "agentdescent") -> None:
        if lineage is not None and not isinstance(lineage, LineageGraph):
            raise TypeError("lineage must be a LineageGraph or None")
        if not isinstance(issuer, str) or not issuer.strip():
            raise ValueError("issuer is required")
        self.lineage = lineage if lineage is not None else LineageGraph()
        self.issuer = issuer

    def record(self, decision: PromotionDecision, *,
               artifact_id: str, content_hash: str, key_id: str,
               secret: bytes, rules: PromotionRules,
               parents: Iterable[str] = (),
               run_id: str = "", kind: str = "skill",
               metadata: Optional[Mapping[str, Any]] = None,
               created_at: Optional[float] = None,
               verification_keyring: Optional[Mapping[str, bytes]] = None
               ) -> ReleaseRecord:
        if not isinstance(decision, PromotionDecision):
            raise TypeError("decision must be a PromotionDecision")
        if not isinstance(rules, PromotionRules):
            raise TypeError("rules must be the PromotionRules that produced the decision")
        if decision.action != PromotionAction.PROMOTE or not decision.accepted:
            raise ValueError("only an accepted promotion can be released")
        if decision.artifact_id != str(artifact_id):
            raise ValueError(
                f"decision is for {decision.artifact_id!r}, not {artifact_id!r}")
        expected_hash = decision.evidence.get("artifact_hash")
        if expected_hash is not None and expected_hash != content_hash:
            raise ValueError("content_hash does not match the promotion artifact_hash")
        evidence_hash = hashlib.sha256(
            _canonical(decision.evidence).encode()).hexdigest()
        if evidence_hash != decision.evidence_hash:
            raise ValueError("decision evidence_hash does not match its evidence")
        decision_hash = hashlib.sha256(_canonical({
            "evidence_hash": decision.evidence_hash,
            "policy": asdict(rules),
            "action": decision.action}).encode()).hexdigest()
        if decision_hash != decision.decision_hash:
            raise ValueError("decision_hash does not match the supplied rules")
        if artifact_id in self.lineage:
            raise ValueError(f"artifact {artifact_id!r} already has a lineage node")
        if not isinstance(secret, (bytes, bytearray)) or not secret:
            raise ValueError("secret must be non-empty bytes")
        parent_ids = tuple(str(parent) for parent in parents)
        issued = time.time() if created_at is None else float(created_at)
        manifest = ArtifactManifest(
            artifact_id=str(artifact_id), content_hash=str(content_hash),
            kind=str(kind), created_at=issued, parents=parent_ids,
            evidence={
                "decision_hash": decision.decision_hash,
                "evidence_hash": decision.evidence_hash,
                "action": decision.action,
                "reasons": list(decision.reasons),
                "risk": decision.risk,
            },
            metadata={"issuer": self.issuer, **dict(metadata or {})})
        envelope = sign_manifest(manifest, key_id=key_id, secret=bytes(secret))
        keyring = (dict(verification_keyring) if verification_keyring is not None
                   else {key_id: bytes(secret)})
        verification = verify_manifest(envelope, keyring)
        if not verification.valid:
            raise RuntimeError(
                f"release signature failed: {verification.reason}")
        node = self.lineage.add_record(
            artifact_id=str(artifact_id), parent_ids=parent_ids,
            run_id=str(run_id), score=None, slot="",
            metadata={"manifest_hash": manifest.manifest_hash(),
                      "decision_hash": decision.decision_hash,
                      "evidence_hash": decision.evidence_hash,
                      "key_id": key_id, "issuer": self.issuer})
        detail = (f"{artifact_id} signed by {key_id}; "
                  f"manifest={manifest.manifest_hash()[:12]}")
        return ReleaseRecord(
            manifest=manifest, envelope=envelope, verification=verification,
            lineage=node, detail=detail)
