"""Proof-carrying artifacts: ship the invariant evidence with the change.

A safety gate that only returns a boolean is not auditable. A production
artifact should carry a **certificate**: the invariant suite it was checked
against, which obligations passed, the artifact hash, the issuer and a digest.
Later, a reviewer or an admission controller can verify the certificate without
re-running the whole world.

This is a practical proof-carrying slice, not a theorem prover:

* a `ProofInvariant` is a named callable check with an error/warning severity;
* `ProofRunner.prove` executes the suite, records exceptions as failures, and
  emits a `ProofCertificate`;
* `ProofRunner.verify` / `ProofCarryingArtifact.verify` check the certificate
  digest and artifact hash, and that every error-severity obligation passed;
* the certificate is JSON-serialisable for `manifest` / `lineage` / policy.

The digest is a hash commitment for tamper detection, not an issuer signature:
it does not prove who ran the checks. Add a signature layer if authenticity is
required. `ProofInvariant` is deliberately distinct from the governance type
`agentdescent.invariants.Invariant`.

The callable can wrap Hypothesis, a symbolic executor, an SMT check or a simple
assertion; the certificate contract is the same.

    runner = ProofRunner([
        ProofInvariant("no-network", lambda art: "requests" not in art.source),
        ProofInvariant("has-tests", lambda art: bool(art.tests), severity="warning"),
    ])
    certificate = runner.prove(artifact, artifact_id="skill-1")
    runner.verify(certificate, artifact)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ProofInvariant",
    "InvariantResult",
    "ProofCertificate",
    "ProofVerification",
    "ProofRunner",
    "ProofCarryingArtifact",
]

InvariantCheck = Callable[[Any], bool]


def _artifact_hash(artifact: Any) -> str:
    try:
        body = json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False, default=_reject)
    except (TypeError, ValueError):
        body = repr(artifact)
    return hashlib.sha256(body.encode()).hexdigest()


def _reject(value: Any) -> Any:
    raise TypeError(f"non-JSON artifact value: {type(value).__name__}")


@dataclass(frozen=True)
class ProofInvariant:
    name: str
    check: InvariantCheck
    description: str = ""
    severity: str = "error"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ProofInvariant.name is required")
        if not callable(self.check):
            raise TypeError("ProofInvariant.check must be callable")
        if self.severity not in ("error", "warning"):
            raise ValueError("severity must be error or warning")


@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    severity: str
    reason: str = ""
    description: str = ""


@dataclass(frozen=True)
class ProofCertificate:
    artifact_id: str
    artifact_hash: str
    passed: bool
    results: Tuple[InvariantResult, ...]
    issuer: str
    issued_at: float
    digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "passed": self.passed,
            "issuer": self.issuer,
            "issued_at": self.issued_at,
            "digest": self.digest,
            "results": [
                {"name": r.name, "passed": r.passed, "severity": r.severity,
                 "reason": r.reason, "description": r.description}
                for r in self.results
            ],
        }


@dataclass(frozen=True)
class ProofVerification:
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class ProofCarryingArtifact:
    artifact_id: str
    artifact_hash: str
    certificate: ProofCertificate

    def verify(self, artifact: Any) -> ProofVerification:
        if self.artifact_id != self.certificate.artifact_id:
            return ProofVerification(False, "artifact id mismatch")
        if self.artifact_hash != self.certificate.artifact_hash:
            return ProofVerification(False, "artifact hash does not match certificate")
        return _verify_certificate(self.certificate, artifact)


class ProofRunner:
    """Run an invariant suite and issue/verify certificates."""

    def __init__(self, invariants: Sequence[Invariant], *,
                 issuer: str = "agentdescent") -> None:
        if not invariants:
            raise ValueError("ProofRunner needs at least one invariant")
        names = [invariant.name for invariant in invariants]
        if len(set(names)) != len(names):
            raise ValueError("invariant names must be unique")
        if not issuer:
            raise ValueError("issuer is required")
        self.invariants = tuple(invariants)
        self.issuer = issuer

    def prove(self, artifact: Any, *, artifact_id: str = "",
              issued_at: Optional[float] = None) -> ProofCertificate:
        if not artifact_id:
            artifact_id = str(getattr(artifact, "artifact_id", "") or "artifact")
        results: List[InvariantResult] = []
        for invariant in self.invariants:
            try:
                passed = bool(invariant.check(artifact))
                reason = "" if passed else "invariant returned false"
            except Exception as exc:  # noqa: BLE001 - a crashing check is a fail
                passed = False
                reason = f"{type(exc).__name__}: {exc}"
            results.append(InvariantResult(invariant.name, passed,
                                           invariant.severity, reason,
                                           invariant.description))
        failed_errors = [r for r in results
                         if not r.passed and r.severity == "error"]
        passed = not failed_errors
        issued = time.time() if issued_at is None else float(issued_at)
        if not math.isfinite(issued):
            raise ValueError("issued_at must be finite")
        artifact_hash = _artifact_hash(artifact)
        body = json.dumps({
            "artifact_id": artifact_id,
            "artifact_hash": artifact_hash,
            "passed": passed,
            "issuer": self.issuer,
            "issued_at": issued,
            "results": [
                {"name": r.name, "passed": r.passed, "severity": r.severity,
                 "reason": r.reason, "description": r.description}
                for r in results
            ],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(body.encode()).hexdigest()
        return ProofCertificate(artifact_id, artifact_hash, passed,
                                tuple(results), self.issuer, issued, digest)

    def verify(self, certificate: ProofCertificate, artifact: Any) -> ProofVerification:
        return _verify_certificate(certificate, artifact)


def _certificate_body(certificate: ProofCertificate) -> Dict[str, Any]:
    return {
        "artifact_id": certificate.artifact_id,
        "artifact_hash": certificate.artifact_hash,
        "passed": certificate.passed,
        "issuer": certificate.issuer,
        "issued_at": certificate.issued_at,
        "results": [
            {"name": r.name, "passed": r.passed, "severity": r.severity,
             "reason": r.reason, "description": r.description}
            for r in certificate.results
        ],
    }


def _verify_certificate(certificate: ProofCertificate,
                        artifact: Any) -> ProofVerification:
    if not isinstance(certificate, ProofCertificate):
        raise TypeError("verify expects ProofCertificate")
    if _artifact_hash(artifact) != certificate.artifact_hash:
        return ProofVerification(False, "artifact hash mismatch")
    expected = hashlib.sha256(json.dumps(
        _certificate_body(certificate), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if expected != certificate.digest:
        return ProofVerification(False, "certificate digest mismatch")
    failed = [r.name for r in certificate.results
              if not r.passed and r.severity == "error"]
    if failed:
        return ProofVerification(False, "failed obligations: " + ",".join(failed))
    if not certificate.passed:
        return ProofVerification(False, "certificate is marked failed")
    return ProofVerification(True, "verified")
