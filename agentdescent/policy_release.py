"""Policy-as-code admission and signed release attestations.

A production RSI action needs two answers before it lands:

1. **Policy** -- is this artifact allowed under the org's rules? (risk tier,
   required evidence, canary, approval).
2. **Provenance** -- is this exactly the artifact that was evaluated and
   approved? A hash in a database is not enough if the manifest can be edited.

This module is the small, adapter-friendly seam:

* :class:`PolicyBundle` evaluates ordered rules into
  ``allow / require_approval / deny`` (deny overrides; no match denies);
* :class:`ArtifactManifest` is a canonical, content-hashed release record;
* :class:`SignedEnvelope` uses HMAC-SHA256 here as a dependency-free demo
  signer; a production deployment swaps it for cosign/Sigstore/Ed25519 without
  changing the admission API;
* :class:`AdmissionController` combines policy + signature into one decision.

The policy language is intentionally not invented here. A production adapter
can translate an OPA/Rego or Cedar decision into the same ``PolicyDecision``.

    bundle = PolicyBundle("release", 1, [
        PolicyRule("deny-high-risk", "deny", lambda ctx: ctx["risk"] > 0.7),
        PolicyRule("canary", "require_approval", lambda ctx: ctx["risk"] > 0.3,
                   obligations=("canary",)),
    ])
    controller = AdmissionController(bundle, keyring={"ci": b"..."})
    result = controller.admit(manifest, envelope, {"risk": 0.4})
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PolicyRule",
    "PolicyBundle",
    "PolicyDecision",
    "ArtifactManifest",
    "SignedEnvelope",
    "VerificationResult",
    "AdmissionResult",
    "AdmissionController",
    "sign_manifest",
    "verify_manifest",
    "verify_envelope",
]

Effect = str
Rule = Callable[[Mapping[str, Any]], bool]


def _effect(value: Effect) -> Effect:
    if value not in ("allow", "deny", "require_approval"):
        raise ValueError("effect must be allow, deny or require_approval")
    return value


@dataclass(frozen=True)
class PolicyRule:
    """One ordered policy rule."""

    name: str
    effect: Effect
    when: Rule
    reason: str = ""
    obligations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("PolicyRule.name must be a non-empty string")
        _effect(self.effect)
        if not callable(self.when):
            raise TypeError("PolicyRule.when must be callable")
        if isinstance(self.obligations, str):
            raise ValueError("obligations must be a sequence, not a bare string")


@dataclass(frozen=True)
class PolicyDecision:
    """The final policy decision and the rules that produced it."""

    effect: Effect
    reason: str
    matched: Tuple[str, ...] = ()
    obligations: Tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.effect == "allow"

    @property
    def needs_approval(self) -> bool:
        return self.effect == "require_approval"


class PolicyBundle:
    """An ordered rule set with deny-overrides semantics."""

    def __init__(self, bundle_id: str, version: int,
                 rules: Sequence[PolicyRule]) -> None:
        if not bundle_id or isinstance(version, bool) or not isinstance(version, int) \
                or version < 1:
            raise ValueError("bundle_id and version >= 1 are required")
        self.bundle_id = bundle_id
        self.version = version
        self.rules = tuple(rules)

    def evaluate(self, context: Mapping[str, Any]) -> PolicyDecision:
        matched: list[str] = []
        obligations: set[str] = set()
        deny = None
        approval = None
        allow = None
        for rule in self.rules:
            try:
                applies = bool(rule.when(context))
            except Exception as exc:  # noqa: BLE001 - fail closed on a broken rule
                return PolicyDecision("deny", f"policy error in {rule.name}: {exc}",
                                      (rule.name,))
            if not applies:
                continue
            matched.append(rule.name)
            obligations.update(rule.obligations)
            if rule.effect == "deny" and deny is None:
                deny = rule
            elif rule.effect == "require_approval" and approval is None:
                approval = rule
            elif rule.effect == "allow" and allow is None:
                allow = rule
        if deny is not None:
            return PolicyDecision("deny", deny.reason or f"denied by {deny.name}",
                                  tuple(matched), tuple(sorted(obligations)))
        if approval is not None:
            return PolicyDecision("require_approval",
                                  approval.reason or f"approval required by {approval.name}",
                                  tuple(matched), tuple(sorted(obligations)))
        if allow is not None:
            return PolicyDecision("allow", allow.reason or f"allowed by {allow.name}",
                                  tuple(matched), tuple(sorted(obligations)))
        return PolicyDecision("deny", "no rule matched", (), ())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in canonical payload")
        return value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical payload keys must be strings")
            out[key] = _json_safe(item)
        return out
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    raise TypeError(f"non-JSON value in canonical payload: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ArtifactManifest:
    """The release record that policy and signatures bind to."""

    artifact_id: str
    content_hash: str
    kind: str = "artifact"
    created_at: float = 0.0
    parents: Tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.content_hash or not self.kind:
            raise ValueError("artifact_id, content_hash and kind are required")
        if not math.isfinite(float(self.created_at)):
            raise ValueError("created_at must be finite")
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "metadata", dict(self.metadata))
        # Fail at construction, not at signing, for non-JSON provenance.
        _canonical(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "kind": self.kind,
            "created_at": self.created_at,
            "parents": list(self.parents),
            "evidence": dict(self.evidence),
            "metadata": dict(self.metadata),
        }

    def canonical(self) -> str:
        return _canonical(self.to_dict())

    def manifest_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()


@dataclass(frozen=True)
class SignedEnvelope:
    """A manifest plus its detached signature."""

    manifest: ArtifactManifest
    key_id: str
    signature: str
    algorithm: str = "hmac-sha256"
    signed_at: float = 0.0


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    reason: str = ""
    key_id: str = ""


@dataclass(frozen=True)
class AdmissionResult:
    decision: PolicyDecision
    signature: VerificationResult
    allowed: bool
    obligations: Tuple[str, ...] = ()
    detail: str = ""
    manifest_hash: str = ""
    bundle_id: str = ""
    bundle_version: int = 0
    key_id: str = ""
    context_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect": self.decision.effect, "reason": self.decision.reason,
            "allowed": self.allowed, "obligations": list(self.obligations),
            "manifest_hash": self.manifest_hash,
            "bundle_id": self.bundle_id, "bundle_version": self.bundle_version,
            "key_id": self.key_id, "context_hash": self.context_hash,
            "signature": self.signature.reason, "detail": self.detail,
        }

    @property
    def needs_approval(self) -> bool:
        return self.decision.needs_approval


def _signature(manifest: ArtifactManifest, key_id: str, secret: bytes) -> str:
    message = (key_id + "." + manifest.manifest_hash()).encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def sign_manifest(manifest: ArtifactManifest, *, key_id: str,
                  secret: bytes) -> SignedEnvelope:
    if not key_id or not secret:
        raise ValueError("key_id and secret are required")
    return SignedEnvelope(manifest=manifest, key_id=key_id,
                          signature=_signature(manifest, key_id, secret),
                          signed_at=time.time())


def verify_manifest(envelope: SignedEnvelope,
                    keyring: Mapping[str, bytes]) -> VerificationResult:
    if envelope.algorithm != "hmac-sha256":
        return VerificationResult(False, "unsupported algorithm", envelope.key_id)
    secret = keyring.get(envelope.key_id)
    if not isinstance(secret, (bytes, bytearray)) or not secret:
        return VerificationResult(False, "missing or empty key", envelope.key_id)
    try:
        expected = _signature(envelope.manifest, envelope.key_id, bytes(secret))
        if not hmac.compare_digest(expected, envelope.signature):
            return VerificationResult(False, "signature mismatch", envelope.key_id)
        return VerificationResult(True, "verified", envelope.key_id)
    except Exception as exc:  # noqa: BLE001 - verification fails closed
        return VerificationResult(False, f"verification error: {exc}", envelope.key_id)


verify_envelope = verify_manifest


class AdmissionController:
    """Policy + signature + obligations in one admission decision."""

    def __init__(self, bundle: PolicyBundle, *,
                 keyring: Optional[Mapping[str, bytes]] = None,
                 require_signature: bool = True) -> None:
        self.bundle = bundle
        self.keyring = dict(keyring or {})
        for key_id, secret in self.keyring.items():
            if not isinstance(secret, (bytes, bytearray)) or not secret:
                raise ValueError(f"keyring entry {key_id!r} must be non-empty bytes")
        self.require_signature = require_signature

    def admit(self, manifest: ArtifactManifest,
              envelope: Optional[SignedEnvelope] = None,
              context: Optional[Mapping[str, Any]] = None) -> AdmissionResult:
        context = dict(context or {})
        # The verified manifest is authoritative; a caller cannot override it.
        context["manifest"] = manifest.to_dict()
        if envelope is None:
            signature = VerificationResult(not self.require_signature,
                                           "missing signature" if self.require_signature
                                           else "not required")
        else:
            if envelope.manifest != manifest:
                signature = VerificationResult(False, "envelope/manifest mismatch")
            else:
                signature = verify_manifest(envelope, self.keyring)
        decision = self.bundle.evaluate(context)
        if not signature.valid:
            decision = PolicyDecision("deny", f"signature: {signature.reason}",
                                      decision.matched, decision.obligations)
        allowed = decision.allowed and signature.valid
        detail = (f"{decision.effect}; signature={signature.reason}; "
                  f"matched={','.join(decision.matched) or 'none'}")
        return AdmissionResult(
            decision=decision, signature=signature, allowed=allowed,
            obligations=decision.obligations, detail=detail,
            manifest_hash=manifest.manifest_hash(),
            bundle_id=self.bundle.bundle_id, bundle_version=self.bundle.version,
            key_id=signature.key_id, context_hash=hashlib.sha256(
                _canonical(context).encode()).hexdigest())
