"""Policy and provenance must agree before an artifact is released."""

import pytest

from agentdescent.policy_release import (
    AdmissionController,
    ArtifactManifest,
    PolicyBundle,
    PolicyDecision,
    PolicyRule,
    SignedEnvelope,
    sign_manifest,
    verify_manifest,
)


def _manifest(**kwargs):
    return ArtifactManifest("a1", "sha256:abc", kind="skill",
                            created_at=100.0, **kwargs)


def _bundle():
    return PolicyBundle("release", 1, [
        PolicyRule("deny-high", "deny", lambda ctx: ctx.get("risk", 0) > 0.7,
                   reason="too risky"),
        PolicyRule("canary", "require_approval",
                   lambda ctx: ctx.get("risk", 0) > 0.3,
                   obligations=("canary",)),
        PolicyRule("evidence", "allow",
                   lambda ctx: bool(ctx.get("evidence", True))),
    ])


def test_policy_deny_overrides_allow_and_default_is_deny():
    assert _bundle().evaluate({"risk": 0.9}).effect == "deny"
    approval = _bundle().evaluate({"risk": 0.4})
    assert approval.effect == "require_approval"
    assert approval.obligations == ("canary",)
    assert _bundle().evaluate({"risk": 0.1}).effect == "allow"
    empty = PolicyBundle("empty", 1, []).evaluate({})
    assert empty.effect == "deny" and empty.reason == "no rule matched"


def test_a_broken_rule_fails_closed():
    bundle = PolicyBundle("x", 1, [
        PolicyRule("boom", "allow", lambda _ctx: 1 / 0),
    ])
    assert bundle.evaluate({}).effect == "deny"


def test_manifest_hash_is_canonical_and_tamper_evident():
    one = _manifest(evidence={"b": 2, "a": 1})
    two = ArtifactManifest("a1", "sha256:abc", kind="skill", created_at=100.0,
                           evidence={"a": 1, "b": 2})
    assert one.canonical() == two.canonical()
    assert one.manifest_hash() == two.manifest_hash()
    changed = ArtifactManifest("a1", "sha256:def", kind="skill", created_at=100.0)
    assert changed.manifest_hash() != one.manifest_hash()


def test_sign_and_verify_detects_wrong_key_tamper_and_unknown_key():
    manifest = _manifest()
    envelope = sign_manifest(manifest, key_id="ci", secret=b"secret")
    assert verify_manifest(envelope, {"ci": b"secret"}).valid
    assert not verify_manifest(envelope, {"ci": b"other"}).valid
    assert not verify_manifest(envelope, {"other": b"secret"}).valid
    tampered = SignedEnvelope(manifest, envelope.key_id, "00" * 32)
    assert not verify_manifest(tampered, {"ci": b"secret"}).valid
    unsupported = SignedEnvelope(manifest, envelope.key_id, envelope.signature,
                                 algorithm="ed25519")
    assert not verify_manifest(unsupported, {"ci": b"secret"}).valid


def test_admission_requires_a_valid_signature_when_configured():
    manifest = _manifest()
    controller = AdmissionController(_bundle(), keyring={"ci": b"secret"})
    envelope = sign_manifest(manifest, key_id="ci", secret=b"secret")
    assert controller.admit(manifest, envelope, {"risk": 0.1}).allowed
    assert not controller.admit(manifest, None, {"risk": 0.1}).allowed
    tampered = ArtifactManifest("a1", "sha256:evil", kind="skill")
    result = controller.admit(tampered, envelope, {"risk": 0.1})
    assert not result.allowed
    assert "signature" in result.decision.reason


def test_approval_obligation_survives_admission():
    manifest = _manifest()
    controller = AdmissionController(_bundle(), keyring={"ci": b"secret"})
    envelope = sign_manifest(manifest, key_id="ci", secret=b"secret")
    result = controller.admit(manifest, envelope, {"risk": 0.4})
    assert result.allowed is False
    assert result.needs_approval
    assert result.obligations == ("canary",)


def test_signature_can_be_optional_for_development():
    manifest = _manifest()
    controller = AdmissionController(_bundle(), require_signature=False)
    assert controller.admit(manifest, None, {"risk": 0.1}).allowed


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        PolicyRule("", "allow", lambda _ctx: True)
    with pytest.raises(ValueError):
        PolicyRule("x", "maybe", lambda _ctx: True)
    with pytest.raises(TypeError):
        PolicyRule("x", "allow", None)
    with pytest.raises(ValueError):
        PolicyBundle("", 1, [])
    with pytest.raises(ValueError):
        PolicyBundle("x", 0, [])
    with pytest.raises(ValueError):
        ArtifactManifest("", "h", "skill")
    with pytest.raises(ValueError):
        ArtifactManifest("a", "h", "skill", created_at=float("nan"))
    with pytest.raises(ValueError):
        sign_manifest(_manifest(), key_id="", secret=b"s")
    with pytest.raises(ValueError):
        sign_manifest(_manifest(), key_id="k", secret=b"")


def test_context_cannot_override_the_signed_manifest():
    manifest = _manifest(evidence={"tests": "fail"})
    bundle = PolicyBundle("x", 1, [PolicyRule(
        "evidence", "allow",
        lambda ctx: ctx["manifest"]["evidence"]["tests"] == "pass")])
    controller = AdmissionController(bundle, keyring={"ci": b"secret"})
    envelope = sign_manifest(manifest, key_id="ci", secret=b"secret")
    result = controller.admit(manifest, envelope, {
        "risk": 0.1, "manifest": {"evidence": {"tests": "pass"}}})
    assert not result.allowed


def test_admission_result_carries_manifest_policy_and_context_identity():
    manifest = _manifest(evidence={"tests": "pass"})
    controller = AdmissionController(_bundle(), keyring={"ci": b"secret"})
    envelope = sign_manifest(manifest, key_id="ci", secret=b"secret")
    result = controller.admit(manifest, envelope, {"risk": 0.1})
    assert result.allowed
    assert result.manifest_hash == manifest.manifest_hash()
    assert result.bundle_id == "release" and result.bundle_version == 1
    assert result.key_id == "ci" and len(result.context_hash) == 64
    assert result.to_dict()["manifest_hash"] == result.manifest_hash


def test_empty_or_string_keyring_entries_fail_closed():
    with pytest.raises(ValueError):
        AdmissionController(_bundle(), keyring={"ci": b""})
    with pytest.raises(ValueError):
        AdmissionController(_bundle(), keyring={"ci": "secret"})
    envelope = sign_manifest(_manifest(), key_id="ci", secret=b"secret")
    assert not verify_manifest(envelope, {"ci": b""}).valid
    assert not verify_manifest(envelope, {"ci": "secret"}).valid


def test_provenance_rejects_non_json_and_non_finite_values():
    with pytest.raises((TypeError, ValueError)):
        ArtifactManifest("a", "h", "skill", evidence={"x": object()})
    with pytest.raises((TypeError, ValueError)):
        ArtifactManifest("a", "h", "skill", evidence={"x": float("nan")})
    with pytest.raises((TypeError, ValueError)):
        ArtifactManifest("a", "h", "skill", evidence={"x": (1, 2)})
