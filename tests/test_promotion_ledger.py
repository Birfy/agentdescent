"""An accepted promotion must become a signed, lineage-linked release."""

import pytest

from agentdescent.evalops import EvalResult
from agentdescent.lineage import LineageGraph
from agentdescent.policy_release import (
    ArtifactManifest, SignedEnvelope, verify_manifest,
)
from agentdescent.promotion import (
    PromotionDecision, PromotionRequest, PromotionRules, promote,
)
from agentdescent.promotion_ledger import PromotionLedger
from agentdescent.safety_controller import canary


def _decision():
    incumbent = {f"t{i}": 0.5 for i in range(6)}
    candidate = {key: 0.7 for key in incumbent}
    result = EvalResult(
        eval_id="e", dataset_id="d", version=1, content_hash="h", n=20,
        score=0.7, baseline=0.5, delta=0.2, ci=(0.1, 0.3), passed=True,
        cost=0.0, per_task={}, detail="ok")
    rules = PromotionRules()
    decision = promote(PromotionRequest(
        artifact_id="skill-1", artifact_hash="content", eval_result=result,
        contaminated=False, canary=canary(incumbent, candidate), risk=0.2,
        approved=True), rules)
    assert decision.action == "promote"
    return decision, rules


def _release(ledger, decision, rules, **overrides):
    values = dict(
        artifact_id="skill-1", content_hash="content", key_id="ci",
        secret=b"secret", rules=rules, parents=("skill-0",), run_id="run-8",
        created_at=100.0)
    values.update(overrides)
    return ledger.record(decision, **values)


def test_promotion_is_signed_and_written_to_lineage():
    lineage = LineageGraph()
    ledger = PromotionLedger(lineage, issuer="ci")
    decision, rules = _decision()
    release = _release(ledger, decision, rules)
    assert release.verification.valid
    assert release.manifest.evidence["decision_hash"] == decision.decision_hash
    assert release.manifest.evidence["evidence_hash"] == decision.evidence_hash
    assert release.manifest.metadata["issuer"] == "ci"
    assert verify_manifest(release.envelope, {"ci": b"secret"}).valid
    assert lineage.parents("skill-1") == ("skill-0",)
    assert lineage["skill-1"].run_id == "run-8"
    data = release.to_dict()
    assert data["verification"]["valid"] is True
    assert data["decision_hash"] == decision.decision_hash
    envelope = data["envelope"]
    rebuilt = SignedEnvelope(
        ArtifactManifest(**envelope["manifest"]), envelope["key_id"],
        envelope["signature"], envelope["algorithm"], envelope["signed_at"])
    assert verify_manifest(rebuilt, {"ci": b"secret"}).valid


def test_binding_and_tamper_checks_refuse_release():
    ledger = PromotionLedger()
    decision, rules = _decision()
    with pytest.raises(ValueError):
        _release(ledger, decision, rules, artifact_id="other")
    with pytest.raises(ValueError):
        _release(ledger, decision, rules, content_hash="other")
    tampered = PromotionDecision(
        artifact_id=decision.artifact_id, action=decision.action,
        accepted=decision.accepted, reasons=decision.reasons,
        evidence=decision.evidence, evidence_hash="a" * 64,
        decision_hash=decision.decision_hash, risk=decision.risk)
    with pytest.raises(ValueError):
        _release(ledger, tampered, rules)
    with pytest.raises(ValueError):
        _release(ledger, decision, PromotionRules(require_eval=False))
    _release(ledger, decision, rules)
    with pytest.raises(ValueError):
        _release(ledger, decision, rules)


def test_refused_promotions_and_invalid_inputs_are_not_released():
    ledger = PromotionLedger()
    decision, rules = _decision()
    rejected = PromotionDecision(
        artifact_id="x", action="reject", accepted=False,
        evidence_hash="a" * 64, decision_hash="b" * 64)
    with pytest.raises(ValueError):
        _release(ledger, rejected, rules, artifact_id="x")
    with pytest.raises(ValueError):
        _release(ledger, decision, rules, secret=b"")
    with pytest.raises(TypeError):
        _release(ledger, object(), rules)
    with pytest.raises(TypeError):
        _release(ledger, decision, object())
    with pytest.raises(TypeError):
        PromotionLedger(object())
    with pytest.raises(ValueError):
        PromotionLedger(issuer="")
