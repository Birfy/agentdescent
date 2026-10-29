"""Promotion must be one fail-closed decision over independent evidence."""

import json
from types import SimpleNamespace

import pytest

from agentdescent.conformal import ConformalGate
from agentdescent.evalops import EvalResult
from agentdescent.policy_release import (
    AdmissionResult,
    PolicyDecision,
    VerificationResult,
)
from agentdescent.promotion import (
    PromotionAction,
    PromotionRules,
    PromotionRequest,
    ProofEvidence,
    promote,
)
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.safety_controller import canary
from agentdescent.sequential import sequential_test


def _eval(*, passed=True, delta=0.1, n=20):
    return EvalResult(
        eval_id="eval", dataset_id="ds", version=1, content_hash="hash",
        n=n, score=0.7, baseline=0.6, delta=delta, ci=(0.02, 0.18),
        passed=passed, cost=0.0, per_task={}, detail="ok")


def _conformal(decision="accept"):
    return ConformalGate(alpha=0.2, min_samples=1).fit(
        [0.1, 0.2, 0.3, 0.4, 0.5]).decide() if decision == "accept" else (
            ConformalGate(alpha=0.05).fit([0.1, 0.2]).decide())


def _sequential(decision="accept"):
    if decision == "accept":
        return sequential_test([0.5] * 20)
    return sequential_test([0.0] * 4, max_n=4)


def _canary(verdict="promote"):
    incumbent = {f"t{i}": 0.5 for i in range(6)}
    if verdict == "promote":
        return canary(incumbent, {key: 0.6 for key in incumbent})
    if verdict == "rollback":
        return canary(incumbent, {key: 0.3 for key in incumbent})
    return canary(incumbent, {key: 0.5 for key in incumbent})


def _proof(*, source="print(1)"):
    artifact = SimpleNamespace(artifact_id="skill-1", source=source, tests=("t",))
    runner = ProofRunner([
        ProofInvariant("no-network", lambda art: "requests" not in art.source),
    ])
    certificate = runner.prove(artifact, issued_at=100.0)
    return ProofEvidence(certificate, runner.verify(certificate, artifact))


def _admission(*, effect="allow", manifest_hash="", signature_valid=True):
    if effect == "allow":
        decision = PolicyDecision("allow", "ok")
        allowed = True
        detail = "allow"
    elif effect == "require_approval":
        decision = PolicyDecision("require_approval", "approval")
        allowed = False
        detail = "approval"
    else:
        decision = PolicyDecision("deny", "denied")
        allowed = False
        detail = "denied"
    return AdmissionResult(
        decision=decision,
        signature=VerificationResult(signature_valid, "verified"
                                     if signature_valid else "signature mismatch",
                                     "ci"),
        allowed=allowed, detail=detail, manifest_hash=manifest_hash)


def _request(**overrides):
    values = dict(
        artifact_id="skill-1", eval_result=_eval(), contaminated=False,
        canary=_canary(), risk=0.2, approved=True)
    values.update(overrides)
    return PromotionRequest(**values)


def test_all_required_evidence_green_promotes_with_a_stable_hash():
    request = _request()
    decision = promote(request)
    assert decision.action == PromotionAction.PROMOTE
    assert decision.accepted
    assert decision.missing == ()
    assert decision.evidence_hash
    assert json.loads(json.dumps(decision.to_dict()))["artifact_id"] == "skill-1"
    assert promote(_request()).evidence_hash == decision.evidence_hash


def test_missing_evidence_defers_instead_of_passing():
    decision = promote(PromotionRequest("skill-1"))
    assert decision.action == PromotionAction.DEFER
    assert not decision.accepted
    assert set(decision.missing) == {"contamination", "eval", "canary", "risk"}


def test_contaminated_dataset_and_failed_eval_reject():
    assert promote(_request(contaminated=True)).action == PromotionAction.REJECT
    assert promote(_request(contaminated=None)).action == PromotionAction.DEFER
    assert promote(_request(eval_result=_eval(passed=False))).action \
        == PromotionAction.REJECT
    assert promote(_request(eval_result=_eval(delta=0.0)),
                   PromotionRules(min_effect=0.01)).action \
        == PromotionAction.REJECT
    small = promote(_request(eval_result=_eval(n=1)),
                    PromotionRules(min_tasks=5))
    assert small.action == PromotionAction.DEFER


def test_conformal_and_sequential_evidence_fail_closed():
    policy = PromotionRules(require_conformal=True, require_sequential=True)
    assert promote(_request(), policy).action == PromotionAction.DEFER
    insufficient = promote(
        _request(conformal=_conformal("insufficient"),
                 sequential=_sequential()),
        policy)
    assert insufficient.action == PromotionAction.DEFER
    assert "conformal" in " ".join(insufficient.reasons).lower()
    continuing = promote(
        _request(conformal=_conformal(), sequential=sequential_test([0.0])),
        policy)
    assert continuing.action == PromotionAction.DEFER
    assert "sequential" in " ".join(continuing.reasons).lower()


def test_canary_hold_defers_and_rollback_wins_over_other_failures():
    assert promote(_request(canary=_canary("hold"))).action \
        == PromotionAction.DEFER
    decision = promote(_request(canary=_canary("rollback"), contaminated=True))
    assert decision.action == PromotionAction.ROLLBACK
    assert decision.accepted is False


def test_proof_and_admission_are_required_only_when_asked():
    policy = PromotionRules(require_proof=True, require_admission=True)
    missing = promote(_request(), policy)
    assert missing.action == PromotionAction.DEFER
    assert set(missing.missing) == {"proof", "admission"}

    good = promote(_request(proof=_proof(), admission=_admission()), policy)
    assert good.action == PromotionAction.PROMOTE

    bad_proof = _proof(source="import requests")
    rejected = promote(_request(proof=bad_proof, admission=_admission()), policy)
    assert rejected.action == PromotionAction.REJECT

    approval = promote(
        _request(proof=_proof(),
                 admission=_admission(effect="require_approval")),
        policy)
    assert approval.action == PromotionAction.DEFER

    denied = promote(
        _request(proof=_proof(), admission=_admission(effect="deny")),
        policy)
    assert denied.action == PromotionAction.REJECT


def test_risk_and_human_approval_are_explicit_gates():
    assert promote(_request(risk=0.9),
                   PromotionRules(max_risk=0.5)).action \
        == PromotionAction.REJECT
    assert promote(_request(approved=None),
                   PromotionRules(require_approval=True)).action \
        == PromotionAction.DEFER
    assert promote(_request(approved=True),
                   PromotionRules(require_approval=True)).action \
        == PromotionAction.PROMOTE


def test_evidence_must_be_bound_to_the_candidate():
    proof = _proof()
    mismatched = promote(_request(artifact_id="skill-EVIL", proof=proof),
                         PromotionRules(require_proof=True))
    assert mismatched.action == PromotionAction.REJECT
    assert "proof artifact" in " ".join(mismatched.reasons)

    hash_mismatch = promote(
        _request(artifact_hash="00" * 32, proof=proof),
        PromotionRules(require_proof=True))
    assert hash_mismatch.action == PromotionAction.REJECT
    assert "hash" in " ".join(hash_mismatch.reasons)

    matched = promote(
        _request(artifact_hash=proof.certificate.artifact_hash, proof=proof),
        PromotionRules(require_proof=True))
    assert matched.action == PromotionAction.PROMOTE

    admission = _admission(manifest_hash="ff" * 32)
    rejected = promote(
        _request(artifact_hash=proof.certificate.artifact_hash,
                 proof=proof, admission=admission),
        PromotionRules(require_proof=True, require_admission=True))
    assert rejected.action == PromotionAction.REJECT


def test_invalid_admission_signature_rejects_before_approval():
    decision = promote(
        _request(admission=_admission(effect="require_approval",
                                      signature_valid=False)),
        PromotionRules(require_admission=True))
    assert decision.action == PromotionAction.REJECT
    assert "signature" in " ".join(decision.reasons)


def test_decision_hash_covers_the_rules_as_well_as_the_evidence():
    request = _request()
    one = promote(request)
    two = promote(request, PromotionRules(max_risk=0.1))
    assert one.evidence_hash == two.evidence_hash
    assert one.decision_hash != two.decision_hash
    assert promote(request).decision_hash == one.decision_hash


def test_supplied_red_evidence_counts_even_when_not_required():
    policy = PromotionRules(require_canary=False, require_proof=False)
    rollback = promote(_request(canary=_canary("rollback")), policy)
    assert rollback.action == PromotionAction.ROLLBACK
    bad_proof = promote(_request(proof=_proof(source="import requests")), policy)
    assert bad_proof.action == PromotionAction.REJECT


def test_explicit_approval_denial_rejects_and_canary_missing_is_surfaced():
    denied = promote(_request(approved=False),
                     PromotionRules(require_approval=True))
    assert denied.action == PromotionAction.REJECT

    partial = canary({"a": 0.5, "b": 0.5}, {"a": 0.6})
    withheld = promote(_request(canary=partial))
    assert withheld.action == PromotionAction.DEFER
    assert "missing" in " ".join(withheld.reasons)


def test_invalid_records_are_refused():
    with pytest.raises(ValueError):
        PromotionRequest("")
    with pytest.raises(TypeError):
        PromotionRequest("a", contaminated="yes")
    with pytest.raises(ValueError):
        PromotionRequest("a", artifact_hash="")
    with pytest.raises(ValueError):
        PromotionRequest("a", risk=2)
    with pytest.raises(ValueError):
        PromotionRules(min_effect=-1)
    with pytest.raises(ValueError):
        PromotionRules(max_risk=2)
    with pytest.raises(TypeError):
        promote({"artifact_id": "x"})
