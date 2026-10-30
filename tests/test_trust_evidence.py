"""The evidence builder must connect the independent trust modules."""

import pytest

from agentdescent.conformal import ConformalReport
from agentdescent.goodhart import detect_goodhart
from agentdescent.metamorphic import MetamorphicSuite, invariance
from agentdescent.promotion import PromotionAction, PromotionRules, promote
from agentdescent.trust_evidence import build_trust_evidence
from agentdescent.verifier_ensemble import VerifierEnsemble, VerifierSpec


def _scores(gain=0.5, n=50):
    before = {f"t{i}": 0.5 for i in range(n)}
    after = {f"t{i}": 0.5 + gain for i in range(n)}
    return before, after


def _verifier(passed=True):
    return VerifierEnsemble([
        VerifierSpec("check", lambda _candidate: passed, cost=1),
    ])


def _metamorphic():
    return MetamorphicSuite([
        invariance("same", lambda task: task),
    ])


def test_build_evidence_composes_every_report_and_promotes():
    before, after = _scores()
    evidence = build_trust_evidence(
        before, after,
        verifier=_verifier(), candidate="candidate",
        calibration=([0.9] * 10, [1.0] * 10),
        proxy_gains=[0.1] * 6, oracle_gains=[0.1] * 6,
        metamorphic=_metamorphic(),
        metamorphic_run=lambda task: task, metamorphic_tasks=[1, 2, 3],
        contaminated=False, risk=0.1, approved=True)
    assert evidence.eval_result.passed
    assert evidence.canary.verdict == "promote"
    assert evidence.conformal.decision == "accept"
    assert evidence.sequential.decision == "accept"
    assert evidence.goodhart is not None and evidence.goodhart.aligned
    assert evidence.metamorphic is not None and evidence.metamorphic.ok
    assert evidence.verifier is not None and evidence.verifier.passed
    assert evidence.calibration is not None
    decision = promote(
        evidence.to_promotion_request("skill-1"),
        PromotionRules(require_goodhart=True, require_metamorphic=True,
                       require_verifier=True, require_calibration=True))
    assert decision.action == PromotionAction.PROMOTE
    assert "goodhart" in decision.evidence
    assert "metamorphic" in decision.evidence
    assert "verifier" in decision.evidence
    assert "calibration" in decision.evidence


def test_goodhart_gaming_metamorphic_failure_and_verifier_failure_reject():
    before, after = _scores()
    gaming = build_trust_evidence(
        before, after, proxy_gains=[0.5] * 6, oracle_gains=[0.0] * 6,
        contaminated=False, risk=0.1)
    assert gaming.goodhart is not None and gaming.goodhart.gaming
    decision = promote(gaming.to_promotion_request("a"),
                       PromotionRules(require_goodhart=True))
    assert decision.action == PromotionAction.REJECT

    failing = MetamorphicSuite([invariance("shift", lambda task: task + 1)])
    bad_meta = build_trust_evidence(
        before, after, metamorphic=failing,
        metamorphic_run=lambda task: task, metamorphic_tasks=[1, 2],
        contaminated=False, risk=0.1)
    decision = promote(bad_meta.to_promotion_request("a"),
                       PromotionRules(require_metamorphic=True))
    assert decision.action == PromotionAction.REJECT

    bad_verifier = build_trust_evidence(
        before, after, verifier=_verifier(False), candidate="c",
        contaminated=False, risk=0.1)
    decision = promote(bad_verifier.to_promotion_request("a"),
                       PromotionRules(require_verifier=True))
    assert decision.action == PromotionAction.REJECT


def test_missing_new_evidence_defers_only_when_required():
    before, after = _scores()
    evidence = build_trust_evidence(before, after, contaminated=False, risk=0.1)
    request = evidence.to_promotion_request("a")
    assert promote(request).action == PromotionAction.PROMOTE
    decision = promote(request, PromotionRules(require_goodhart=True))
    assert decision.action == PromotionAction.DEFER
    assert "goodhart" in decision.missing


def test_missing_risk_is_not_silently_zero():
    before, after = _scores()
    evidence = build_trust_evidence(before, after, contaminated=False)
    decision = promote(evidence.to_promotion_request("a"))
    assert decision.action == PromotionAction.DEFER
    assert "risk" in decision.missing


def test_eval_means_use_only_the_paired_tasks():
    before = {f"t{i}": 0.5 for i in range(20)}
    before.update({f"extra{i}": 0.0 for i in range(20)})
    after = {f"t{i}": 0.9 for i in range(20)}
    evidence = build_trust_evidence(before, after, contaminated=False, risk=0.0)
    result = evidence.eval_result
    assert result.n == 20
    assert result.score == pytest.approx(0.9)
    assert result.baseline == pytest.approx(0.5)
    assert result.score - result.baseline == pytest.approx(result.delta)


def test_verifier_without_candidate_is_refused():
    before, after = _scores()
    with pytest.raises(ValueError):
        build_trust_evidence(before, after, verifier=_verifier())


def test_builder_refuses_unpaired_or_partial_inputs():
    before, after = _scores()
    with pytest.raises(ValueError):
        build_trust_evidence({"a": 0.5}, {"b": 0.6})
    with pytest.raises(ValueError):
        build_trust_evidence(before, after, proxy_gains=[0.1])
    with pytest.raises(ValueError):
        build_trust_evidence(before, after, metamorphic=_metamorphic())
    with pytest.raises(ValueError):
        build_trust_evidence(before, after, alpha=1.0)
    with pytest.raises(TypeError):
        build_trust_evidence(before, after, contaminated="yes")
