"""Calibration must happen before the deferral threshold is learned."""

import pytest

from agentdescent.calibrated_deferral import CalibratedDeferral
from agentdescent.deferral import DeferralConfig, DeferralPolicy, DeferralSample
from agentdescent.self_model import SelfModel


def _router():
    return CalibratedDeferral(
        SelfModel(),
        DeferralPolicy(DeferralConfig(target_risk=0.15, min_samples=5)))


def _rows():
    confidences = [0.9] * 10 + [0.5] * 10
    outcomes = [1.0] * 9 + [0.0] + [1.0] * 8 + [0.0] * 2
    risks = [0.1] * 20
    safe = [value == 1.0 for value in outcomes]
    return confidences, outcomes, risks, safe


def test_fit_calibrates_then_learns_the_deferral_region():
    router = _router()
    report = router.fit(*_rows())
    assert report.calibration.n == 20
    assert report.deferral.verdict == "auto-all"
    assert router.fitted
    assert report.to_dict()["deferral"]["coverage"] == pytest.approx(1.0)


def test_decide_uses_calibrated_confidence_and_hard_risk():
    router = _router()
    router.fit(*_rows())
    router.policy.threshold = 0.85
    high = router.decide(0.9, 0.1)
    assert high.action == "auto-approve"
    low = router.decide(0.5, 0.1)
    assert low.action == "defer"
    denied = router.decide(0.9, 0.9)
    assert denied.action == "deny"
    assert high.raw_confidence == 0.9
    assert high.calibrated_confidence == pytest.approx(high.raw_confidence, abs=1e-4)
    routed = router.route([(0.9, 0.1), (0.5, 0.1)])
    assert [row.action for row in routed] == ["auto-approve", "defer"]


def test_calibration_changes_the_routing_decision():
    confidences = [0.9] * 20 + [0.95] * 20 + [0.5] * 20
    outcomes = [1.0] * 20 + [0.0] * 40
    risks = [0.1] * 60
    safe = [value == 1.0 for value in outcomes]
    config = DeferralConfig(target_risk=0.5, min_samples=5)
    router = CalibratedDeferral(SelfModel(), DeferralPolicy(config))
    report = router.fit(confidences, outcomes, risks, safe)
    assert report.fitted
    raw_policy = DeferralPolicy(config)
    raw_policy.fit(DeferralSample(c, r, s)
                   for c, r, s in zip(confidences, risks, safe))
    assert raw_policy.decide(0.9, 0.1) == "auto-approve"
    decision = router.decide(0.9, 0.1)
    assert decision.calibrated_confidence < decision.raw_confidence
    assert decision.action == "defer"
    assert report.raw_calibration.verdict != report.calibration.verdict


def test_unfitted_and_invalid_inputs_are_refused():
    router = _router()
    with pytest.raises(RuntimeError):
        router.decide(0.9, 0.1)
    with pytest.raises(RuntimeError):
        router.route([(0.9, 0.1)])
    with pytest.raises(ValueError):
        router.fit([0.9], [1.0, 0.0], [0.1], [True])
    with pytest.raises(ValueError):
        router.fit([0.9], [0.5], [0.1], [True])
    with pytest.raises(ValueError):
        router.fit([0.9], [1.0], [2.0], [True])
    with pytest.raises(TypeError):
        router.fit([0.9], [1.0], [0.1], ["yes"])
    with pytest.raises(TypeError):
        CalibratedDeferral(object())
