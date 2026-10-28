"""Deferral must buy review capacity without hiding risky approvals."""

import pytest

from agentdescent.deferral import (
    DeferralConfig,
    DeferralPolicy,
    DeferralSample,
)


def test_fit_finds_the_lowest_safe_confidence_band():
    policy = DeferralPolicy(DeferralConfig(target_risk=0.0, min_coverage=0.4))
    report = policy.fit([
        DeferralSample(0.90, 0.1, True),
        DeferralSample(0.85, 0.1, True),
        DeferralSample(0.40, 0.1, False),
        DeferralSample(0.30, 0.1, False),
        DeferralSample(0.95, 0.1, True),
    ])
    assert policy.threshold == pytest.approx(0.85)
    assert report.coverage == pytest.approx(0.6)
    assert report.auto_risk == 0.0
    assert report.review_rate == pytest.approx(0.4)
    assert policy.decide(0.9, 0.1) == "auto-approve"
    assert policy.decide(0.4, 0.1) == "defer"


def test_unsafe_confident_rows_collapse_to_defer_all():
    policy = DeferralPolicy(DeferralConfig(target_risk=0.0, min_samples=1))
    report = policy.fit([
        DeferralSample(0.9, 0.1, False),
        DeferralSample(0.8, 0.1, True),
        DeferralSample(0.3, 0.1, True),
    ])
    assert policy.threshold == float("inf")
    assert report.verdict in ("defer-all", "infeasible")
    assert report.coverage == 0.0
    assert policy.decide(1.0, 0.1) == "defer"


def test_min_coverage_can_force_a_more_conservative_policy():
    policy = DeferralPolicy(DeferralConfig(target_risk=0.0, min_coverage=0.9, min_samples=1))
    report = policy.fit([
        DeferralSample(0.9, 0.1, True),
        DeferralSample(0.8, 0.1, False),
        DeferralSample(0.3, 0.1, True),
    ])
    assert report.verdict == "infeasible"
    assert report.coverage == 0.0
    assert policy.decide(1.0, 0.1) == "defer"


def test_hard_risk_is_denied_even_when_confidence_is_high():
    policy = DeferralPolicy(DeferralConfig(target_risk=0.1, hard_risk=0.5, min_samples=1))
    policy.fit([
        DeferralSample(0.9, 0.2, True),
        DeferralSample(0.8, 0.2, True),
        DeferralSample(0.4, 0.2, False),
    ])
    assert policy.decide(0.99, 0.9) == "deny"
    decisions, report = policy.route([
        DeferralSample(0.9, 0.2, True),
        DeferralSample(0.9, 0.9, True),
    ])
    assert decisions == ("auto-approve", "deny")
    assert report.deny_rate == 0.5


def test_route_materialises_generators_and_reports_counts():
    policy = DeferralPolicy(DeferralConfig(target_risk=0.0, min_samples=1))
    policy.fit([
        DeferralSample(0.9, 0.1, True),
        DeferralSample(0.8, 0.1, True),
        DeferralSample(0.3, 0.1, False),
    ])
    decisions, report = policy.route(
        DeferralSample(c, 0.1, c >= 0.8) for c in (0.9, 0.8, 0.3))
    assert decisions == ("auto-approve", "auto-approve", "defer")
    assert report.n == 3
    assert report.review_rate == pytest.approx(1 / 3)
    assert report.detail.startswith("2 auto")


def test_unfitted_policy_refuses_to_route():
    policy = DeferralPolicy()
    with pytest.raises(RuntimeError):
        policy.decide(0.9, 0.1)
    with pytest.raises(RuntimeError):
        policy.route([])


def test_invalid_samples_and_config_are_refused():
    with pytest.raises(ValueError):
        DeferralSample(1.2, 0.1, True)
    with pytest.raises(ValueError):
        DeferralSample(0.5, -0.1, True)
    for kwargs in ({"target_risk": -0.1}, {"target_risk": 1.2},
                   {"min_coverage": 1.5}, {"hard_risk": 2}):
        with pytest.raises(ValueError):
            DeferralConfig(**kwargs)
    with pytest.raises(ValueError):
        DeferralPolicy().fit([])


def test_unsafe_confidence_one_can_never_auto_approve():
    policy = DeferralPolicy(DeferralConfig(target_risk=0.0, min_samples=1))
    report = policy.fit([
        DeferralSample(1.0, 0.1, False),
        DeferralSample(1.0, 0.1, False),
        DeferralSample(0.5, 0.1, True),
    ])
    assert policy.threshold == float("inf")
    assert report.coverage == 0.0
    assert policy.decide(1.0, 0.1) == "defer"


def test_small_sample_is_insufficient_and_defers():
    policy = DeferralPolicy()
    report = policy.fit([DeferralSample(0.05, 0.1, True)])
    assert report.verdict == "insufficient"
    assert policy.decide(0.05, 0.1) == "defer"


def test_invalid_safe_flag_and_bool_confidence_are_refused():
    with pytest.raises(ValueError):
        DeferralSample(0.5, 0.1, "yes")
    with pytest.raises(ValueError):
        DeferralSample(True, 0.1, True)
