"""The conformal gate must be finite-sample correct and conservative."""

import math
import random

import pytest

from agentdescent.conformal import (
    ConformalGate,
    conformal_lower_bound,
    conformal_p_value,
)


def test_lower_bound_uses_the_floor_rank():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    # floor((5+1)*0.1) = 0 -> no finite bound at this sample size
    assert conformal_lower_bound(values, 0.1) == float("-inf")
    # floor(6*0.5) = 3 -> third smallest
    assert conformal_lower_bound(values, 0.5) == 2.0
    # floor(6*0.99) = 5 -> largest
    assert conformal_lower_bound(values, 0.99) == 4.0


def test_small_sample_has_no_finite_bound_and_gate_says_so():
    gate = ConformalGate(alpha=0.1).fit([0.1, 0.2, 0.3, 0.4])
    assert gate.threshold() == float("-inf")
    report = gate.decide()
    assert report.decision == "insufficient"
    assert "cannot certify" in report.detail


def test_rank_is_conservative_for_ties():
    # floor(7*0.3)=2; the duplicate at the bottom cannot be over-certified.
    values = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    assert conformal_lower_bound(values, 0.3) == 0.0
    assert conformal_lower_bound(values, 0.1) == float("-inf")


def test_p_value_is_the_smeared_upper_tail_and_only_diagnostic():
    values = [0.0, 1.0, 2.0]
    assert conformal_p_value(values, -1.0) == 1.0
    assert conformal_p_value(values, 2.0) == pytest.approx(2 / 4)
    assert conformal_p_value(values, 3.0) == pytest.approx(1 / 4)
    assert conformal_p_value(values, 2.0) < conformal_p_value(values, 1.0)


def test_observed_value_does_not_gate_the_conformal_decision():
    gate = ConformalGate(alpha=0.25).fit([-0.1, 0.1, 0.2, 0.3])
    # threshold is the floor rank: floor(5*0.25)=1 -> -0.1, so reject.
    assert gate.decide(10.0).decision == "reject"
    gate.fit([0.1, 0.2, 0.3, 0.4])
    # floor(5*.25)=1 -> 0.1 > 0. The p-value of 0.3 is 0.6, but that is only a
    # diagnostic and must not turn a positive lower bound into a rejection.
    report = gate.decide(0.3)
    assert report.decision == "accept"
    assert report.p_value == pytest.approx(0.6)


def test_all_positive_improvements_are_accepted():
    gate = ConformalGate(alpha=0.1).fit([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                        0.8, 0.9, 1.0])
    report = gate.decide()
    assert report.decision == "accept"
    assert report.threshold > 0
    assert report.accepted


def test_a_bad_lower_tail_is_rejected():
    gate = ConformalGate(alpha=0.25).fit([-1.0, -0.5, 0.4, 1.0, 1.1])
    # floor(6*.25)=1 -> -1.0
    assert gate.decide().decision == "reject"


def test_insufficient_calibration_is_explicit():
    gate = ConformalGate(alpha=0.05).fit([0.1] * 18)
    report = gate.decide()
    assert report.decision == "insufficient"
    assert "need at least 19" in report.detail


def test_min_gain_is_a_cost_floor_not_a_fake_win():
    gate = ConformalGate(alpha=0.1, min_samples=5, min_gain=0.2).fit(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert gate.threshold() == pytest.approx(0.1)
    assert gate.decide().decision == "reject"


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, float("nan")])
def test_alpha_must_be_a_probability(alpha):
    with pytest.raises(ValueError):
        ConformalGate(alpha=alpha)


def test_empty_and_non_finite_calibration_are_refused():
    with pytest.raises(ValueError):
        ConformalGate().fit([])
    with pytest.raises(ValueError):
        ConformalGate().fit([0.1, math.nan])
    with pytest.raises(ValueError):
        ConformalGate().threshold()
    with pytest.raises(ValueError):
        conformal_lower_bound([], 0.5)


def test_calibration_below_fraction_is_reported():
    gate = ConformalGate(alpha=0.25, min_samples=1).fit([-1.0, -0.5, 0.0, 1.0])
    report = gate.decide()
    assert 0.0 <= report.calibration_below_fraction <= 1.0
    assert report.n == 4
    assert report.alpha == 0.25


def test_seeded_monte_carlo_coverage_is_at_most_alpha():
    """A new draw falls below the bound no more than alpha of the time."""
    rng = random.Random(7)
    n = 10
    alpha = 0.1
    trials = 5000
    misses = 0
    for _ in range(trials):
        values = [rng.gauss(0.0, 1.0) for _ in range(n)]
        bound = conformal_lower_bound(values, alpha)
        new = rng.gauss(0.0, 1.0)
        if new < bound:
            misses += 1
    rate = misses / trials
    sigma = math.sqrt(alpha * (1 - alpha) / trials)
    assert rate <= alpha + 3 * sigma, (rate, alpha, sigma)
