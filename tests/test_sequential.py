"""An e-process must stay near 1 under the null and cross 1/alpha under a win."""

import math

import pytest

from agentdescent.sequential import EProcess, sequential_test


def test_zero_improvement_does_not_manufacture_evidence():
    report = sequential_test([0.0] * 200)
    assert report.e_value == pytest.approx(1.0)
    assert report.decision == "continue"
    assert report.p_value == 1.0


def test_strong_positive_streams_accept_anytime():
    report = sequential_test([1.0] * 100)
    assert report.decision == "accept"
    assert report.e_value >= 20
    assert report.p_value <= 0.05
    assert report.n < 100


def test_negative_stream_never_accepts_and_max_n_can_close_it():
    report = sequential_test([-0.5] * 50, max_n=20)
    assert report.decision == "reject"
    assert report.n == 20
    assert not report.accepted


def test_decision_is_reproducible_and_stateful():
    stream = EProcess(alpha=0.05)
    first = stream.update(0.5)
    second = stream.update(0.5)
    assert second.n == 2
    assert second.e_value >= first.e_value
    stream.reset()
    assert stream.n == 0
    assert stream.e_value == pytest.approx(1.0)


def test_log_space_does_not_overflow_on_a_long_stream():
    stream = EProcess(alpha=0.05)
    for _ in range(5000):
        report = stream.update(1.0)
    assert math.isfinite(report.log_e)
    assert report.e_value > 1 / 0.05


def test_deltas_are_clipped_by_the_bounded_mapping():
    low = sequential_test([-50.0] * 10).e_value
    high = sequential_test([50.0] * 10).e_value
    assert high > low
    assert math.isfinite(low) and math.isfinite(high)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, float("nan")])
def test_alpha_must_be_a_probability(alpha):
    with pytest.raises(ValueError):
        EProcess(alpha=alpha)


@pytest.mark.parametrize("grid", [[], [0.0], [0.5], [0.6], [-0.1]])
def test_betting_grid_must_be_inside_the_safe_half_open_interval(grid):
    with pytest.raises(ValueError):
        EProcess(betting_grid=grid)


def test_non_finite_delta_is_refused():
    with pytest.raises(ValueError):
        EProcess().update(float("nan"))
    with pytest.raises(ValueError):
        EProcess().update(float("inf"))


def test_max_n_must_be_positive():
    with pytest.raises(ValueError):
        EProcess(max_n=0)


def test_empty_stream_reports_continue_not_accept():
    report = EProcess().report()
    assert report.n == 0
    assert report.decision == "continue"
    assert report.mean == 0.0
