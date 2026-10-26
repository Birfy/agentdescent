"""Stopping has to reflect the recent reward/call slope, not a round count."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentdescent import MarginalGainStop


@dataclass
class _Info:
    round: int
    held_out_reward: float
    calls: int


def _feed(stop, rows):
    return [stop(_Info(i, reward, calls)) for i, (reward, calls) in enumerate(rows)]


def test_it_stops_after_a_flat_reward_per_call_streak():
    stop = MarginalGainStop(min_gain_per_call=0.01, patience=2, window=2)
    decisions = _feed(stop, [(0.2, 10), (0.2, 20), (0.2, 30)])
    assert decisions == [False, False, True]
    assert "per call" in stop.reason


def test_a_rising_reward_never_stops():
    stop = MarginalGainStop(min_gain_per_call=0.01, patience=1, window=2)
    decisions = _feed(stop, [(0.1, 10), (0.3, 20), (0.6, 30)])
    assert not any(decisions)


def test_a_single_good_measurement_resets_the_streak():
    stop = MarginalGainStop(min_gain_per_call=0.01, patience=2, window=2)
    decisions = _feed(stop, [(0.2, 10), (0.2, 20), (0.5, 30),
                             (0.5, 40), (0.5, 50), (0.5, 60)])
    assert decisions == [False, False, False, False, False, True]


def test_a_round_without_new_calls_does_not_establish_a_slope():
    stop = MarginalGainStop(patience=1)
    assert stop(_Info(0, 0.5, 10)) is False
    assert stop(_Info(1, 0.5, 10)) is False
    assert "no additional calls" in stop.reason


def test_a_small_positive_gain_under_the_per_call_bar_stops():
    stop = MarginalGainStop(min_gain_per_call=1e-2, patience=1)
    decisions = _feed(stop, [(0.5 + 1e-6 * k, 10 * (k + 1))
                             for k in range(8)])
    assert decisions[-1] is True
    assert "per call" in stop.reason


def test_min_gain_is_an_optional_absolute_floor():
    stop = MarginalGainStop(min_gain=0.05, min_gain_per_call=1e-9, patience=1)
    decisions = _feed(stop, [(0.5, 10), (0.51, 20)])
    assert decisions[-1] is True


def test_reset_makes_the_policy_reusable():
    stop = MarginalGainStop(min_gain_per_call=0.01, patience=1, window=2)
    assert _feed(stop, [(0.2, 10), (0.2, 20)])[-1] is True
    stop.reset()
    assert _feed(stop, [(0.2, 10), (0.5, 20)]) == [False, False]


def test_parameters_are_validated():
    with pytest.raises(ValueError, match="patience"):
        MarginalGainStop(patience=0)
    with pytest.raises(ValueError, match="window"):
        MarginalGainStop(window=0)
    with pytest.raises(ValueError, match="min_gain"):
        MarginalGainStop(min_gain=float("nan"))
    with pytest.raises(ValueError, match="min_gain_per_call"):
        MarginalGainStop(min_gain_per_call=float("inf"))
