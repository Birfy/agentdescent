"""The controller must stop for the right reason and record each session."""

from __future__ import annotations

import pytest

from agentdescent import (ControllerConfig, ControllerReport, SessionOutcome,
                          run_controller)


class _Log:
    def __init__(self):
        self.rows = []

    def record(self, **kwargs):
        self.rows.append(kwargs)


class _Switch:
    def __init__(self, engaged=False, reason=""):
        self._engaged, self._reason = engaged, reason

    def engaged(self):
        return self._engaged

    def reason(self):
        return self._reason


def test_it_stops_after_patience_and_keeps_the_best():
    rewards = iter([0.2, 0.8, 0.8, 0.8])
    log = _Log()
    report = run_controller(
        lambda i, seed: SessionOutcome(next(rewards), calls=10),
        ControllerConfig(max_sessions=4, patience=1, decision_log=log))
    assert isinstance(report, ControllerReport)
    assert report.stopped_reason == "patience"
    assert report.best_reward == 0.8 and len(report.sessions) == 3
    assert [s.improved for s in report.sessions] == [None, True, False]
    assert len(log.rows) == 3 and log.rows[1]["evidence"]["improved"] is True


def test_the_kill_switch_stops_before_a_session():
    switch = _Switch(True, "cost spike")
    calls = []
    report = run_controller(lambda i, seed: calls.append(i) or SessionOutcome(1.0),
                            ControllerConfig(kill_switch=switch))
    assert report.stopped_reason == "kill-switch" and report.sessions == ()
    assert calls == []


def test_the_call_budget_stops_the_loop():
    report = run_controller(lambda i, seed: SessionOutcome(0.5, calls=60),
                            ControllerConfig(max_sessions=5, max_calls=100))
    assert report.stopped_reason == "budget" and report.total_calls == 120


def test_an_error_stops_and_is_recorded():
    report = run_controller(
        lambda i, seed: SessionOutcome(0.9, calls=5) if i == 0
        else SessionOutcome(0.0, 5, "boom"),
        ControllerConfig(max_sessions=3, patience=2))
    assert report.stopped_reason == "error"
    assert report.sessions[-1].error == "boom"
    assert report.best_reward == 0.9


def test_a_raised_session_is_an_error_record():
    def boom(i, seed):
        raise RuntimeError("upstream down")
    report = run_controller(boom, ControllerConfig(max_sessions=2))
    assert report.stopped_reason == "error"
    assert "RuntimeError: upstream down" in report.sessions[0].error


def test_config_is_validated():
    with pytest.raises(ValueError, match="max_sessions"):
        ControllerConfig(max_sessions=0)
    with pytest.raises(ValueError, match="max_calls"):
        ControllerConfig(max_calls=0)
    with pytest.raises(ValueError, match="patience"):
        ControllerConfig(patience=0)


def test_an_errored_session_cannot_become_the_best():
    report = run_controller(
        lambda i, seed: SessionOutcome(0.5, 5) if i == 0
        else SessionOutcome(9.9, 1, "judge crashed"),
        ControllerConfig(max_sessions=2, patience=5))
    assert report.best_reward == 0.5
    assert report.sessions[1].improved is False
    assert report.stopped_reason == "error"


def test_a_raised_first_session_has_no_best():
    def boom(i, seed):
        raise RuntimeError("down")
    report = run_controller(boom, ControllerConfig(max_sessions=1))
    assert report.best_reward is None and report.sessions[0].improved is False


def test_nan_reward_and_bad_calls_are_errors():
    with pytest.raises(ValueError, match="finite"):
        SessionOutcome(float("nan"))
    with pytest.raises(ValueError, match="calls"):
        SessionOutcome(0.5, calls=-1)
    with pytest.raises(ValueError, match="calls"):
        SessionOutcome(0.5, calls=True)


def test_empty_error_string_still_stops():
    report = run_controller(lambda i, seed: SessionOutcome(0.5, 1, ""),
                            ControllerConfig(max_sessions=3))
    assert report.stopped_reason == "error"


def test_a_non_session_outcome_is_an_error():
    report = run_controller(lambda i, seed: 0.5, ControllerConfig(max_sessions=1))
    assert report.stopped_reason == "error"
    assert "SessionOutcome" in report.sessions[0].error
