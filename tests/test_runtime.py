"""A guarded run must inject acceptance, record the decision and honour STOP."""

from __future__ import annotations

import json

import pytest

from agentdescent import run_guarded
from agentdescent.policies import AcceptDecision
from tests.test_evolvespec import _dir_spec


class _Allow:
    def accept(self, ctx):
        return AcceptDecision(True)

    def describe(self):
        return ("allow",)


class _Deny:
    def accept(self, ctx):
        return AcceptDecision(False, "below-threshold", "gold found")

    def describe(self):
        return ("deny",)


class _Log:
    def __init__(self):
        self.records = []

    def record(self, **kwargs):
        rec = type("R", (), kwargs)()
        self.records.append(rec)
        return rec


class _Switch:
    def __init__(self, engaged=False, reason=""):
        self._engaged, self._reason = engaged, reason

    def engaged(self):
        return self._engaged

    def reason(self):
        return self._reason


def test_a_guarded_run_records_acceptance_and_decision(tmp_path):
    spec = _dir_spec(str(tmp_path))
    log = _Log()
    guarded = run_guarded(spec, acceptance=_Allow(), decision_log=log,
                          repo_path=str(tmp_path / "repo"))
    assert guarded.result.error is None
    assert guarded.acceptance is not None
    assert log.records and log.records[0].accepted is True
    assert log.records[0].evidence["acceptance"] == ["allow", "DefaultAcceptance"]


def test_a_rejecting_acceptance_stops_adoption(tmp_path):
    spec = _dir_spec(str(tmp_path))
    guarded = run_guarded(spec, acceptance=_Deny(),
                          repo_path=str(tmp_path / "repo"))
    assert guarded.result.error is None
    assert guarded.result.outcomes().get("committed", 0) == 0
    assert guarded.result.outcomes().get("below-threshold", 0) > 0


def test_the_kill_switch_is_checked_before_compose(tmp_path):
    with pytest.raises(RuntimeError, match="cost spike"):
        run_guarded(_dir_spec(str(tmp_path)),
                    kill_switch=_Switch(True, "cost spike"))


def test_the_supplied_acceptance_is_the_one_injected(tmp_path):
    spec = _dir_spec(str(tmp_path))
    guarded = run_guarded(spec, acceptance=_Allow(),
                          repo_path=str(tmp_path / "repo"))
    assert guarded.acceptance.describe() == ("allow", "DefaultAcceptance")


def test_no_acceptance_is_allowed_and_noted(tmp_path):
    guarded = run_guarded(_dir_spec(str(tmp_path)), repo_path=str(tmp_path / "repo"))
    assert any("no acceptance" in note for note in guarded.notes)


def test_the_default_gate_still_runs_under_an_allowing_guard(tmp_path, monkeypatch):
    from agentdescent.defaults import DefaultAcceptance

    def deny(self, ctx):
        return AcceptDecision(False, "below-threshold", "default refused")

    monkeypatch.setattr(DefaultAcceptance, "accept", deny)
    guarded = run_guarded(_dir_spec(str(tmp_path)), acceptance=_Allow(),
                          repo_path=str(tmp_path / "repo"))
    assert guarded.result.error is None
    assert guarded.result.outcomes().get("committed", 0) == 0
    assert guarded.result.outcomes().get("below-threshold", 0) > 0


def test_chain_forwards_stateful_hooks():
    seen = []

    class Stateful:
        def accept(self, ctx):
            return AcceptDecision(True)

        def checkpoint_state(self):
            seen.append("checkpoint")
            return {"n": 1}

        def restore_state(self, state):
            seen.append(("restore", state))

        def on_screen_reject(self):
            seen.append("reject")

    from agentdescent.runtime import _Chain

    chain = _Chain(Stateful(), Stateful())
    state = chain.checkpoint_state()
    chain.restore_state(state)
    chain.on_screen_reject()
    assert state == {"first": {"n": 1}, "second": {"n": 1}}
    assert seen == ["checkpoint", "checkpoint", ("restore", {"n": 1}),
                    ("restore", {"n": 1}), "reject", "reject"]


def test_an_aggregator_factory_is_noted(tmp_path, monkeypatch):
    import agentdescent.runtime as runtime

    class Result:
        error = None
        final_reward = 0.0
        history = []
        usage = None
        wallclock = 0.0
        stop_reason = "rounds"

        def outcomes(self):
            return {}

    class Comp:
        notes = []
        kwargs = {"policies": None, "aggregator_factory": object()}

        def run(self):
            return Result()

    spec = _dir_spec(str(tmp_path))
    monkeypatch.setattr(runtime, "compose", lambda spec, **kw: Comp())
    guarded = runtime.run_guarded(spec, acceptance=_Allow())
    assert any("aggregator_factory" in note for note in guarded.notes)


def test_decision_log_marks_named_algorithm_adoptions_accepted(tmp_path, monkeypatch):
    """A gepa-select (etc.) outcome must set decision.accepted, not only committed."""
    from types import SimpleNamespace
    from agentdescent import runtime as runtime_mod

    class _Comp:
        def __init__(self):
            self.kwargs = {"policies": None, "artifact_id": "prompt"}
            self.notes = ()

        def run(self):
            history = [SimpleNamespace(committed=1, reasons={"gepa-select": 1}),
                       SimpleNamespace(committed=1, reasons={"gepa-select": 1})]
            return SimpleNamespace(
                error=None, final_reward=0.75, history=history,
                stop_reason="max_rounds", wallclock=1.0, usage=None,
                outcomes=lambda: {"gepa-select": 2})

    monkeypatch.setattr(runtime_mod, "compose", lambda spec, **kw: _Comp())
    log = _Log()
    guarded = run_guarded(_dir_spec(str(tmp_path)), decision_log=log,
                          repo_path=str(tmp_path / "repo"))
    assert log.records and log.records[0].accepted is True
    assert guarded.result.outcomes() == {"gepa-select": 2}
