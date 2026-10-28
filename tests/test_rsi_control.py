"""The control plane must turn outcomes into the next action, not just count."""

from types import SimpleNamespace

import pytest

from agentdescent.guards import GuardDecision, GuardPipelineResult
from agentdescent.policies import AcceptDecision
from agentdescent.rsi_control import (
    ControlAction,
    ControlConfig,
    RSIController,
    SessionResult,
)


class _Log:
    def __init__(self):
        self.rows = []

    def record(self, **kwargs):
        self.rows.append(kwargs)
        return kwargs


def test_baseline_then_improvement_then_stagnation():
    controller = RSIController(ControlConfig(patience=2, max_sessions=10))
    baseline = controller.observe(SessionResult(0.4))
    assert baseline.action == ControlAction.ACCEPT
    assert controller.best_reward == 0.4
    improved = controller.observe(SessionResult(0.6))
    assert improved.action == ControlAction.ACCEPT
    assert improved.gain == pytest.approx(0.2)
    assert controller.best_reward == 0.6
    flat = controller.observe(SessionResult(0.55))
    assert flat.action == ControlAction.REJECT
    stopped = controller.observe(SessionResult(0.55))
    assert stopped.action == ControlAction.REJECT and stopped.stop
    assert "without a new best" in stopped.reason
    assert controller.stopped


def test_target_keeps_the_accept_verdict_and_stops_separately():
    target = RSIController(ControlConfig(target=0.5))
    decision = target.observe(SessionResult(0.5))
    assert decision.action == ControlAction.ACCEPT
    assert decision.accepted and decision.stop
    assert target.report().accepted == 1
    assert target.report().best_reward == 0.5
    assert target.report().stopped_reason == "target"


def test_target_does_not_fire_on_a_rejected_session():
    controller = RSIController(ControlConfig(target=0.9, patience=9),
                               gate=lambda *_: False)
    decision = controller.observe(SessionResult(0.95))
    assert decision.action == ControlAction.REJECT
    assert not decision.stop
    assert controller.best_reward is None


def test_budget_and_max_sessions_stop_without_losing_acceptance():
    budget = RSIController(ControlConfig(max_cost=3.0, max_sessions=10))
    decision = budget.observe(SessionResult(0.1, cost=3.0))
    assert decision.action == ControlAction.ACCEPT and decision.stop
    assert budget.report().stopped_reason == "budget"

    sessions = RSIController(ControlConfig(max_sessions=1))
    decision = sessions.observe(SessionResult(0.1))
    assert decision.action == ControlAction.ACCEPT and decision.stop
    assert sessions.report().stopped_reason == "max_sessions"


def test_risky_new_session_is_rejected_not_rolled_back():
    controller = RSIController(ControlConfig(risk_limit=0.2, max_sessions=10))
    assert controller.observe(SessionResult(0.5, risk=0.1)).action == ControlAction.ACCEPT
    rejected = controller.observe(SessionResult(0.9, risk=0.8))
    assert rejected.action == ControlAction.REJECT
    assert controller.best_reward == 0.5


def test_an_explicit_rollback_reverts_the_accepted_best():
    controller = RSIController(ControlConfig(risk_limit=1.0, max_sessions=10),
                               risk_gate=lambda result, *_: "rollback" if result.reward > 0.8 else None)
    controller.observe(SessionResult(0.5, risk=0.0))
    controller.observe(SessionResult(0.7, risk=0.0))
    assert controller.best_reward == 0.7
    rolled = controller.observe(SessionResult(0.9, risk=0.0))
    assert rolled.action == ControlAction.ROLLBACK
    assert controller.best_reward == 0.5
    assert controller.report().accepted == 1


def test_gate_forms_are_normalised_and_nested_guards_unwrap():
    def run(gate):
        c = RSIController(ControlConfig(patience=9), gate=gate)
        c.observe(SessionResult(0.5))
        return c.observe(SessionResult(0.9))

    assert run(lambda *_: False).action == ControlAction.REJECT
    assert run(lambda *_: "defer").action == ControlAction.DEFER
    assert run(lambda *_: SimpleNamespace(accepted=False, detail="calibration")).action == ControlAction.REJECT
    assert run(lambda *_: SimpleNamespace(action="rollback", detail="unstable")).action == ControlAction.ROLLBACK
    assert run(lambda *_: None).action == ControlAction.ACCEPT
    guard = GuardDecision("g", AcceptDecision(accept=False, category="regress"))
    assert run(lambda *_: guard).action == ControlAction.REJECT
    pipeline = GuardPipelineResult(accepted=False, failed_at="g")
    assert run(lambda *_: pipeline).action == ControlAction.REJECT


def test_unrecognised_gate_output_fails_closed():
    controller = RSIController(ControlConfig(patience=9), gate=lambda *_: "block")
    with pytest.raises(ValueError):
        controller.observe(SessionResult(0.5))
    unknown = RSIController(ControlConfig(patience=9), gate=lambda *_: object())
    with pytest.raises(ValueError):
        unknown.observe(SessionResult(0.5))


def test_risk_gate_can_defer_or_stop():
    def run(value):
        c = RSIController(ControlConfig(patience=9),
                          risk_gate=lambda *_: value)
        c.observe(SessionResult(0.5))
        return c.observe(SessionResult(0.9))

    assert run("defer").action == ControlAction.DEFER
    assert run("stop").action == ControlAction.STOP


def test_budget_policy_sees_state_and_can_drive_run_budget():
    seen = []

    def policy(state):
        seen.append(dict(state))
        return 2.0

    controller = RSIController(ControlConfig(max_sessions=2), budget_policy=policy)
    first = controller.observe(SessionResult(0.5, cost=1.0))
    assert first.budget_scale == 2.0
    assert seen[0]["index"] == 0 and seen[0]["best"] is None
    assert seen[0]["gain"] is None

    scales = []

    def run_session(index, seed, budget):
        scales.append(budget)
        return SessionResult(0.1 * index)

    looped = RSIController(ControlConfig(max_sessions=2), budget_policy=policy)
    looped.run(run_session)
    assert scales == [1.0, 2.0]

    bad = RSIController(ControlConfig(), budget_policy=lambda _: float("nan"))
    with pytest.raises(ValueError):
        bad.observe(SessionResult(0.5))


def test_error_and_kill_switch_stop_immediately():
    errored = RSIController(ControlConfig())
    assert errored.observe(SessionResult(0.0, error="boom")).action == ControlAction.STOP
    assert errored.report().stopped_reason == "error"
    assert "boom" in errored.report().reason

    class Switch:
        def engaged(self):
            return True

        def reason(self):
            return "operator"

    stopped = RSIController(ControlConfig(), kill_switch=Switch())
    assert stopped.observe(SessionResult(1.0)).action == ControlAction.STOP
    assert stopped.report().stopped_reason == "kill_switch"


def test_decision_log_is_optional_and_a_broken_log_does_not_kill_the_loop():
    log = _Log()
    controller = RSIController(ControlConfig(), decision_log=log)
    controller.observe(SessionResult(0.5, cost=2.0))
    assert log.rows[0]["kind"] == "control"
    assert log.rows[0]["accepted"] is True

    class Broken:
        def record(self, **kwargs):
            raise RuntimeError("disk full")

    broken = RSIController(ControlConfig(), decision_log=Broken())
    assert broken.observe(SessionResult(0.5)).action == ControlAction.ACCEPT


def test_run_session_drives_the_loop_and_report():
    outcomes = [SessionResult(0.2), SessionResult(0.7),
                SessionResult(0.7), SessionResult(0.7)]

    def run_session(index, seed):
        assert seed == 10 + index
        return outcomes[index]

    controller = RSIController(ControlConfig(patience=2, max_sessions=4))
    report = controller.run(run_session, base_seed=10)
    assert report.accepted == 2
    assert report.best_reward == 0.7
    assert report.stopped_reason == "patience"
    assert len(report.decisions) == 4


def test_defer_does_not_count_as_stagnation():
    controller = RSIController(ControlConfig(patience=1, max_sessions=5),
                               gate=lambda *_: "defer")
    first = controller.observe(SessionResult(0.5))
    second = controller.observe(SessionResult(0.6))
    assert first.action == ControlAction.DEFER
    assert second.action == ControlAction.DEFER
    assert not second.stop


def test_rollback_still_respects_the_budget():
    controller = RSIController(ControlConfig(risk_limit=1.0, max_cost=1,
                                              max_sessions=10),
                               risk_gate=lambda *_: "rollback")
    controller.observe(SessionResult(0.5, cost=0, risk=0.1))
    stopped = controller.observe(SessionResult(0.9, cost=2, risk=0.1))
    assert stopped.action == ControlAction.ROLLBACK
    assert stopped.stop
    assert controller.report().stopped_reason == "budget"


def test_malformed_config_and_results_are_refused():
    with pytest.raises(ValueError):
        ControlConfig(max_sessions=0)
    with pytest.raises(ValueError):
        ControlConfig(patience=0)
    with pytest.raises(ValueError):
        ControlConfig(risk_limit=1.5)
    with pytest.raises(ValueError):
        ControlConfig(max_cost=0)
    with pytest.raises(ValueError):
        SessionResult(float("nan"))
    with pytest.raises(ValueError):
        SessionResult(0.1, cost=-1)
    with pytest.raises(ValueError):
        SessionResult(0.1, risk=1.2)
    with pytest.raises(TypeError):
        RSIController(ControlConfig()).observe({"reward": 1.0})
