"""The loop kernel must resume, record and reuse the existing RSI seams."""

import json

import pytest

from agentdescent.decisions import DecisionLog
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.loop_kernel import (
    LoopConfig,
    LoopKernel,
    LoopObservation,
)
from agentdescent.rsi_control import RSIController
from agentdescent.rsi_telemetry import RSITelemetry
from agentdescent.task_foundry import FoundryConfig, TaskFoundry, failure_seeds
from agentdescent.verifier_ensemble import VerifierEnsemble, VerifierSpec


class _Kill(BaseException):
    """A process kill that the durable step loop must not catch."""


def _kernel(tmp_path, **overrides):
    config = LoopConfig("tenant-a", "ws-1", **overrides)
    store = SqliteRunStore(str(tmp_path / "runs.sqlite3"))
    return LoopKernel(store, config), store


def test_accepts_improvements_and_stops_at_target(tmp_path):
    kernel, store = _kernel(tmp_path, max_generations=5, target=0.9)
    run_id = kernel.submit()
    applied = []

    def step(context):
        return LoopObservation(
            0.5 + 0.1 * context.generation,
            cost=1.0, artifact_id=f"a{context.generation}")

    def apply(context, observation):
        applied.append(observation.artifact_id)

    report = kernel.run(run_id, "worker-1", step, apply=apply)
    assert report.status == "completed"
    assert report.stopped_reason == "target"
    assert report.best_reward == pytest.approx(0.9)
    assert report.accepted == 5
    assert [row.generation for row in report.generations] == [0, 1, 2, 3, 4]
    assert [row.action for row in report.generations] == ["accept"] * 5
    assert applied == ["a0", "a1", "a2", "a3", "a4"]
    assert all(row.applied for row in report.generations)
    assert json.loads(json.dumps(report.to_dict()))["run_id"] == run_id
    store.close()


def test_resume_replays_completed_generations_without_repeating_them(tmp_path):
    kernel, store = _kernel(tmp_path, max_generations=3)
    run_id = kernel.submit()
    calls = []

    def crashing_step(context):
        calls.append(context.generation)
        if context.generation == 1:
            raise _Kill()
        return LoopObservation(0.5, artifact_id=f"a{context.generation}")

    with pytest.raises(_Kill):
        kernel.run(run_id, "worker-1", crashing_step)
    first = kernel.report(run_id)
    assert [row.generation for row in first.generations] == [0]
    assert first.best_reward == pytest.approx(0.5)

    resumed_calls = []

    def resumed_step(context):
        resumed_calls.append(context.generation)
        return LoopObservation(0.6, artifact_id=f"a{context.generation}")

    resumed = LoopKernel(store, kernel.config).run(
        run_id, "worker-2", resumed_step)
    assert resumed_calls == [1, 2]
    assert calls == [0, 1]
    assert [row.generation for row in resumed.generations] == [0, 1, 2]
    assert resumed.best_reward == pytest.approx(0.6)
    store.close()


def test_decision_log_and_telemetry_share_the_same_generations(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    telemetry = RSITelemetry()
    config = LoopConfig("tenant-a", "ws-1", max_generations=2)
    store = SqliteRunStore(str(tmp_path / "runs.sqlite3"))
    kernel = LoopKernel(store, config, decision_log=log, telemetry=telemetry)
    run_id = kernel.submit()

    def step(context):
        return LoopObservation(0.4 + 0.2 * context.generation,
                               artifact_id=f"a{context.generation}")

    report = kernel.run(run_id, "worker-1", step)
    records = [record for record in log if record.kind == "loop"]
    assert len(records) == 2
    assert [record.evidence["generation"] for record in records] == [0, 1]
    assert len([span for span in telemetry.spans if span.kind == "loop"]) == 2
    assert all(row.span_id for row in report.generations)
    store.close()


def test_rollback_callback_fires_when_the_gate_asks_for_it(tmp_path):
    store = SqliteRunStore(str(tmp_path / "runs.sqlite3"))
    config = LoopConfig("tenant-a", "ws-1", max_generations=2)
    rolled_back = []

    def gate(result, gain, risk):
        return "rollback" if gain is not None and gain < 0 else None

    kernel = LoopKernel(
        store, config,
        controller_factory=lambda: RSIController(
            config.control_config(), gate=gate))
    run_id = kernel.submit()

    def step(context):
        return LoopObservation(
            1.0 if context.generation == 0 else 0.2,
            artifact_id=f"a{context.generation}")

    def rollback(context, observation):
        rolled_back.append(observation.artifact_id)

    report = kernel.run(run_id, "worker-1", step, rollback=rollback)
    assert [row.action for row in report.generations] == ["accept", "rollback"]
    assert rolled_back == ["a1"]
    assert report.stopped_reason == "max_sessions"
    assert report.best_reward is None
    store.close()


def test_composes_task_foundry_and_verifier_ensemble(tmp_path):
    seeds = failure_seeds([
        {"task_id": "t1", "prompt": "fix the CSV parser", "reward": 0.0},
    ])
    foundry = TaskFoundry(
        FoundryConfig(per_seed=1, max_candidates=1, holdout_fraction=0.0),
        generator=lambda seed, index: seed,
        validator=lambda candidate: 1.0)
    tasks = foundry.run(seeds).train
    assert tasks
    ensemble = VerifierEnsemble([
        VerifierSpec("passes", lambda candidate: candidate == "good", cost=1),
    ])
    kernel, store = _kernel(tmp_path, max_generations=2, target=1.0)
    run_id = kernel.submit()

    def step(context):
        candidate = "good" if context.generation > 0 else "bad"
        vote = ensemble.evaluate(candidate)
        return LoopObservation(vote.score, cost=vote.cost,
                               artifact_id=tasks[0].task_id)

    report = kernel.run(run_id, "worker-1", step)
    assert report.best_reward == 1.0
    assert report.stopped_reason == "target"
    store.close()


def test_stateful_gate_decisions_are_persisted_and_replayed_without_calls(tmp_path):
    calls = []

    def gate(result, gain, risk):
        calls.append((result.reward, gain))
        return "rollback" if len(calls) == 2 else None

    config = LoopConfig("tenant-a", "ws-1", max_generations=3)
    store = SqliteRunStore(str(tmp_path / "runs.sqlite3"))
    kernel = LoopKernel(
        store, config,
        controller_factory=lambda: RSIController(
            config.control_config(), gate=gate))
    run_id = kernel.submit()
    applied = []
    rolled_back = []

    def step(context):
        if context.generation == 2:
            raise _Kill()
        return LoopObservation(
            1.0 if context.generation == 0 else 0.2,
            artifact_id=f"a{context.generation}")

    def apply(context, observation):
        applied.append(observation.artifact_id)

    def rollback(context, observation):
        rolled_back.append(observation.artifact_id)

    with pytest.raises(_Kill):
        kernel.run(run_id, "worker-1", step, apply=apply, rollback=rollback)
    first = kernel.report(run_id)
    assert [row.action for row in first.generations] == ["accept", "rollback"]
    assert applied == ["a0"]
    assert rolled_back == ["a1"]
    assert len(calls) == 2

    # Reporting must not re-run the gate.
    kernel.report(run_id)
    kernel.report(run_id)
    assert len(calls) == 2

    def resumed(context):
        return LoopObservation(0.5, artifact_id=f"a{context.generation}")

    resumed_kernel = LoopKernel(
        store, config,
        controller_factory=lambda: RSIController(
            config.control_config(), gate=gate))
    report = resumed_kernel.run(
        run_id, "worker-2", resumed, apply=apply, rollback=rollback)
    assert [row.action for row in report.generations] == [
        "accept", "rollback", "accept"]
    assert applied == ["a0", "a2"]
    assert report.generations[0].applied
    assert len(calls) == 3
    store.close()


def test_apply_retry_uses_the_retry_budget_without_losing_the_generation(tmp_path):
    kernel, store = _kernel(tmp_path, max_generations=1, max_retries=2)
    run_id = kernel.submit()
    attempts = []

    def step(context):
        return LoopObservation(0.5, artifact_id="a0")

    def apply(context, observation):
        attempts.append(context.generation)
        if len(attempts) == 1:
            raise RuntimeError("transient apply failure")

    report = kernel.run(run_id, "worker-1", step, apply=apply)
    assert report.status == "completed"
    assert [row.generation for row in report.generations] == [0]
    assert attempts == [0, 0]
    store.close()


def test_cancel_is_terminal_and_reported(tmp_path):
    kernel, store = _kernel(tmp_path, max_generations=2)
    run_id = kernel.submit()
    report = kernel.cancel(run_id, reason="operator stop")
    assert report.status == "cancelled"
    assert report.stopped_reason == "cancelled"
    assert report.generations == ()
    store.close()


def test_invalid_inputs_and_step_errors_are_recorded(tmp_path):
    with pytest.raises(ValueError):
        LoopConfig("", "ws")
    with pytest.raises(ValueError):
        LoopConfig("t", "ws", max_generations=0)
    with pytest.raises(ValueError):
        LoopConfig("t", "ws", risk_limit=2)
    with pytest.raises(ValueError):
        LoopObservation(float("nan"))
    with pytest.raises(ValueError):
        LoopObservation(0.0, risk=-1)
    with pytest.raises(ValueError):
        LoopObservation.from_dict({"artifact_id": "missing-reward"})

    kernel, store = _kernel(tmp_path, max_generations=2)
    run_id = kernel.submit()

    def broken(context):
        return "not an observation"

    report = kernel.run(run_id, "worker-1", broken)
    assert report.generations[0].action == "stop"
    assert "TypeError" in (report.generations[0].error or "")
    assert report.stopped_reason == "error"
    store.close()
