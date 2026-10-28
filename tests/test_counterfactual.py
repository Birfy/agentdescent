"""Counterfactual replay must answer “what if this step were absent?”."""

import pytest

from agentdescent.counterfactual import (
    Intervention,
    ReplayLab,
    ReplayTrace,
    TraceStep,
    apply_interventions,
)


def _trace(*weights, trace_id="t"):
    return ReplayTrace(trace_id, tuple(
        TraceStep(f"s{i}", payload=weight) for i, weight in enumerate(weights)))


def _sum(steps):
    return sum(step.payload for step in steps if not step.metadata.get("disabled"))


def test_baseline_and_leave_one_out_are_additive():
    lab = ReplayLab(_sum)
    trace = _trace(1.0, 2.0, 3.0)
    assert lab.baseline(trace) == 6.0
    report = lab.leave_one_out(trace)
    assert report.baseline == 6.0
    assert [row.delta for row in report.results] == [1.0, 2.0, 3.0]
    assert report.largest_effect.intervention.step_id == "s2"
    assert report.largest_effect.delta == 3.0


def test_remove_replace_disable_and_apply_interventions():
    trace = _trace(1.0, 2.0)
    assert [s.payload for s in apply_interventions(trace, [Intervention("s0")])] == [2.0]
    replaced = apply_interventions(trace, [
        Intervention("s0", "replace", TraceStep("s0", payload=10.0))])
    assert [s.payload for s in replaced] == [10.0, 2.0]
    disabled = apply_interventions(trace, [Intervention("s0", "disable")])
    assert disabled[0].metadata["disabled"] is True
    assert _sum(disabled) == 2.0


def test_counterfactual_delta_is_baseline_minus_replay():
    lab = ReplayLab(_sum)
    trace = _trace(1.0, 2.0)
    result = lab.counterfactual(trace, [Intervention("s1", "remove")])
    assert result.score == 1.0
    assert result.delta == 2.0
    assert "intervention" in result.detail


def test_shapley_splits_a_synergy_between_steps():
    trace = ReplayTrace("syn", (
        TraceStep("a", payload="a"),
        TraceStep("b", payload="b"),
    ))

    def evaluate(steps):
        ids = {step.payload for step in steps}
        return 2.0 if {"a", "b"} <= ids else 0.0

    report = ReplayLab(evaluate).shapley(trace)
    assert report.gain == 2.0
    assert [row.credit for row in report.ranking] == [1.0, 1.0]


def test_cache_avoids_replaying_identical_traces():
    calls = []

    def evaluate(steps):
        calls.append(tuple(step.step_id for step in steps))
        return _sum(steps)

    lab = ReplayLab(evaluate)
    trace = _trace(1.0, 2.0)
    lab.baseline(trace)
    lab.baseline(trace)
    assert len(calls) == 1
    lab.leave_one_out(trace)
    assert len(calls) == 3
    assert lab.calls == 3


def test_unknown_step_and_empty_interventions_are_refused():
    lab = ReplayLab(_sum)
    trace = _trace(1.0)
    with pytest.raises(KeyError):
        lab.counterfactual(trace, [Intervention("missing")])
    with pytest.raises(ValueError):
        lab.counterfactual(trace, [])


def test_non_finite_replay_and_invalid_records_are_refused():
    with pytest.raises(ValueError):
        ReplayLab(lambda _steps: float("nan")).baseline(_trace(1.0))
    with pytest.raises(ValueError):
        TraceStep("", payload=1)
    with pytest.raises(ValueError):
        TraceStep("s", cost=-1)
    with pytest.raises(ValueError):
        ReplayTrace("t", (TraceStep("s"), TraceStep("s")))
    with pytest.raises(ValueError):
        ReplayTrace("t", (TraceStep("s"),), score=float("inf"))
    with pytest.raises(ValueError):
        Intervention("s", "replace")
    with pytest.raises(ValueError):
        Intervention("s", "replace", TraceStep("other"))
    with pytest.raises(ValueError):
        Intervention("s", "remove", TraceStep("s"))
    with pytest.raises(ValueError):
        Intervention("s", "magic")


def test_cache_does_not_reuse_ids_across_fresh_payload_objects():
    lab = ReplayLab(lambda steps: sum(step.payload[0] for step in steps))
    for value in range(50):
        trace = ReplayTrace("t", (TraceStep("a", payload=[value]),))
        assert lab.baseline(trace) == float(value)


def test_metadata_only_replacement_is_not_a_stale_cache_hit():
    payload = object()

    def evaluate(steps):
        return 10.0 if steps[0].metadata.get("bonus") else 1.0

    lab = ReplayLab(evaluate)
    trace = ReplayTrace("t", (TraceStep("a", payload=payload),))
    assert lab.baseline(trace) == 1.0
    replaced = TraceStep("a", payload=payload, metadata={"bonus": True})
    result = lab.counterfactual(trace, [Intervention("a", "replace", replaced)])
    assert result.score == 10.0


def test_multi_intervention_result_keeps_the_real_tuple():
    lab = ReplayLab(_sum)
    trace = _trace(1.0, 2.0, 3.0)
    result = lab.counterfactual(trace, [
        Intervention("s0", "remove"),
        Intervention("s1", "disable"),
    ])
    assert [item.action for item in result.interventions] == ["remove", "disable"]
    assert result.intervention is None


def test_conflicting_duplicate_interventions_are_refused():
    lab = ReplayLab(_sum)
    with pytest.raises(ValueError):
        lab.counterfactual(_trace(1.0), [
            Intervention("s0", "remove"),
            Intervention("s0", "disable"),
        ])


def test_trace_steps_are_coerced_to_a_tuple_and_type_checked():
    trace = ReplayTrace("t", [TraceStep("a", payload=1)])
    assert isinstance(trace.steps, tuple)
    with pytest.raises(TypeError):
        ReplayTrace("t", [object()])


def test_metadata_object_keys_do_not_reuse_freed_addresses():
    class Probe:
        pass

    def evaluate(steps):
        return float(steps[0].metadata["o"].v)

    lab = ReplayLab(evaluate)
    for value in range(50):
        o = Probe()
        o.v = value
        trace = ReplayTrace("t", (TraceStep("a", payload=1, metadata={"o": o}),))
        assert lab.baseline(trace) == float(value)
        del o
