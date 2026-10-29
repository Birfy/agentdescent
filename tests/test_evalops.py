"""EvalOps must make a regression gate reproducible and actionable."""

import pytest

from agentdescent.evalops import (
    Dataset,
    DatasetRegistry,
    DriftMonitor,
    EvalOps,
    EvalOutput,
    EvalSpec,
    RegressionGate,
)


def _tasks(n=6):
    return tuple({"task_id": f"t{i}", "prompt": f"p{i}"} for i in range(n))


def test_dataset_registry_versions_and_content_hash():
    registry = DatasetRegistry()
    one = registry.add(Dataset("gsm", 1, _tasks(3)))
    two = registry.add(Dataset("gsm", 2, _tasks(4)))
    assert registry.get("gsm").version == 2
    assert registry.get("gsm", 1).content_hash == one.content_hash
    assert registry.versions("gsm") == (1, 2)
    assert one.content_hash != two.content_hash
    with pytest.raises(ValueError):
        registry.add(Dataset("gsm", 1, _tasks(3)))
    with pytest.raises(KeyError):
        registry.get("missing")


def test_run_computes_paired_delta_ci_and_cost():
    ops = EvalOps(clock=lambda: 100.0)
    ops.add_dataset(Dataset("d", 1, _tasks(6)))
    ops.add_spec(EvalSpec("e", "d", seeds=(0,), min_effect=0.1, bootstrap=500))
    result = ops.run(
        "e",
        lambda task, _seed: EvalOutput(0.8, cost=1),
        lambda task, _seed: EvalOutput(0.5, cost=1),
    )
    assert result.delta == pytest.approx(0.3)
    assert result.passed
    assert result.ci[0] > 0.1
    assert result.cost == 12
    assert result.to_dict()["dataset_id"] == "d"


def test_gate_rejects_small_or_uncertain_deltas():
    ops = EvalOps()
    ops.add_dataset(Dataset("d", 1, _tasks(3)))
    ops.add_spec(EvalSpec("e", "d", min_effect=0.02, bootstrap=200))
    ops.run("e", lambda *_: 0.51, lambda *_: 0.5)
    assert not ops.gate("e").passed
    assert "delta" in ops.gate("e").reason


def test_drift_monitor_detects_mean_and_fraction_shift():
    monitor = DriftMonitor(threshold=0.1, fraction=0.2)
    report = monitor.observe({"a": 1.0, "b": 1.0, "c": 1.0},
                             {"a": 0.5, "b": 1.1, "c": 0.9})
    assert report.drift
    assert report.mean_shift < 0
    assert report.fraction_beyond > 0
    assert not monitor.observe({"a": 1.0}, {"a": 1.0}).drift
    assert monitor.observe({}, {}).shared == 0


def test_schedule_and_snapshot():
    clock = {"now": 100.0}
    ops = EvalOps(clock=lambda: clock["now"])
    ops.add_dataset(Dataset("d", 1, _tasks(2)))
    ops.add_spec(EvalSpec("e", "d", seeds=(0,), schedule_seconds=60))
    assert ops.due() == ("e",)
    ops.run("e", lambda *_: 1.0, lambda *_: 0.0)
    assert ops.due() == ()
    clock["now"] = 161.0
    assert ops.due() == ("e",)
    data = ops.snapshot()
    assert data["datasets"] and data["specs"] and data["results"]


def test_invalid_specs_datasets_and_outputs_are_refused():
    with pytest.raises(ValueError):
        Dataset("", 1, _tasks())
    with pytest.raises(ValueError):
        Dataset("d", 0, _tasks())
    with pytest.raises(ValueError):
        Dataset("d", 1, ())
    with pytest.raises(ValueError):
        EvalSpec("e", "d", seeds=())
    with pytest.raises(ValueError):
        EvalSpec("e", "d", alpha=1.0)
    with pytest.raises(ValueError):
        EvalSpec("e", "d", schedule_seconds=0)
    with pytest.raises(ValueError):
        EvalOutput(float("nan"))
    with pytest.raises(ValueError):
        EvalOutput(0.1, cost=-1)
    ops = EvalOps()
    ops.add_dataset(Dataset("d", 1, _tasks()))
    with pytest.raises(KeyError):
        ops.run("missing", lambda *_: 0.1)
    with pytest.raises(KeyError):
        ops.gate("missing")
    with pytest.raises(ValueError):
        DriftMonitor(fraction=2)


def test_default_gate_uses_the_stored_min_effect():
    ops = EvalOps()
    ops.add_dataset(Dataset("d", 1, _tasks(6)))
    ops.add_spec(EvalSpec("e", "d", min_effect=0.01, bootstrap=500))
    result = ops.run("e", lambda *_: 0.8, lambda *_: 0.5)
    assert result.passed
    assert ops.gate("e").passed


def test_dataset_tasks_are_deep_frozen_and_hash_is_cached():
    tasks = [{"task_id": "a", "nested": {"x": 1}}]
    dataset = Dataset("d", 1, tasks)
    before = dataset.content_hash
    with pytest.raises(TypeError):
        dataset.tasks[0]["nested"]["x"] = 2
    tasks[0]["nested"]["x"] = 3
    assert dataset.content_hash == before


def test_duplicate_task_ids_and_contaminated_datasets_fail_closed():
    with pytest.raises(ValueError):
        Dataset("d", 1, [{"task_id": "same"}, {"task_id": "same"}])
    ops = EvalOps()
    ops.add_dataset(Dataset("d", 1, _tasks(3), contaminated=True))
    ops.add_spec(EvalSpec("e", "d"))
    with pytest.raises(ValueError):
        ops.run("e", lambda *_: 1.0)
    ops.add_spec(EvalSpec("override", "d", allow_contaminated=True))
    assert ops.run("override", lambda *_: 1.0).score == 1.0


def test_public_result_and_spec_maps_are_read_only():
    ops = EvalOps()
    ops.add_dataset(Dataset("d", 1, _tasks(2)))
    ops.add_spec(EvalSpec("e", "d"))
    ops.run("e", lambda *_: 1.0)
    with pytest.raises(TypeError):
        ops.results["forged"] = object()
    with pytest.raises(TypeError):
        ops.specs["forged"] = object()
