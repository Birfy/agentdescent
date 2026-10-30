"""Production traces must become a validated train/holdout task stream."""

import pytest

from agentdescent.data_flywheel import DataFlywheel, FlywheelConfig, TraceSample
from agentdescent.task_foundry import FoundryConfig, TaskFoundry
from agentdescent.task_pipeline import (
    TaskPipeline, TaskPipelineConfig,
)


def _sample(i, *, reward=0.1, task=None, output=None, error=None):
    return TraceSample(
        f"t{i}", task or f"fix task {i}", output or f"out {i}",
        reward=reward, cost=1.0, error=error)


def _pipeline(*, holdout_fraction=0.5, contamination=None, **foundry):
    foundry = TaskFoundry(
        FoundryConfig(holdout_fraction=holdout_fraction, max_candidates=20,
                      **foundry),
        generator=lambda seed, _attempt: seed,
        validator=lambda _candidate: 1.0,
        contamination=contamination or (lambda _candidate: 0.0))
    return TaskPipeline(
        DataFlywheel(FlywheelConfig(max_per_cluster=2)),
        foundry, config=TaskPipelineConfig(failure_threshold=0.5))


def test_failures_become_train_and_holdout_tasks():
    pipeline = _pipeline()
    report = pipeline.run([
        _sample(1, reward=0.9, task="already works"),
        _sample(2, reward=0.1, task="fix csv"),
        _sample(3, reward=0.1, task="fix csv", output="out 2"),
        _sample(4, reward=0.2, task="fix parser"),
        _sample(5, reward=0.1, task="errored", error="tool"),
    ])
    assert report.flywheel.accepted_count == 3
    assert len(report.seeds) == 2
    assert len(report.accepted) == 2
    assert report.train and report.holdout
    assert report.verdict == "ready"
    assert "1 train / 1 holdout" in report.detail
    data = report.to_dict()
    assert data["flywheel"]["accepted"] == 3
    assert data["seeds"] == 2
    assert data["verdict"] == "ready"


def test_no_failures_is_empty_not_an_error():
    report = _pipeline().run([_sample(1, reward=0.9)])
    assert report.verdict == "empty"
    assert report.seeds == ()
    assert report.accepted == ()


def test_holdout_can_be_required_or_optional():
    one_train = _pipeline(holdout_fraction=0.0)
    report = one_train.run([_sample(1, reward=0.1)])
    assert report.verdict == "no-holdout"
    optional = TaskPipeline(
        one_train.flywheel, one_train.foundry,
        config=TaskPipelineConfig(require_holdout=False))
    assert optional.run([_sample(2, reward=0.1)]).verdict == "ready"


def test_foundry_refusals_are_not_turned_into_ready():
    insufficient = _pipeline(min_tasks=5)
    report = insufficient.run([_sample(1, reward=0.1)])
    assert report.foundry.verdict == "insufficient"
    assert report.verdict == "insufficient"

    contaminated = _pipeline(
        max_contamination=0.1,
        contamination=lambda candidate: 0.6 if candidate.task_id == "t2" else 0.0)
    report = contaminated.run([
        _sample(1, reward=0.1, task="good"),
        _sample(2, reward=0.2, task="bad"),
    ])
    assert report.foundry.verdict == "filtered"
    assert report.verdict == "filtered"


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        TaskPipelineConfig(failure_threshold=2)
    with pytest.raises(ValueError):
        TaskPipelineConfig(max_seeds=0)
    with pytest.raises(TypeError):
        TaskPipeline(object(), _pipeline().foundry)
    with pytest.raises(TypeError):
        _pipeline().run([{"trace_id": "x"}])
