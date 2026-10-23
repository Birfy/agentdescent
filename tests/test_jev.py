"""The distilled JEV student has to learn oracle labels and survive a checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from agentdescent import JEVModel


@dataclass(frozen=True)
class _Task:
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Art:
    name: str = "a"


def _features(artifact, task):
    return [float(task.meta["x"])]


def _oracle(artifact, task):
    return 1.0 if task.meta["x"] >= 0.5 else 0.0


def test_it_learns_a_linear_boundary_from_oracle_labels():
    model = JEVModel(_features, learning_rate=0.5)
    samples = [(_Art(), _Task({"x": i / 100}), float(i >= 50))
               for i in range(101)]
    model.fit(samples, epochs=40)
    correct = sum((model.predict(a, t) >= 0.5) == bool(_oracle(a, t))
                  for a, t, _ in samples)
    assert correct / len(samples) >= 0.9


def test_predictions_are_bounded_and_observations_counted():
    model = JEVModel(_features)
    model.observe(_Art(), _Task({"x": 10.0}), 1.0)
    assert 0.0 <= model.predict(_Art(), _Task({"x": 10.0})) <= 1.0
    assert model.updates == 1


def test_a_feature_length_change_is_refused():
    model = JEVModel(lambda a, t: [1.0] * t.meta.get("n", 1))
    model.observe(_Art(), _Task({"n": 1}), 1.0)
    with pytest.raises(ValueError, match="features"):
        model.predict(_Art(), _Task({"n": 2}))


def test_the_documented_raw_length_features_learn_without_manual_scaling():
    model = JEVModel(lambda a, t: [float(t.meta["n"]), 1.0],
                     learning_rate=0.1)
    samples = [(_Art(), _Task({"n": n}), float(n < 200))
               for n in range(20, 401, 10)]
    model.fit(samples, epochs=10)
    correct = sum((model.predict(a, t) >= 0.5) == bool(n < 200)
                  for a, t, n in [(a, t, t.meta["n"]) for a, t, _ in samples])
    assert correct / len(samples) >= 0.8


def test_a_non_finite_or_out_of_range_target_is_refused():
    model = JEVModel(_features)
    with pytest.raises(ValueError, match="finite"):
        model.observe(_Art(), _Task({"x": 0.5}), float("nan"))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        model.observe(_Art(), _Task({"x": 0.5}), 1.5)


def test_non_finite_features_are_refused():
    model = JEVModel(lambda a, t: [math.nan])
    with pytest.raises(ValueError, match="finite"):
        model.predict(_Art(), _Task())


def test_checkpoint_roundtrip_preserves_predictions():
    model = JEVModel(_features)
    model.fit([(_Art(), _Task({"x": x}), float(x >= 0.5)) for x in (0.1, 0.6, 0.9)],
              epochs=5)
    task = _Task({"x": 0.7})
    before = model.predict(_Art(), task)
    restored = JEVModel.from_dict(model.to_dict(), _features)
    assert restored.predict(_Art(), task) == pytest.approx(before)
    assert restored.updates == model.updates


def test_a_label_loop_reduces_the_student_oracle_gap():
    model = JEVModel(_features, learning_rate=0.5)
    tasks = [_Task({"x": i / 10}) for i in range(11)]
    art = _Art()

    def mean_gap():
        return sum(abs(model.predict(art, task) - _oracle(art, task))
                   for task in tasks) / len(tasks)

    before = mean_gap()
    for _ in range(5):
        hardest = sorted(tasks, key=lambda task: -abs(
            model.predict(art, task) - _oracle(art, task)))[:3]
        for task in hardest:
            model.observe(art, task, _oracle(art, task))
    assert model.updates == 15
    assert mean_gap() <= before
