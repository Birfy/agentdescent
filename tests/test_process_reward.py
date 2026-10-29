"""Step rewards must rank correct steps and report calibration."""

import pytest

from agentdescent.process_reward import (
    ProcessRewardModel,
    StepExample,
)


def test_learns_clear_step_labels_and_reranks():
    examples = [
        StepExample((1.0, 0.0), 1.0),
        StepExample((0.0, 1.0), 0.0),
        StepExample((1.0, 0.2), 1.0),
        StepExample((0.2, 1.0), 0.0),
    ]
    model = ProcessRewardModel()
    report = model.fit(examples, min_n=4)
    assert report.accuracy == 1.0
    assert report.verdict == "calibrated"
    assert model.report is report
    assert model.score((1.0, 0.0)) > model.score((0.0, 1.0))
    assert model.select([(0.0, 1.0), (1.0, 0.0)]) == 1
    assert model.rank([(0.0, 1.0), (1.0, 0.0)]) == (1, 0)


def test_calibration_fields_are_reported():
    report = ProcessRewardModel().fit([
        StepExample((0.0,), 0.0),
        StepExample((0.0,), 0.0),
        StepExample((0.0,), 0.0),
    ])
    assert 0.0 <= report.ece <= 1.0
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.log_loss <= 5.0


def test_small_fits_are_marked_insufficient():
    model = ProcessRewardModel()
    report = model.fit([StepExample((1.0,), 1.0)])
    assert report.n == 1
    assert report.verdict == "insufficient"
    assert model.score((1.0,)) > 0.5


def test_invalid_inputs_and_unfitted_use_are_refused():
    with pytest.raises(ValueError):
        StepExample((), 1.0)
    with pytest.raises(ValueError):
        StepExample((1.0,), 1.5)
    with pytest.raises(ValueError):
        StepExample((1.0,), 1.0, step_index=-1)
    model = ProcessRewardModel()
    with pytest.raises(RuntimeError):
        model.score((1.0,))
    with pytest.raises(ValueError):
        model.fit([])
    with pytest.raises(ValueError):
        model.fit([StepExample((1.0,), 1.0)], min_n=0)
    with pytest.raises(ValueError):
        model.fit([StepExample((1.0,), 1.0), StepExample((1.0, 2.0), 0.0)])
    model.fit([StepExample((1.0,), 1.0)])
    with pytest.raises(ValueError):
        model.score((1.0, 2.0))
    with pytest.raises(ValueError):
        model.select([])
