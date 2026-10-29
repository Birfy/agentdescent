"""A preference model has to rank the side a human actually preferred."""

import pytest

from agentdescent.preference import (
    PreferenceBuffer,
    PreferenceModel,
    PreferencePair,
)


def test_clear_pairwise_preferences_are_learned():
    pairs = [
        PreferencePair((1.0, 0.0), (0.0, 1.0), "left"),
        PreferencePair((1.0, 0.2), (0.2, 1.0), "left"),
        PreferencePair((0.0, 1.0), (1.0, 0.0), "right"),
    ]
    model = PreferenceModel()
    report = model.fit(pairs)
    assert report.accuracy == 1.0
    assert report.verdict == "learned"
    assert model.weights[0] > model.weights[1]
    assert model.probability((1, 0), (0, 1)) > 0.5
    assert model.score((1, 0)) > model.score((0, 1))


def test_ties_are_targets_not_dropped():
    pair = PreferencePair((1.0, 0.0), (0.0, 1.0), "tie")
    report = PreferenceModel().fit([pair])
    assert report.n == 1
    assert report.accuracy == 0.0
    assert report.log_loss == pytest.approx(0.693147, abs=1e-3)


def test_buffer_selects_the_farthest_candidate():
    buffer = PreferenceBuffer()
    buffer.add(PreferencePair((0.0, 0.0), (1.0, 0.0), "left"))
    index = buffer.select_next([(0.1, 0.0), (5.0, 5.0), (0.0, 1.0)])
    assert index == 1
    assert len(buffer.to_pairs()) == 1


def test_invalid_pairs_and_unfitted_use_are_refused():
    with pytest.raises(ValueError):
        PreferencePair((), (1,), "left")
    with pytest.raises(ValueError):
        PreferencePair((1,), (1, 2), "left")
    with pytest.raises(ValueError):
        PreferencePair((1,), (2,), "maybe")
    with pytest.raises(ValueError):
        PreferencePair((float("nan"),), (1,), "left")
    model = PreferenceModel()
    with pytest.raises(RuntimeError):
        model.score((1,))
    with pytest.raises(RuntimeError):
        model.probability((1,), (2,))
    with pytest.raises(ValueError):
        model.fit([])
    with pytest.raises(ValueError):
        PreferenceBuffer().select_next([])
