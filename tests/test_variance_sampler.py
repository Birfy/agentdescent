"""The sampler must chase recent movement, not a frozen lifetime average."""

from __future__ import annotations

import pytest

from agentdescent import VarianceSampler


def test_a_flip_flopping_task_beats_a_settled_one_after_exploration():
    sampler = VarianceSampler(window=5, c=0.01)
    keys = ["settled", "moving"]
    for _ in range(5):
        sampler.record("settled", 1.0)
    for score in (0.0, 1.0, 0.0, 1.0, 0.0):
        sampler.record("moving", score)
    assert sampler.pick(keys, 0) == "moving"


def test_every_task_is_sampled_at_least_once():
    sampler = VarianceSampler(c=0.5)
    seen = set()
    for i in range(20):
        key = sampler.pick(["a", "b", "c"], i)
        seen.add(key)
        sampler.record(key, 0.5)
    assert seen == {"a", "b", "c"}


def test_ties_rotate_by_round_index():
    sampler = VarianceSampler(c=0.0, prior=0.25)
    picks = [sampler.pick(["a", "b"], i) for i in range(4)]
    assert picks == ["a", "b", "a", "b"]


def test_the_window_forgets_old_movement():
    sampler = VarianceSampler(window=2, c=0.0)
    sampler.record("x", 0.0)
    sampler.record("x", 1.0)
    assert sampler.pick(["x"], 0) == "x"
    # two settled values replace the moving pair
    sampler.record("x", 0.5)
    sampler.record("x", 0.5)
    assert sampler._weight("x") == pytest.approx(0.0)


def test_recorded_scores_are_clamped_and_validated():
    sampler = VarianceSampler()
    sampler.record("x", 2.0)
    sampler.record("x", -1.0)
    assert list(sampler._scores["x"]) == [1.0, 0.0]
    with pytest.raises(ValueError, match="window"):
        VarianceSampler(window=0)
    with pytest.raises(ValueError, match="prior"):
        VarianceSampler(prior=1.5)
