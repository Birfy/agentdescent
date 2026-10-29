"""Continual meta-updates must improve on validation or roll back."""

import pytest

from agentdescent.continual_meta import (
    ContinualMetaLearner,
    MetaExample,
    ReplayBuffer,
)


def _example(feature, best):
    rewards = {"a": 1.0 if best == "a" else 0.0,
               "b": 1.0 if best == "b" else 0.0}
    return MetaExample((feature,), rewards)


def test_learner_improves_and_accepts_validated_updates():
    learner = ContinualMetaLearner(["a", "b"], dim=1, learning_rate=0.5,
                                   min_gain=0.0)
    for _ in range(5):
        for feature, best in ((1.0, "a"), (-1.0, "b")):
            learner.observe(_example(feature, best))
        update = learner.step([_example(1.0, "a"), _example(-1.0, "b")],
                              batch_size=4, seed=0)
        assert update.after >= update.before
    assert learner.report().accepted > 0
    assert learner.policy.expected_reward(
        [_example(1.0, "a"), _example(-1.0, "b")]) > 0.5
    assert learner.act((1.0,)) == "a"
    assert learner.act((-1.0,)) == "b"


def test_a_bad_update_is_rejected_and_rolled_back():
    learner = ContinualMetaLearner(["a", "b"], dim=1, min_gain=10.0)
    for _ in range(3):
        learner.observe(_example(1.0, "a"))
    before = learner.policy.weights
    update = learner.step([_example(1.0, "a")], batch_size=3, seed=0)
    assert not update.accepted
    assert learner.policy.weights == before


def test_replay_is_bounded_and_drift_is_reported():
    buffer = ReplayBuffer(max_size=2)
    buffer.add(_example(1.0, "a"))
    buffer.add(_example(2.0, "a"))
    buffer.add(_example(3.0, "a"))
    assert len(buffer) == 2
    learner = ContinualMetaLearner(["a", "b"], dim=1)
    learner.observe(_example(0.0, "a"))
    update = learner.step([_example(100.0, "a")], batch_size=1)
    assert update.drift > 0


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        MetaExample((), {"a": 1})
    with pytest.raises(ValueError):
        MetaExample((1,), {})
    with pytest.raises(ValueError):
        MetaExample((1,), {"a": float("nan")})
    with pytest.raises(ValueError):
        ReplayBuffer(0)
    with pytest.raises(ValueError):
        ReplayBuffer().sample(1)
    with pytest.raises(ValueError):
        ContinualMetaLearner([], dim=1)
    with pytest.raises(ValueError):
        ContinualMetaLearner(["a"], dim=0)
    learner = ContinualMetaLearner(["a"], dim=1)
    with pytest.raises(ValueError):
        learner.step([])
    with pytest.raises(ValueError):
        learner.policy.act((1,), epsilon=2)
