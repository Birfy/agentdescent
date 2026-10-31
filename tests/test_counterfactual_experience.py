"""Counterfactual credit must become replayable experience."""

import pytest

from agentdescent.counterfactual import ReplayLab, ReplayTrace, TraceStep
from agentdescent.counterfactual_experience import CounterfactualExperience
from agentdescent.continual_meta import ContinualMetaLearner
from agentdescent.experience import ExperienceEngine, ExperienceStore


def _trace():
    return ReplayTrace("trace-1", (
        TraceStep("e1", "edit", payload=1.0),
        TraceStep("e2", "edit", payload=2.0),
    ), score=3.0)


def _engine(actions=("e1", "e2")):
    return ExperienceEngine(
        ContinualMetaLearner(list(actions), dim=2, replay_size=8),
        ExperienceStore())


def _bridge(engine=None, *, action_of=None, **kwargs):
    return CounterfactualExperience(
        ReplayLab(lambda steps: sum(step.payload for step in steps)),
        engine or _engine(), action_of or (lambda step: step.step_id), **kwargs)


def test_credit_becomes_an_experience_record():
    bridge = _bridge()
    report = bridge.ingest(_trace(), (0.2, 0.8), accepted=True, cost=1.0,
                           artifact_id="skill-1")
    assert report.credit.gain == pytest.approx(3.0)
    assert report.record.rewards["e1"] == pytest.approx(1.0)
    assert report.record.rewards["e2"] == pytest.approx(2.0)
    assert report.record.run_id == "trace-1"
    assert report.record.artifact_id == "skill-1"
    assert report.observed == 1
    assert bridge.engine.report().learner.replay == 1
    assert report.to_dict()["credit"]["gain"] == pytest.approx(3.0)


def test_action_mapping_aggregates_and_validates():
    engine = _engine(actions=("edit",))
    bridge = _bridge(engine, action_of=lambda _step: "edit")
    report = bridge.ingest(_trace(), (0.2, 0.8))
    assert report.record.rewards["edit"] == pytest.approx(3.0)

    bad = _bridge(_engine(actions=("direct",)))
    with pytest.raises(ValueError):
        bad.ingest(_trace(), (0.2, 0.8))


def test_metadata_cannot_override_provenance_and_min_credit_is_opt_in():
    bridge = _bridge()
    report = bridge.ingest(
        _trace(), (0.2, 0.8),
        metadata={"shapley_method": "fake", "note": "caller"})
    assert report.record.metadata["shapley_method"] == "shapley"
    assert report.record.metadata["note"] == "caller"
    assert report.record.metadata["samples"] >= 1

    filtered = _bridge(min_credit=5.0)
    with pytest.raises(ValueError):
        filtered.ingest(_trace(), (0.2, 0.8))


def test_invalid_inputs_and_feature_dimension():
    with pytest.raises(ValueError):
        _bridge(min_credit=float("nan"))
    with pytest.raises(ValueError):
        _bridge(permutations=0)
    with pytest.raises(TypeError):
        CounterfactualExperience(
            ReplayLab(lambda _s: 0.0), _engine(), None)
    with pytest.raises(TypeError):
        _bridge().ingest("trace", (0.2, 0.8))
    with pytest.raises(ValueError):
        _bridge().ingest(ReplayTrace("empty", ()), (0.2, 0.8))
    with pytest.raises(ValueError):
        _bridge().ingest(_trace(), (0.2,))
    with pytest.raises(TypeError):
        CounterfactualExperience(object(), _engine(), lambda step: step.step_id)
