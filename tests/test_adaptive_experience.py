"""One adaptive cycle: credit in, preference-selected anchors, guarded update."""

import json

import pytest

from agentdescent.adaptive_experience import (
    AdaptiveExperience,
    AdaptiveExperienceReport,
    ExperiencePlan,
)
from agentdescent.continual_meta import ContinualMetaLearner
from agentdescent.counterfactual import ReplayLab, ReplayTrace, TraceStep
from agentdescent.counterfactual_experience import CounterfactualExperience
from agentdescent.experience import (
    ExperienceEngine,
    ExperienceRecord,
    ExperienceStore,
)
from agentdescent.preference import PreferenceModel, PreferencePair
from agentdescent.preference_replay import PreferenceReplay


def _record(run_id, features, reward=1.0):
    return ExperienceRecord(run_id, features, {"a": reward})


def _model():
    model = PreferenceModel()
    model.fit([
        PreferencePair((1.0, 0.0), (0.0, 1.0), "left"),
        PreferencePair((0.8, 0.2), (0.2, 0.8), "left"),
        PreferencePair((1.0, 0.0), (0.5, 0.5), "left"),
    ])
    return model


def _source_store():
    store = ExperienceStore()
    store.add(_record("bad", (0.0, 1.0), 0.2))
    store.add(_record("good", (1.0, 0.0), 1.0))
    store.add(_record("mid", (0.5, 0.5), 0.5))
    return store


def _engine(store=None, actions=("a",)):
    return ExperienceEngine(
        ContinualMetaLearner(list(actions), dim=2, replay_size=8),
        store if store is not None else ExperienceStore())


def _trace():
    return ReplayTrace("trace-1", (
        TraceStep("e1", "edit", payload=1.0),
        TraceStep("e2", "edit", payload=2.0),
    ), score=3.0)


def _bridge(engine):
    return CounterfactualExperience(
        ReplayLab(lambda steps: sum(step.payload for step in steps)),
        engine, lambda step: step.step_id)


def test_plan_uses_preference_ranked_anchors_and_scores():
    replay = PreferenceReplay(_model(), _source_store())
    adaptive = AdaptiveExperience(_engine(), replay=replay)

    plan = adaptive.plan([_record("current", (1.0, 0.0))], k=2)

    assert [row.run_id for row in plan.anchors] == ["good", "mid"]
    assert plan.source == "preference"
    assert plan.scores["good"] > plan.scores["mid"]
    assert len(plan.validation) == 1
    assert plan.to_dict()["source"] == "preference"


def test_plan_falls_back_to_feature_recall_without_preference():
    store = ExperienceStore()
    store.add(_record("near", (0.0, 0.0)))
    store.add(_record("far", (1.0, 1.0)))
    adaptive = AdaptiveExperience(_engine(store))

    plan = adaptive.plan(
        [_record("current", (0.05, 0.05))], k=1, features=(0.05, 0.05))

    assert plan.source == "recall"
    assert [row.run_id for row in plan.anchors] == ["near"]
    assert plan.scores == {}


def test_plan_falls_back_to_recent_history_excluding_current_rows():
    store = ExperienceStore()
    store.add(_record("hist-a", (0.0, 0.0)))
    store.add(_record("hist-b", (0.1, 0.1)))
    store.add(_record("hist-c", (0.2, 0.2)))
    current = _record("current", (0.3, 0.3))
    store.add(current)
    adaptive = AdaptiveExperience(_engine(store))

    plan = adaptive.plan([current], k=2)

    assert plan.source == "history"
    assert [row.run_id for row in plan.anchors] == ["hist-b", "hist-c"]
    assert current.run_id not in [row.run_id for row in plan.anchors]


def test_first_cycle_self_validates_and_updates():
    adaptive = AdaptiveExperience(_engine())
    row = _record("first", (1.0, 0.0))

    plan = adaptive.plan([row])

    assert plan.source == "self-validation"
    assert plan.anchors == (row,)

    report = adaptive.cycle([row])
    assert report.plan.source == "self-validation"
    assert report.plan.validation == (row,)
    assert report.update.version >= 1


def test_cycle_ingests_dedupes_updates_and_serializes():
    store = ExperienceStore()
    known = _record("known", (0.0, 0.0))
    store.add(known)
    adaptive = AdaptiveExperience(_engine(store))
    fresh = _record("fresh", (1.0, 0.0))

    report = adaptive.cycle([known, fresh])

    assert isinstance(report, AdaptiveExperienceReport)
    assert report.ingested == ("fresh",)
    assert report.plan.validation == (fresh,)
    assert len(store) == 2
    payload = report.to_dict()
    assert payload["ingested"] == ["fresh"]
    json.dumps(payload)

    again = adaptive.cycle([known, fresh])
    assert again.ingested == ()


def test_bridge_credit_is_ingested_and_requires_features():
    engine = _engine(actions=("e1", "e2"))
    adaptive = AdaptiveExperience(engine, bridge=_bridge(engine))

    report = adaptive.cycle(trace=_trace(), features=(0.2, 0.8))

    assert report.ingested == ("trace-1",)
    assert report.plan.validation[0].run_id == "trace-1"
    assert set(report.plan.validation[0].rewards) == {"e1", "e2"}
    assert [row.run_id for row in engine.store.rows] == ["trace-1"]
    json.dumps(report.to_dict())

    with pytest.raises(ValueError):
        adaptive.cycle(trace=_trace())


def test_empty_validation_and_invalid_types_are_refused():
    with pytest.raises(TypeError):
        AdaptiveExperience(object())
    with pytest.raises(TypeError):
        AdaptiveExperience(_engine(), replay=object())
    with pytest.raises(TypeError):
        AdaptiveExperience(_engine(), bridge=object())

    adaptive = AdaptiveExperience(_engine())
    with pytest.raises(ValueError):
        adaptive.plan([])
    with pytest.raises(TypeError):
        adaptive.plan([object()])
    with pytest.raises(TypeError):
        adaptive.ingest("record")
    with pytest.raises(TypeError):
        adaptive.ingest([object()])
    with pytest.raises(ValueError):
        adaptive.cycle([], validation=[])
    with pytest.raises(ValueError):
        adaptive.cycle([])
    with pytest.raises(ValueError):
        adaptive.plan([_record("row", (1.0, 0.0))], k=0)


def test_explicit_anchors_are_used_as_given():
    adaptive = AdaptiveExperience(_engine())
    validation = _record("validation", (1.0, 0.0))
    anchor = _record("anchor", (0.0, 1.0))

    report = adaptive.cycle(
        [validation], validation=[validation], anchors=[anchor])

    assert report.plan.source == "explicit"
    assert report.plan.anchors == (anchor,)


def test_plan_dataclass_is_frozen_and_json_safe():
    row = _record("row", (1.0, 0.0))
    plan = ExperiencePlan((row,), (row,), {"row": 0.5}, "self-validation")

    with pytest.raises(Exception):
        plan.source = "changed"
    assert plan.to_dict()["validation"][0]["run_id"] == "row"
    json.dumps(plan.to_dict())
