"""A preference model must choose which experience is worth replaying."""

import pytest

from agentdescent.continual_meta import ContinualMetaLearner
from agentdescent.experience import ExperienceEngine, ExperienceRecord, ExperienceStore
from agentdescent.preference import PreferenceModel, PreferencePair
from agentdescent.preference_replay import PreferenceReplay


def _model():
    model = PreferenceModel()
    model.fit([
        PreferencePair((1.0, 0.0), (0.0, 1.0), "left"),
        PreferencePair((0.8, 0.2), (0.2, 0.8), "left"),
        PreferencePair((1.0, 0.0), (0.5, 0.5), "left"),
    ])
    return model


def _store(path=None):
    store = ExperienceStore(path)
    store.add(ExperienceRecord("bad", (0.0, 1.0), {"a": 0.2}))
    store.add(ExperienceRecord("good", (1.0, 0.0), {"a": 1.0}))
    store.add(ExperienceRecord("mid", (0.5, 0.5), {"a": 0.5}))
    return store


def _engine(store=None):
    return ExperienceEngine(
        ContinualMetaLearner(["a"], dim=2, replay_size=8),
        store if store is not None else ExperienceStore())


def test_preferred_records_are_ranked_and_replayed_without_duplicates():
    store = _store()
    replay = PreferenceReplay(_model(), store)
    assert [row.run_id for row in replay.rank()] == ["good", "mid", "bad"]
    assert [row.run_id for row in replay.rank(k=2)] == ["good", "mid"]

    engine = _engine()
    report = replay.replay(engine, k=2)
    assert [row.run_id for row in report.selected] == ["good", "mid"]
    assert report.ingested == ("good", "mid")
    assert len(engine.store) == 2
    assert report.scores["good"] > report.scores["mid"]
    assert replay.replay(engine, k=2).ingested == ()
    assert report.to_dict()["ingested"] == ["good", "mid"]


def test_mixed_dimensions_are_skipped_and_same_store_is_refused(tmp_path):
    store = _store()
    store.add(ExperienceRecord("odd", (1.0, 0.0, 0.0), {"a": 1.0}))
    replay = PreferenceReplay(_model(), store)
    assert [row.run_id for row in replay.rank()] == ["good", "mid", "bad"]
    report = replay.replay(_engine(), k=3)
    assert report.skipped == 1
    assert "1 skipped" in report.detail

    same = ExperienceStore()
    with pytest.raises(ValueError):
        PreferenceReplay(_model(), same).replay(_engine(same))


def test_unfitted_model_and_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        PreferenceReplay(PreferenceModel(), ExperienceStore())
    with pytest.raises(ValueError):
        PreferenceReplay(_model(), _store()).rank(k=0)
    with pytest.raises(TypeError):
        PreferenceReplay(object(), ExperienceStore())
    with pytest.raises(TypeError):
        PreferenceReplay(_model(), object())
    with pytest.raises(TypeError):
        PreferenceReplay(_model(), _store(), featurize="x")
    assert [row.run_id for row in PreferenceReplay(_model(), _store()).rank(k=99)] \
        == ["good", "mid", "bad"]


def test_persisted_store_does_not_double_append(tmp_path):
    path = tmp_path / "experience.jsonl"
    store = _store(str(path))
    replay = PreferenceReplay(_model(), store)
    report = replay.replay(_engine(), k=2)
    assert report.ingested == ("good", "mid")
    reloaded = ExperienceStore(str(path))
    assert len(reloaded) == 3
