"""Cross-run experience must recall, update, and guard against forgetting."""

import pytest

from agentdescent.continual_meta import ContinualMetaLearner
from agentdescent.experience import (
    ExperienceEngine,
    ExperienceRecord,
    ExperienceStore,
)


def _learner(*, learning_rate=0.5, min_gain=0.0):
    return ContinualMetaLearner(
        ["a", "b"], dim=1, learning_rate=learning_rate, l2=0.0,
        replay_size=16, min_gain=min_gain, seed=0)


def _record(run_id, *, feature=1.0, rewards=None):
    return ExperienceRecord(
        run_id, (feature,), rewards or {"a": 1.0, "b": 0.0},
        artifact_id=f"artifact-{run_id}")


def test_ingest_and_recall_returns_the_nearest_history():
    store = ExperienceStore()
    engine = ExperienceEngine(_learner(), store, require_anchors=False)
    engine.ingest(_record("near", feature=1.0))
    engine.ingest(_record("mid", feature=2.0))
    engine.ingest(_record("far", feature=10.0))
    hits = engine.recall((1.1,), k=2)
    assert [hit.record.run_id for hit in hits] == ["near", "mid"]
    assert hits[0].distance == pytest.approx(0.1)
    assert engine.report().learner.replay == 3


def test_update_accepts_a_validation_improvement_without_anchor_loss():
    learner = _learner()
    engine = ExperienceEngine(learner, ExperienceStore())
    rows = [_record(f"v{i}") for i in range(3)]
    for row in rows:
        engine.ingest(row)
    update = engine.update(validation=rows, anchors=rows)
    assert update.accepted
    assert not update.rolled_back
    assert update.after > update.before
    assert learner.version == 2
    assert update.anchor_after >= update.anchor_before


def test_forgetting_guard_rolls_back_an_anchor_regression():
    learner = _learner(learning_rate=1.0, min_gain=-1.0)
    engine = ExperienceEngine(learner, ExperienceStore())
    validation = [
        _record(f"v{i}", rewards={"a": 0.0, "b": 1.0})
        for i in range(3)
    ]
    for row in validation:
        engine.ingest(row)
    anchor = _record("anchor", rewards={"a": 1.0, "b": 0.0})
    update = engine.update(validation=validation, anchors=[anchor])
    assert update.rolled_back
    assert not update.accepted
    assert "forgetting guard" in update.reason
    assert update.anchor_after < update.anchor_before
    assert f"{update.anchor_after:.6g}" in update.reason
    assert learner.version == 1
    assert learner.report().updates == 1
    assert learner.report().accepted == 0
    assert learner.report().rejected == 1


def test_anchor_tolerance_can_allow_a_small_regression():
    learner = _learner(learning_rate=1.0, min_gain=-1.0)
    engine = ExperienceEngine(
        learner, ExperienceStore(), anchor_tolerance=1.0)
    validation = [
        _record(f"v{i}", rewards={"a": 0.0, "b": 1.0})
        for i in range(3)
    ]
    for row in validation:
        engine.ingest(row)
    anchor = _record("anchor", rewards={"a": 1.0, "b": 0.0})
    update = engine.update(validation=validation, anchors=[anchor])
    assert update.accepted
    assert not update.rolled_back
    assert learner.version == 2


def test_store_persists_and_skips_corrupt_lines(tmp_path):
    path = tmp_path / "experience.jsonl"
    store = ExperienceStore(str(path))
    store.add(_record("one"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
        handle.write("\n")
    reloaded = ExperienceStore(str(path))
    assert len(reloaded) == 1
    assert reloaded.skipped == 1
    assert reloaded.rows[0].run_id == "one"
    assert reloaded.recall((1.0,), k=1)[0].record.run_id == "one"

    warmed = ExperienceEngine(_learner(), reloaded)
    assert warmed.report().learner.replay == 1
    assert warmed.report().skipped == 1
    assert len(reloaded) == 1


def test_anchor_requirement_and_invalid_records():
    learner = _learner()
    engine = ExperienceEngine(learner, ExperienceStore())
    with pytest.raises(ValueError):
        engine.update(validation=[_record("v")], anchors=[])

    optional = ExperienceEngine(
        learner, ExperienceStore(), require_anchors=False)
    optional.ingest(_record("v"))
    update = optional.update(validation=[_record("v")], anchors=[])
    assert update.anchor_before is None
    assert update.anchor_after is None

    with pytest.raises(ValueError):
        ExperienceRecord("r", (), {"a": 1.0})
    with pytest.raises(ValueError):
        ExperienceRecord("r", (1.0,), {})
    with pytest.raises(ValueError):
        ExperienceRecord("r", (1.0,), {"a": 1.0}, cost=-1)
    with pytest.raises(ValueError):
        ExperienceRecord.from_dict({
            "run_id": "r", "features": [1.0], "rewards": {"a": 1.0},
            "accepted": "false"})
    with pytest.raises(ValueError):
        ExperienceRecord("", (1.0,), {"a": 1.0})
    with pytest.raises(ValueError):
        ExperienceStore().recall((1.0,))
    with pytest.raises(ValueError):
        ExperienceStore().recall((1.0,), k=0)
    with pytest.raises(TypeError):
        ExperienceEngine(object(), ExperienceStore())
    with pytest.raises(ValueError):
        ExperienceEngine(learner, ExperienceStore(), anchor_tolerance=-1)
