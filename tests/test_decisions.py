"""A decision record must survive the log and stay linkable."""

from __future__ import annotations

import json

from agentdescent import DecisionLog, DecisionRecord


def test_record_append_get_children_and_count(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    parent = log.record(kind="session", artifact_id="skill", accepted=True,
                        evidence={"final_reward": 0.8})
    child = log.record(kind="adoption", artifact_id="skill", diff_id="d1",
                       accepted=False, category="leakage:gold",
                       evidence={"leaked": ["321"]},
                       cost={"calls": 12}, parent_id=parent.record_id)
    assert log.count() == 2
    assert log.get(parent.record_id).record_id == parent.record_id
    assert [r.record_id for r in log.children(parent.record_id)] == [child.record_id]


def test_non_finite_values_are_serialized_as_null(tmp_path):
    log = DecisionLog(str(tmp_path / "d.jsonl"))
    log.record(kind="adoption", artifact_id="a", accepted=False,
               evidence={"gain": float("nan")})
    payload = json.loads((tmp_path / "d.jsonl").read_text().strip())
    assert payload["evidence"]["gain"] is None


def test_a_malformed_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"record_id":"x","kind":"k","artifact_id":"a",'
                    '"diff_id":"","accepted":true}\nnot json\n')
    records = list(DecisionLog(str(path)))
    assert len(records) == 1 and records[0].record_id == "x"


def test_schema_roundtrip():
    record = DecisionRecord(record_id="r", kind="adoption", artifact_id="a",
                            diff_id="d", accepted=False, category="c",
                            detail="why", evidence={"x": 1}, cost={"calls": 2},
                            parent_id="p", created_at=1.5)
    assert DecisionRecord.from_dict(record.to_dict()) == record


def test_an_unknown_record_is_none(tmp_path):
    assert DecisionLog(str(tmp_path / "d.jsonl")).get("nope") is None


def test_json_lines_that_are_not_objects_are_skipped(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("123\nnull\n[]\n\"str\"\n"
                    '{"record_id":"x","kind":"k","artifact_id":"a",'
                    '"diff_id":"","accepted":true}\n')
    log = DecisionLog(str(path))
    assert [r.record_id for r in log] == ["x"]
    assert log.count() == 1


def test_a_torn_last_line_does_not_eat_the_next_record(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"record_id":"x","kind"')
    log = DecisionLog(str(path))
    record = log.record(kind="session", artifact_id="a", accepted=True)
    assert log.get(record.record_id) == record
    assert log.count() == 1


def test_duplicate_record_ids_are_refused(tmp_path):
    import pytest
    log = DecisionLog(str(tmp_path / "d.jsonl"))
    log.record(kind="session", artifact_id="a", accepted=True,
               record_id="same")
    with pytest.raises(ValueError, match="duplicate record_id"):
        log.record(kind="session", artifact_id="a", accepted=True,
                   record_id="same")


def test_parent_id_is_normalized_on_write(tmp_path):
    log = DecisionLog(str(tmp_path / "d.jsonl"))
    record = log.record(kind="session", artifact_id="a", accepted=True,
                        parent_id=5)
    assert record.parent_id == "5"
    assert log.children("5")[0].record_id == record.record_id


def test_two_logs_on_one_path_share_a_lock(tmp_path):
    first = DecisionLog(str(tmp_path / "d.jsonl"))
    second = DecisionLog(str(tmp_path / "d.jsonl"))
    assert first._lock is second._lock
