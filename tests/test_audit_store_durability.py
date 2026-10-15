"""AuditStore keeps its crash-safety contract on the paths that broke it (#274).

The store promises that a process killed mid-write damages at most the last
line and that a reader skips an unparseable line instead of losing the file.
Each test here pins one way that promise used to fail on ``main``.
"""

import builtins
import json
import math
import os

import pytest

from agentdescent.audit.records import AuditRecord, Purpose
from agentdescent.audit.store import (
    MOMENTS_KIND,
    PRIORITY_KIND,
    AuditStore,
)


def _record(record_id="r1", output="answer", score=0.7):
    return AuditRecord(
        record_id=record_id, task_id="t0", artifact_signature="sig",
        output=output, verifier_version="v1", verifier_score=score,
        inclusion_prob=0.5, purpose=Purpose.CALIBRATION, dispatched_at=1.0)


def _path(tmp_path):
    return str(tmp_path / "audit.jsonl")


# -- torn tails -----------------------------------------------------------------


def test_a_torn_tail_does_not_swallow_the_next_append(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    store.append(_record("r2"))
    with open(path, "rb+") as fh:                  # killed mid-way through r2
        fh.truncate(os.path.getsize(path) - 20)

    store = AuditStore(path)
    assert [r.record_id for r in store] == ["r1"] and store.corrupt == 1
    store.append(_record("r3"))

    reread = AuditStore(path)
    assert [r.record_id for r in reread] == ["r1", "r3"]
    assert reread.corrupt == 1
    assert reread.get("r3") is not None


def test_a_torn_tail_does_not_swallow_a_resolution_either(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"record_id": "r2", "task_')     # torn, no newline
    store = AuditStore(path)
    assert store.resolve("r1", 0.4, at=2.0)
    reread = AuditStore(path)
    assert reread.get("r1").oracle_score == 0.4
    assert reread.corrupt == 1


def test_a_tear_inside_a_multibyte_character_costs_one_line(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1", output="答案"))
    store.append(_record("r2", output="答案"))
    raw = open(path, "rb").read()
    cut = raw.rindex("答".encode("utf-8")) + 1     # inside the code point
    with open(path, "wb") as fh:
        fh.write(raw[:cut])

    reread = AuditStore(path)                      # must open, not raise
    assert [r.record_id for r in reread] == ["r1"]
    assert reread.get("r1").output == "答案"
    assert reread.corrupt == 1


# -- parseable lines that are not records ---------------------------------------


@pytest.mark.parametrize("line", [
    "123",
    "[]",
    '"text"',
    "null",
    json.dumps({"kind": PRIORITY_KIND}),
    json.dumps({"kind": PRIORITY_KIND, "priorities": [1, 2]}),
    json.dumps({"kind": MOMENTS_KIND}),
    json.dumps({"kind": MOMENTS_KIND, "verifier_version": "v1",
                "stratum": "all", "moments": {"n": 1}}),
])
def test_a_parseable_non_record_line_is_counted_not_raised(tmp_path, line):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    store.append(_record("r2"))

    reread = AuditStore(path)
    assert [r.record_id for r in reread] == ["r1", "r2"]
    assert reread.corrupt == 1


def test_a_current_schema_record_with_bad_fields_is_counted(tmp_path):
    path = _path(tmp_path)
    good = _record("r1").to_dict()
    bad = dict(good, record_id="r2", purpose="not-a-purpose")
    missing = {k: v for k, v in good.items() if k != "task_id"}
    with open(path, "w", encoding="utf-8") as fh:
        for payload in (good, bad, missing):
            fh.write(json.dumps(payload) + "\n")
    reread = AuditStore(path)
    assert [r.record_id for r in reread] == ["r1"]
    assert reread.corrupt == 2


def test_a_foreign_schema_version_still_raises(tmp_path):
    path = _path(tmp_path)
    payload = dict(_record().to_dict(), schema_version=99)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        AuditStore(path)


# -- memory never runs ahead of disk --------------------------------------------


def _failing_open(monkeypatch, path):
    real_open = builtins.open

    def fake_open(file, mode="r", *a, **kw):
        if os.fspath(file) == path and any(c in mode for c in "aw+"):
            raise PermissionError("simulated read-only store")
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_a_failed_append_leaves_memory_agreeing_with_disk(tmp_path, monkeypatch):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    _failing_open(monkeypatch, path)
    with pytest.raises(PermissionError):
        store.append(_record("r2"))
    monkeypatch.undo()
    assert len(store) == len(AuditStore(path)) == 1
    assert store.get("r2") is None


def test_a_failed_resolve_and_reopen_leave_memory_agreeing_with_disk(
        tmp_path, monkeypatch):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    _failing_open(monkeypatch, path)
    with pytest.raises(PermissionError):
        store.resolve("r1", 0.3, at=2.0)
    monkeypatch.undo()
    assert not store.get("r1").resolved
    assert not AuditStore(path).get("r1").resolved

    assert store.resolve("r1", 0.3, at=2.0)
    _failing_open(monkeypatch, path)
    with pytest.raises(PermissionError):
        store.reopen("r1")
    monkeypatch.undo()
    assert store.get("r1").oracle_score == 0.3
    assert AuditStore(path).get("r1").oracle_score == 0.3


def test_an_unencodable_record_raises_with_memory_and_disk_unchanged(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    size = os.path.getsize(path)
    with pytest.raises(ValueError):
        store.append(_record("r2", score=float("nan")))
    assert store.get("r2") is None and len(store) == 1
    assert os.path.getsize(path) == size


def test_failed_side_record_writes_leave_memory_unchanged(tmp_path, monkeypatch):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.remember_priorities({"sig": 1.0})
    store.remember_fingerprint("fp-1")
    _failing_open(monkeypatch, path)
    with pytest.raises(PermissionError):
        store.remember_priorities({"other": 2.0})
    with pytest.raises(PermissionError):
        store.remember_fingerprint("fp-2")
    monkeypatch.undo()
    assert store.priorities == {"sig": 1.0}
    assert store.last_fingerprint == "fp-1"
    with pytest.raises(ValueError):
        store.remember_priorities({"nan": float("nan")})
    assert store.priorities == {"sig": 1.0}


# -- non-finite scores ----------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_unlabelled_score_is_refused(tmp_path, bad):
    store = AuditStore(_path(tmp_path))
    store.observe_unlabelled("v1", "all", 0.2)
    store.observe_unlabelled("v1", "all", 0.6)
    with pytest.raises(ValueError, match="finite"):
        store.observe_unlabelled("v1", "all", bad)
    m = store.unlabelled_moments("v1")["all"]
    assert m["n"] == 2 and math.isclose(m["mean"], 0.4)
    assert math.isfinite(m["var"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_oracle_score_is_refused(tmp_path, bad):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    with pytest.raises(ValueError, match="finite"):
        store.resolve("r1", bad)
    assert not store.get("r1").resolved
    assert not AuditStore(path).get("r1").resolved


def test_the_file_never_holds_nan_or_infinity_tokens(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            store.append(_record("x", score=bad))
        with pytest.raises(ValueError):
            store.observe_unlabelled("v1", "all", bad)
        with pytest.raises(ValueError):
            store.resolve("r1", bad)
    store.observe_unlabelled("v1", "all", 0.5)
    store.remember_fingerprint("fp")
    store.flush()
    text = open(path, encoding="utf-8").read()
    assert "NaN" not in text and "Infinity" not in text
    for line in text.splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(c))


def test_moments_and_side_records_still_round_trip(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    store.observe_unlabelled("v1", "all", 0.2)
    store.observe_unlabelled("v1", "all", 0.6)
    store.observe_skipped("v1", "all")
    store.remember_priorities({"sig": 0.9})
    store.remember_fingerprint("fp")
    store.flush()
    reread = AuditStore(path)
    assert reread.corrupt == 0
    assert reread.unlabelled_moments("v1")["all"]["n"] == 2
    assert reread.unlabelled_moments("v1")["all"]["skipped"] == 1
    assert reread.priorities == {"sig": 0.9}
    assert reread.last_fingerprint == "fp"


# -- malformed record ids ----------------------------------------------------------


@pytest.mark.parametrize("bad_id", [[], {}, 7, ""])
def test_a_malformed_record_id_is_one_corrupt_line_not_an_unopenable_store(
        tmp_path, bad_id):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    good = _record("r2").to_dict()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(good, record_id=bad_id)) + "\n")
    store.append(_record("r3"))

    reread = AuditStore(path)
    assert [r.record_id for r in reread] == ["r1", "r3"]
    assert reread.corrupt == 1


@pytest.mark.parametrize("bad_id", [[], {}, 7, ""])
def test_a_malformed_record_id_cannot_be_constructed_or_appended(
        tmp_path, bad_id):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    before = open(path, "rb").read()
    with pytest.raises(TypeError):
        _record(bad_id)
    assert open(path, "rb").read() == before
    assert [r.record_id for r in AuditStore(path)] == ["r1"]


# -- legacy non-finite lines and batch resolution (Codex review on #277) -------


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_a_legacy_non_finite_record_line_is_counted_not_loaded(tmp_path, token):
    # Older builds wrote with allow_nan=True, so these tokens exist on disk.
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    line = json.dumps(_record("r2").to_dict()).replace(
        '"verifier_score": 0.7', f'"verifier_score": {token}')
    assert token in line
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    store.append(_record("r3"))

    reread = AuditStore(path)
    assert [r.record_id for r in reread] == ["r1", "r3"]
    assert reread.corrupt == 1
    assert all(math.isfinite(r.verifier_score) for r in reread)


def test_a_legacy_non_finite_resolution_leaves_the_record_pending(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.append(_record("r1"))
    resolved = dict(_record("r1").to_dict(), resolved_at=2.0)
    line = json.dumps(dict(resolved, oracle_score=0.0)).replace(
        '"oracle_score": 0.0', '"oracle_score": NaN')
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    reread = AuditStore(path)
    assert reread.corrupt == 1
    assert not reread.get("r1").resolved
    assert reread.resolve("r1", 1.0)


def test_legacy_non_finite_moments_do_not_reach_the_estimator(tmp_path):
    path = _path(tmp_path)
    store = AuditStore(path)
    store.observe_unlabelled("v1", "all", 0.2)
    store.observe_unlabelled("v1", "all", 0.6)
    store.flush()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": MOMENTS_KIND, "verifier_version": "v1",
                             "stratum": "all",
                             "moments": {"n": 3, "mean": float("nan"),
                                         "m2": 0.0}}) + "\n")
    reread = AuditStore(path)
    assert reread.corrupt == 1
    m = reread.unlabelled_moments("v1")["all"]
    assert m["n"] == 2 and math.isclose(m["mean"], 0.4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "nan", "oops", None])
def test_a_non_finite_batch_answer_refuses_the_batch_not_counted_as_resolved(
        tmp_path, bad):
    from agentdescent.audit.sources import resolve_from_mapping

    path = _path(tmp_path)
    store = AuditStore(path)
    for rid in ("r1", "r2", "r3"):
        store.append(_record(rid))
    before = open(path, "rb").read()
    with pytest.raises(ValueError, match="r2"):
        resolve_from_mapping(store, {"r1": 1.0, "r2": bad, "r3": 0.0})
    assert open(path, "rb").read() == before
    assert len(store.pending()) == 3


def test_a_replayed_batch_still_counts_duplicates(tmp_path):
    from agentdescent.audit.sources import resolve_from_mapping

    store = AuditStore(_path(tmp_path))
    for rid in ("r1", "r2"):
        store.append(_record(rid))
    first = resolve_from_mapping(store, {"r1": 1.0})
    assert first == 1 and first.already_resolved == 0
    again = resolve_from_mapping(store, {"r1": 0.0, "r2": "1", "nope": 1.0})
    assert again == 1 and again.already_resolved == 1
    assert store.get("r1").oracle_score == 1.0
    assert store.get("r2").oracle_score == 1.0
