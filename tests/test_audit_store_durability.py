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


# -- schema v1 → v2 / scorer_version (#315) ------------------------------------


def test_unstamped_new_records_write_schema_version_2():
    """Current writers stamp SCHEMA_VERSION even when scorer_version is omitted."""
    from agentdescent.audit.records import SCHEMA_VERSION
    payload = _record().to_dict()
    assert SCHEMA_VERSION == 2
    assert payload["schema_version"] == 2
    assert "scorer_version" not in payload


def test_stamped_records_write_schema_version_2_with_scorer():
    from agentdescent.audit.records import SCHEMA_VERSION, AuditRecord, Purpose
    rec = AuditRecord(
        record_id="s1", task_id="t0", artifact_signature="sig",
        output="out", verifier_version="v1+boost", verifier_score=0.5,
        inclusion_prob=0.5, purpose=Purpose.CALIBRATION, dispatched_at=1.0,
        scorer_version="scorer-alone")
    payload = rec.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION == 2
    assert payload["scorer_version"] == "scorer-alone"
    assert AuditRecord.from_dict(payload) == rec


def test_v1_records_migrate_and_preserve_pending_state(tmp_path):
    """A new reader loads pre-scorer v1 lines (pending and resolved) as v2."""
    from agentdescent.audit.records import SCHEMA_VERSION, AuditRecord, Purpose
    path = _path(tmp_path)
    pending = {
        "record_id": "p1", "task_id": "t0", "artifact_signature": "sig",
        "output": "ans", "verifier_version": "v1", "verifier_score": 0.7,
        "inclusion_prob": 0.5, "purpose": "calibration", "stratum": "all",
        "oracle_score": None, "dispatched_at": 1.0, "resolved_at": None,
        "sampler_seed": 0, "schema_version": 1,
    }
    resolved = dict(pending, record_id="r1", oracle_score=0.4, resolved_at=2.0)
    # Mistaken early stamp: scorer_version under schema_version=1 (the #315 bug).
    stamped_as_v1 = dict(pending, record_id="buggy", scorer_version="scorer")
    with open(path, "w", encoding="utf-8") as fh:
        for payload in (pending, resolved, stamped_as_v1):
            fh.write(json.dumps(payload) + "\n")

    store = AuditStore(path)
    assert store.corrupt == 0
    assert [r.record_id for r in store] == ["p1", "r1", "buggy"]
    p1 = store.get("p1")
    assert p1.oracle_score is None and p1.scorer_version == ""
    assert p1.schema_version == SCHEMA_VERSION
    r1 = store.get("r1")
    assert r1.oracle_score == 0.4 and r1.resolved_at == 2.0
    buggy = store.get("buggy")
    assert buggy.scorer_version == "scorer"
    assert buggy.schema_version == SCHEMA_VERSION
    # Re-persisting a migrated pending record must write v2, so an old reader
    # would hit its unsupported-version raise rather than TypeError-skip.
    rewritten = buggy.to_dict()
    assert rewritten["schema_version"] == 2
    assert rewritten["scorer_version"] == "scorer"


def test_mixed_v1_and_v2_lines_load_without_rewriting_bytes(tmp_path):
    path = _path(tmp_path)
    v1 = dict(_record("old").to_dict(), schema_version=1)
    v1.pop("scorer_version", None)
    v2 = _record("new", score=0.9)
    # Force a stamped v2 line onto disk.
    from agentdescent.audit.records import AuditRecord, Purpose
    stamped = AuditRecord(
        record_id="boosted", task_id="t0", artifact_signature="sig",
        output="answer", verifier_version="v1+b", verifier_score=0.5,
        inclusion_prob=0.5, purpose=Purpose.IMPROVEMENT, dispatched_at=1.0,
        scorer_version="scorer")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(v1) + "\n")
        fh.write(json.dumps(v2.to_dict()) + "\n")
        fh.write(json.dumps(stamped.to_dict()) + "\n")
    raw_before = open(path, "rb").read()
    store = AuditStore(path)
    assert store.corrupt == 0
    assert [r.record_id for r in store] == ["old", "new", "boosted"]
    assert store.get("boosted").scorer_version == "scorer"
    # load is read-only: original bytes unchanged
    assert open(path, "rb").read() == raw_before


def test_old_reader_contract_rejects_v2_explicitly_not_as_corrupt():
    """Simulate a v1-only from_dict: schema_version=2 must raise ValueError.

    The #315 failure mode was schema_version=1 + unknown ``scorer_version``
    keyword → TypeError → store counted the line corrupt and skipped pending
    work. Bumping the marker makes an unchanged old reader take its
    unsupported-version path instead.
    """
    from agentdescent.audit.records import AuditRecord, Purpose

    def old_v1_from_dict(d):
        d = dict(d)
        version = d.get("schema_version")
        if version != 1:
            raise ValueError(
                f"audit record schema_version={version!r}, this build reads 1. "
                f"Refusing to guess at the difference.")
        d["purpose"] = Purpose(d["purpose"])
        return AuditRecord(**{k: v for k, v in d.items()
                              if k in AuditRecord.__dataclass_fields__
                              and k != "scorer_version"})

    stamped = AuditRecord(
        record_id="s", task_id="t0", artifact_signature="sig",
        output="o", verifier_version="v", verifier_score=0.5,
        inclusion_prob=0.5, purpose=Purpose.CALIBRATION, dispatched_at=1.0,
        scorer_version="scorer")
    payload = stamped.to_dict()
    assert payload["schema_version"] == 2
    with pytest.raises(ValueError, match="schema_version"):
        old_v1_from_dict(payload)

    # Contrast: the pre-fix shape (v1 + scorer_version) is what caused the
    # silent TypeError skip on a real v0.6.0 constructor that lacked the field.
    broken = dict(payload, schema_version=1)
    try:
        AuditRecord(
            **{k: (Purpose(v) if k == "purpose" else v)
               for k, v in broken.items()
               if k in AuditRecord.__dataclass_fields__})
        # Current constructor accepts scorer_version; emulate rejection:
        raise AssertionError("current build accepts scorer_version")
    except AssertionError:
        pass
    # Document the TypeError an old constructor would raise:
    import dataclasses
    fields_without = tuple(
        f for f in dataclasses.fields(AuditRecord) if f.name != "scorer_version")

    @dataclasses.dataclass(frozen=True)
    class OldAuditRecord:
        pass
    # Cheaper pin: constructing with an unexpected kwarg raises TypeError.
    def old_ctor(**kwargs):
        known = {"record_id", "task_id", "artifact_signature", "output",
                 "verifier_version", "verifier_score", "inclusion_prob",
                 "purpose", "stratum", "oracle_score", "dispatched_at",
                 "resolved_at", "sampler_seed", "schema_version"}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(
                f"__init__() got an unexpected keyword argument "
                f"{unknown.pop()!r}")
        return True
    with pytest.raises(TypeError, match="scorer_version"):
        old_ctor(**{**broken, "purpose": "calibration"})


@pytest.mark.parametrize("drop", ["purpose", "task_id", "verifier_score"])
def test_a_malformed_v1_line_is_one_corrupt_line_not_an_unopenable_store(
        tmp_path, drop):
    """A v1 line that parses but cannot construct a record is skipped and
    counted, exactly like a malformed current-schema line (Codex P1 on #324)."""
    path = _path(tmp_path)
    bad = dict(_record("bad").to_dict(), schema_version=1)
    bad.pop(drop)
    good = _record("good")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(bad) + "\n")
        fh.write(json.dumps(good.to_dict()) + "\n")
    store = AuditStore(path)
    assert store.corrupt == 1
    assert [r.record_id for r in store] == ["good"]


def test_a_v1_line_with_an_invalid_inclusion_prob_is_counted_corrupt(tmp_path):
    path = _path(tmp_path)
    bad = dict(_record("bad").to_dict(), schema_version=1, inclusion_prob=0.0)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(bad) + "\n")
    store = AuditStore(path)
    assert store.corrupt == 1
    assert list(store) == []


@pytest.mark.parametrize("version", [1, 2])
def test_an_unknown_field_is_corrupt_not_silently_defaulted(tmp_path, version):
    """A misspelt optional key must not load under the default value (Codex P2
    on #324): ``strtaum`` would otherwise land the record in stratum "all"."""
    path = _path(tmp_path)
    bad = _record("typo").to_dict()
    bad["schema_version"] = version
    bad["strtaum"] = bad.pop("stratum")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(bad) + "\n")
    store = AuditStore(path)
    assert store.corrupt == 1
    assert list(store) == []


def test_an_unknown_schema_version_still_raises(tmp_path):
    path = _path(tmp_path)
    future = dict(_record("future").to_dict(), schema_version=3)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(future) + "\n")
    with pytest.raises(ValueError, match="schema_version=3"):
        AuditStore(path)
