"""The merge path's ranking, reaching the person who works the queue.

`AuditScheduler` has ranked every merge decision since the beginning and nothing
has ever popped its heap. `test_the_ranking_only_matters_when_the_oracle_is_dear`
records why that was right, and the rest of this file is the wiring for the case
where it stops being right.
"""

import heapq
import uuid

import pytest

from agentdescent.audit import AuditRecord, AuditStore, Purpose
from agentdescent.audit.queue import DrainReport, drain, prioritise
from agentdescent.audit.service import audit_pending
from agentdescent.scheduler import AuditScheduler


class _Diff:
    def __init__(self, target):
        self.target = target


def _scheduler(*submissions, collect=True):
    sched = AuditScheduler(collect=collect)
    for diff_id, artifact_id, radius, uncertainty in submissions:
        sched.submit(diff_id, artifact_id, radius, uncertainty,
                     payload=_Diff(artifact_id))
    return sched


def _by_payload_target(item):
    return getattr(item.payload, "target", None)


def _rec(sig, *, record_id=None, at=0.0):
    return AuditRecord(
        record_id=record_id or uuid.uuid4().hex, task_id=uuid.uuid4().hex,
        artifact_signature=sig, output="o", verifier_version="v1",
        verifier_score=1.0, inclusion_prob=1.0, purpose=Purpose.CALIBRATION,
        dispatched_at=at)


# -- the drain ---------------------------------------------------------------

def test_the_queue_comes_off_as_signature_to_priority():
    sched = _scheduler(("d1", "sig-a", 0.6, 0.5), ("d2", "sig-b", 0.2, 0.1))
    got = drain(sched, signature_of=_by_payload_target)
    assert got.popped == 2 and got.unplaced == 0
    assert got.priorities["sig-a"] > got.priorities["sig-b"]
    assert len(sched) == 0, "drained, not copied"


def test_the_priority_reads_the_way_submit_returned_it():
    """The heap stores it negated; what comes back must not be."""
    sched = AuditScheduler(collect=True)
    returned = sched.submit("d1", "a", 0.8, 0.5, payload=_Diff("sig"))
    got = drain(sched, signature_of=_by_payload_target)
    assert got.priorities["sig"] == pytest.approx(returned)
    assert returned > 0


def test_the_highest_priority_wins_not_the_latest():
    """One alarming merge must not be buried by a run of routine ones."""
    sched = _scheduler(("d1", "sig", 0.9, 0.9), ("d2", "sig", 0.01, 0.01),
                       ("d3", "sig", 0.01, 0.01), ("d4", "sig", 0.01, 0.01))
    got = drain(sched, signature_of=_by_payload_target)
    assert got.priorities["sig"] == pytest.approx(0.81)


def test_a_diff_whose_signature_cannot_be_resolved_is_counted_not_guessed():
    """The scheduler ranks diffs and the tap records signatures; nothing in
    either knows about the other, so a caller that cannot join them has a blind
    spot whose size it should be told."""
    sched = _scheduler(("d1", "sig-a", 0.6, 0.5), ("d2", "sig-b", 0.4, 0.4))
    got = drain(sched, signature_of=lambda item: None)
    assert got.popped == 2 and got.unplaced == 2 and got.placed == 0
    assert got.priorities == {}
    assert set(got.examples) == {"d1", "d2"}


def test_without_a_join_the_queue_is_still_emptied():
    sched = _scheduler(("d1", "a", 0.6, 0.5))
    got = drain(sched)
    assert got.popped == 1 and got.unplaced == 1 and len(sched) == 0


def test_a_limit_leaves_the_rest_on_the_queue():
    sched = _scheduler(*[(f"d{i}", f"sig{i}", 0.5, 0.5) for i in range(5)])
    got = drain(sched, signature_of=_by_payload_target, limit=2)
    assert got.popped == 2 and len(sched) == 3


def test_the_limit_takes_the_highest_priorities_first():
    sched = _scheduler(("low", "sig-low", 0.1, 0.1), ("high", "sig-high", 0.9, 0.9),
                       ("mid", "sig-mid", 0.5, 0.5))
    got = drain(sched, signature_of=_by_payload_target, limit=1)
    assert list(got.priorities) == ["sig-high"]


def test_an_empty_queue_drains_to_nothing():
    got = drain(AuditScheduler(collect=True), signature_of=_by_payload_target)
    assert got == DrainReport()


def test_a_scheduler_that_was_not_collecting_has_nothing_to_drain():
    """`collect=False` is the default and computes priorities without queueing.

    Draining it is not an error -- it is the shipped configuration -- and it
    comes back empty rather than pretending.
    """
    sched = _scheduler(("d1", "a", 0.9, 0.9), collect=False)
    assert drain(sched, signature_of=_by_payload_target).popped == 0


def test_the_ranking_only_matters_when_the_oracle_is_dear():
    """Why the queue sat unread, kept as a test of the docstring's claim.

    `force_oracle` is a threshold, and on the shipped verifier an audit is free:
    `full_eval` measures the same held-out set the acceptance test just did, so
    every merge past the threshold gets one and a ranking has nothing to do.
    """
    from agentdescent.verifier import ThreeLayerVerifier

    verifier = ThreeLayerVerifier(lambda artifact, tasks: 1.0, ["t1", "t2"])
    assert verifier.full_eval_matches_counts, (
        "the audit reuses a measurement already paid for -- so the ranking is "
        "idle until the oracle costs something, which is what this package is "
        "for")


# -- ordering a queue --------------------------------------------------------

def test_pending_units_come_back_worst_first():
    rows = [_rec("calm", at=0.0), _rec("risky", at=1.0), _rec("calm", at=2.0)]
    got = prioritise(rows, {"risky": 0.8, "calm": 0.1})
    assert [r.artifact_signature for r in got] == ["risky", "calm", "calm"]


def test_equal_priorities_keep_the_order_someone_was_working_in():
    """A queue must not reshuffle under a person when a new merge lands."""
    rows = [_rec("a", record_id=f"r{i}", at=float(i)) for i in range(4)]
    got = prioritise(rows, {"a": 0.5})
    assert [r.record_id for r in got] == ["r0", "r1", "r2", "r3"]


def test_an_unranked_signature_sorts_after_every_ranked_one():
    """The ranking is information the queue has; burying it under units nobody
    ranked would waste it."""
    rows = [_rec("unranked", at=0.0), _rec("ranked", at=1.0)]
    got = prioritise(rows, {"ranked": 0.01})
    assert got[0].artifact_signature == "ranked"


def test_the_default_can_push_unranked_units_up():
    rows = [_rec("unranked", at=0.0), _rec("ranked", at=1.0)]
    got = prioritise(rows, {"ranked": 0.01}, default=1.0)
    assert got[0].artifact_signature == "unranked"


def test_prioritising_nothing_is_not_an_error():
    assert prioritise([], {"a": 1.0}) == []


# -- persistence, which is the point -----------------------------------------

def test_a_ranking_survives_the_process_that_computed_it(tmp_path):
    """The whole reason to persist it: the person who acts on it is elsewhere,
    later, with only the JSONL."""
    path = str(tmp_path / "audit.jsonl")
    store = AuditStore(path)
    store.append(_rec("calm", at=0.0))
    store.append(_rec("risky", at=1.0))
    store.remember_priorities({"risky": 0.8, "calm": 0.05})

    reopened = AuditStore(path)
    assert reopened.priorities == {"risky": 0.8, "calm": 0.05}

    got = audit_pending(path, order="priority")
    assert got["order"] == "priority"
    assert [r["artifact_signature"] for r in got["records"]] == ["risky", "calm"]
    assert got["records"][0]["priority"] == 0.8


def test_the_default_order_is_still_dispatch_order(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    store = AuditStore(path)
    store.append(_rec("calm", at=0.0))
    store.append(_rec("risky", at=1.0))
    store.remember_priorities({"risky": 0.8})
    got = audit_pending(path)
    assert got["order"] == "dispatched"
    assert [r["artifact_signature"] for r in got["records"]] == ["calm", "risky"]


def test_asking_for_priority_with_nothing_drained_says_so(tmp_path):
    """An empty ranking silently reordering nothing looks identical to one that
    was applied, which is the difference between "the queue is ordered" and "the
    queue looks ordered"."""
    path = str(tmp_path / "audit.jsonl")
    AuditStore(path).append(_rec("a"))
    got = audit_pending(path, order="priority")
    assert got["order"] == "dispatched"
    assert "nothing has been drained" in got["note"]


def test_a_later_ranking_supersedes_an_earlier_one(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    store = AuditStore(path)
    store.remember_priorities({"a": 0.1})
    store.remember_priorities({"a": 0.9, "b": 0.2})
    assert AuditStore(path).priorities == {"a": 0.9, "b": 0.2}


def test_an_unknown_order_is_refused_rather_than_ignored(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    AuditStore(path).append(_rec("a"))
    assert "error" in audit_pending(path, order="soonest")


def test_priorities_do_not_disturb_the_records(tmp_path):
    """A snapshot line in the same file must not be read as a record."""
    path = str(tmp_path / "audit.jsonl")
    store = AuditStore(path)
    store.append(_rec("a"))
    store.remember_priorities({"a": 0.5})
    store.append(_rec("b"))
    reopened = AuditStore(path)
    assert len(reopened) == 2 and reopened.corrupt == 0
