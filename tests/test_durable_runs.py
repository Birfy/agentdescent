"""A durable run must survive retries, leases and a process reopen."""

import time

import pytest

from agentdescent.durable_runs import (
    DurableRunService,
    LeaseLost,
    RunStatus,
    SqliteRunStore,
    StepOutcome,
)


def _store(tmp_path):
    return SqliteRunStore(str(tmp_path / "runs.sqlite3"))


def test_submit_claim_heartbeat_and_release(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t1", "w1", spec_hash="abc")
    assert service.snapshot(run_id).status == RunStatus.PENDING
    assert store.list_runs(tenant="t1", workspace="w1") == (run_id,)
    assert store.claim(run_id, "worker-a", ttl=10) is True
    assert store.claim(run_id, "worker-b", ttl=10) is False
    assert store.heartbeat(run_id, "worker-a", ttl=10) is True
    assert store.release(run_id, "worker-a") is True
    assert store.claim(run_id, "worker-b", ttl=10) is True


def test_step_loop_completes_and_completion_is_idempotent(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="spec")
    calls = []

    def step(snapshot):
        calls.append(snapshot.events[-1].seq if snapshot.events else 0)
        if len(calls) < 3:
            return StepOutcome(payload={"n": len(calls)})
        return StepOutcome(done=True, payload={"answer": 42})

    snapshot = service.run_step(run_id, "worker", step, max_steps=10)
    assert snapshot.status == RunStatus.COMPLETED
    assert snapshot.result["answer"] == 42
    assert len(calls) == 3
    completed = [e for e in snapshot.events if e.kind == "completed"]
    assert len(completed) == 1
    first = store.append_event(run_id, "completed", payload={"again": True},
                               idempotency_key="completed")
    assert first.seq == completed[0].seq
    assert len([e for e in store.snapshot(run_id).events
                if e.kind == "completed"]) == 1


def test_retryable_step_errors_are_retried_then_completed(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="s")
    attempts = {"n": 0}

    def flaky(_snapshot):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return StepOutcome(done=True)

    snapshot = service.run_step(run_id, "worker", flaky, max_retries=3)
    assert snapshot.status == RunStatus.COMPLETED
    assert attempts["n"] == 3
    assert snapshot.attempts == 2


def test_non_retryable_and_exhausted_retries_fail(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="s")

    def hard(_snapshot):
        return StepOutcome(error="bad input", retryable=False)

    snapshot = service.run_step(run_id, "worker", hard, max_retries=3)
    assert snapshot.status == RunStatus.FAILED
    assert "bad input" in snapshot.error

    run2 = service.submit("t", "w", spec_hash="s2")

    def always_fail(_snapshot):
        raise RuntimeError("nope")

    snapshot = service.run_step(run2, "worker", always_fail, max_retries=2)
    assert snapshot.status == RunStatus.FAILED
    assert snapshot.attempts == 3


def test_expired_lease_can_be_recovered(tmp_path):
    store = _store(tmp_path)
    run_id = store.create("t", "w", "s", now=100)
    assert store.claim(run_id, "dead", ttl=5, now=100)
    assert store.recover_expired(now=120) == (run_id,)
    assert store.claim(run_id, "alive", ttl=5, now=120)


def test_cancel_is_terminal_and_persistent_across_reopen(tmp_path):
    path = str(tmp_path / "runs.sqlite3")
    store = SqliteRunStore(path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="s")
    service.cancel(run_id, reason="stop")
    assert service.snapshot(run_id).status == RunStatus.CANCELLED
    store.close()

    reopened = SqliteRunStore(path)
    snapshot = reopened.snapshot(run_id)
    assert snapshot.status == RunStatus.CANCELLED
    assert any(e.kind == "cancelled" for e in snapshot.events)
    assert DurableRunService(reopened).cancel(run_id).status == RunStatus.CANCELLED


def test_invalid_inputs_are_refused(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).create("", "w", "s")
    with pytest.raises(ValueError):
        StepOutcome(error="x", done=True)
    with pytest.raises(ValueError):
        StepOutcome(cost=-1)
    store = _store(tmp_path)
    run_id = store.create("t", "w", "s")
    with pytest.raises(KeyError):
        store.snapshot("missing")
    with pytest.raises(ValueError):
        store.claim(run_id, "", ttl=1)
    with pytest.raises(ValueError):
        store.claim(run_id, "w", ttl=0)
    with pytest.raises(ValueError):
        store.create("t", "w", "s", run_id=run_id)


def test_input_payload_is_preserved_and_result_is_separate(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="s", payload={"cfg": 1})
    service.run_step(run_id, "worker", lambda _s: StepOutcome(done=True, payload={"r": 2}))
    snapshot = service.snapshot(run_id)
    assert snapshot.payload == {"cfg": 1}
    assert snapshot.result == {"r": 2}


def test_cancel_during_a_step_is_terminal(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="s")

    def cancel_then_done(_snapshot):
        service.cancel(run_id)
        return StepOutcome(done=True, payload={"late": True})

    snapshot = service.run_step(run_id, "worker", cancel_then_done)
    assert snapshot.status == RunStatus.CANCELLED
    with pytest.raises(RuntimeError):
        store.append_event(run_id, "step", payload={})
    assert service.snapshot(run_id).result == {}


def test_stale_worker_cannot_append_after_lease_steal(tmp_path):
    store = _store(tmp_path)
    run_id = store.create("t", "w", "s", now=0)
    assert store.claim(run_id, "a", ttl=1, now=0)
    assert store.claim(run_id, "b", ttl=10, now=5)
    with pytest.raises(LeaseLost):
        store.append_event(run_id, "completed", now=5, worker_id="a",
                           require_lease=True)
    store.append_event(run_id, "completed", now=5, worker_id="b",
                       require_lease=True)


def test_failed_heartbeat_stops_the_loop(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="s")

    def steal_and_done(_snapshot):
        store.claim(run_id, "thief", ttl=10, now=time.time() + 100)
        return StepOutcome(done=True, payload={"by": "victim"})

    snapshot = service.run_step(run_id, "victim", steal_and_done,
                                lease_seconds=0.01)
    assert not snapshot.terminal
    assert snapshot.lease_owner == "thief"
    assert snapshot.result == {}


def test_retries_reset_after_progress_and_boundary(tmp_path):
    store = _store(tmp_path)
    service = DurableRunService(store)
    run_id = service.submit("t", "w", spec_hash="s")
    calls = {"n": 0}

    def flaky(_snapshot):
        calls["n"] += 1
        if calls["n"] in (1, 3):
            raise RuntimeError("blip")
        if calls["n"] == 2:
            return StepOutcome(payload={"progress": True})
        return StepOutcome(done=True)

    snapshot = service.run_step(run_id, "worker", flaky, max_retries=1)
    assert snapshot.status == RunStatus.COMPLETED
    assert calls["n"] == 4

    zero = service.submit("t", "w", spec_hash="s0")

    def always(_snapshot):
        raise RuntimeError("no")

    assert service.run_step(zero, "w", always, max_retries=0).status == RunStatus.FAILED
    one = service.submit("t", "w", spec_hash="s1")
    assert service.run_step(one, "w", always, max_retries=1).status == RunStatus.FAILED


def test_two_connections_share_the_idempotency_contract(tmp_path):
    path = str(tmp_path / "runs.sqlite3")
    one = SqliteRunStore(path)
    two = SqliteRunStore(path)
    run_id = one.create("t", "w", "s")
    a = one.append_event(run_id, "step", idempotency_key="same", payload={"a": 1})
    b = two.append_event(run_id, "step", idempotency_key="same", payload={"a": 2})
    assert a.seq == b.seq
    assert len(one.snapshot(run_id).events) == 1
