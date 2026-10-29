"""A durable, resumable run service for long self-improvement sessions.

``run_guarded`` is an in-process call: a crash loses the run. An industrial RSI
loop is hours or days of sessions, rollouts, merges and gates, often on more
than one worker. This module is the smallest useful durable-execution seam:

* a **SQLite-backed event log** for run state (stdlib only, reopenable);
* **leases** with expiry and heartbeat, so one worker owns a run at a time;
* **idempotency keys** on events, so a retried step or a duplicate completion
  cannot commit twice;
* **at-least-once execution with explicit retries**, and a terminal `failed`
  after ``max_retries``;
* **resume** by reloading the event log, not by restarting the whole run;
* **cancel** and **expired-lease recovery**.

The actual session step is a caller-supplied callable ``snapshot -> StepOutcome``.
That keeps the service independent of the engine: a step can be
``run_guarded``, a campaign session, a population generation or a remote job.

    store = SqliteRunStore("runs.sqlite3")
    service = DurableRunService(store)
    run_id = service.submit("tenant-a", "ws-1", spec_hash="abc")
    run_id = service.run_step(run_id, "worker-1", step)
    snapshot = service.snapshot(run_id)

This is a vertical slice, not a distributed scheduler: one SQLite file, one
lease at a time, and no cross-process pub/sub. The schema and the idempotency
contract are the parts a production adapter (Temporal/Postgres/Redis) keeps.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "RunEvent",
    "RunSnapshot",
    "StepOutcome",
    "SqliteRunStore",
    "DurableRunService",
    "RunStatus",
    "LeaseLost",
]


class LeaseLost(RuntimeError):
    """A worker tried to write after its lease expired or was stolen."""


class RunStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = (COMPLETED, FAILED, CANCELLED)
    ALL = (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)


@dataclass(frozen=True)
class RunEvent:
    """One append-only state transition."""

    run_id: str
    seq: int
    kind: str
    at: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class RunSnapshot:
    """The current run state and its event history."""

    run_id: str
    tenant: str
    workspace: str
    spec_hash: str
    status: str
    attempts: int
    created_at: float
    updated_at: float
    lease_owner: Optional[str] = None
    lease_until: Optional[float] = None
    error: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] = field(default_factory=dict)
    events: Tuple[RunEvent, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.status in RunStatus.TERMINAL

    @property
    def leased(self) -> bool:
        return self.lease_owner is not None


@dataclass(frozen=True)
class StepOutcome:
    """One step's decision: done, retryable error, or another step."""

    done: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    retryable: bool = False
    cost: float = 0.0
    event_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.cost)) or float(self.cost) < 0:
            raise ValueError("StepOutcome.cost must be finite and >= 0")
        if self.done and self.error:
            raise ValueError("a completed StepOutcome cannot also have an error")


StepFn = Callable[[RunSnapshot], StepOutcome]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class SqliteRunStore:
    """A small, reopenable event store with lease and idempotency support."""

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("SqliteRunStore needs a path")
        self.path = path
        self._db = sqlite3.connect(path, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                tenant TEXT NOT NULL,
                workspace TEXT NOT NULL,
                spec_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                lease_owner TEXT,
                lease_until REAL,
                error TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL DEFAULT '{}',
                result TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                kind TEXT NOT NULL,
                at REAL NOT NULL,
                payload TEXT NOT NULL,
                idempotency_key TEXT,
                PRIMARY KEY (run_id, seq),
                UNIQUE (run_id, idempotency_key)
            )
        """)
        # Additive migration for stores created by the first slice.
        try:
            self._db.execute("ALTER TABLE runs ADD COLUMN result TEXT NOT NULL DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def create(self, tenant: str, workspace: str, spec_hash: str, *,
               run_id: Optional[str] = None, now: Optional[float] = None,
               payload: Optional[Mapping[str, Any]] = None) -> str:
        if not tenant or not workspace:
            raise ValueError("tenant and workspace are required")
        run_id = run_id or uuid.uuid4().hex
        at = time.time() if now is None else float(now)
        try:
            self._db.execute(
                "INSERT INTO runs(run_id, tenant, workspace, spec_hash, status, "
                "created_at, updated_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, tenant, workspace, spec_hash, RunStatus.PENDING, at, at,
                 _json(payload or {})))
            self._db.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"run {run_id!r} already exists") from None
        return run_id

    def append_event(self, run_id: str, kind: str, *,
                     payload: Optional[Mapping[str, Any]] = None,
                     idempotency_key: Optional[str] = None,
                     now: Optional[float] = None,
                     worker_id: Optional[str] = None,
                     require_lease: bool = False) -> RunEvent:
        """Append one event atomically.

        Terminal states are sticky. When ``require_lease`` (or ``worker_id``) is
        set, the insert is fenced on ``lease_owner=worker_id`` and
        ``lease_until >= now`` in the same transaction as the event.
        """
        at = time.time() if now is None else float(now)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            if idempotency_key is not None:
                row = self._db.execute(
                    "SELECT run_id, seq, kind, at, payload FROM events "
                    "WHERE run_id=? AND idempotency_key=?",
                    (run_id, idempotency_key)).fetchone()
                if row is not None:
                    self._db.commit()
                    return RunEvent(row["run_id"], row["seq"], row["kind"],
                                    row["at"], json.loads(row["payload"]),
                                    idempotency_key)
            run = self._db.execute(
                "SELECT status, lease_owner, lease_until FROM runs WHERE run_id=?",
                (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"unknown run {run_id!r}")
            if run["status"] in RunStatus.TERMINAL:
                raise RuntimeError(f"run {run_id!r} is {run['status']}")
            if require_lease or worker_id is not None:
                if (run["lease_owner"] != worker_id or run["lease_until"] is None
                        or float(run["lease_until"]) < at):
                    raise LeaseLost(
                        f"run {run_id!r} lease is held by {run['lease_owner']!r}")
            row = self._db.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id=?",
                (run_id,)).fetchone()
            seq = int(row[0]) + 1
            body = _json(payload or {})
            self._db.execute(
                "INSERT INTO events(run_id, seq, kind, at, payload, idempotency_key) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, seq, kind, at, body, idempotency_key))
            status = self._status_for(kind)
            attempts = 1 if kind == "attempt" else 0
            error = str((payload or {}).get("error", "")) if kind == "failed" else ""
            self._db.execute(
                "UPDATE runs SET status=?, updated_at=?, attempts=attempts+?, "
                "error=CASE WHEN ?='' THEN error ELSE ? END WHERE run_id=?",
                (status, at, attempts, error, error, run_id))
            if kind == "completed" and isinstance(payload, Mapping) and "result" in payload:
                self._db.execute(
                    "UPDATE runs SET result=? WHERE run_id=?",
                    (_json(payload.get("result") or {}), run_id))
            self._db.commit()
            return RunEvent(run_id, seq, kind, at, payload or {}, idempotency_key)
        except Exception:
            self._db.rollback()
            raise

    @staticmethod
    def _status_for(kind: str) -> str:
        if kind == "completed":
            return RunStatus.COMPLETED
        if kind == "failed":
            return RunStatus.FAILED
        if kind == "cancelled":
            return RunStatus.CANCELLED
        return RunStatus.RUNNING

    def claim(self, run_id: str, worker_id: str, *, ttl: float,
              now: Optional[float] = None) -> bool:
        if not worker_id:
            raise ValueError("worker_id is required")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        at = time.time() if now is None else float(now)
        cur = self._db.execute(
            "UPDATE runs SET lease_owner=?, lease_until=? "
            "WHERE run_id=? AND status IN (?, ?) "
            "AND (lease_owner IS NULL OR lease_owner=? OR lease_until < ?)",
            (worker_id, at + ttl, run_id, RunStatus.PENDING, RunStatus.RUNNING,
             worker_id, at))
        self._db.commit()
        return cur.rowcount == 1

    def heartbeat(self, run_id: str, worker_id: str, *, ttl: float,
                  now: Optional[float] = None) -> bool:
        at = time.time() if now is None else float(now)
        cur = self._db.execute(
            "UPDATE runs SET lease_until=? WHERE run_id=? AND lease_owner=? "
            "AND status IN (?, ?)",
            (at + ttl, run_id, worker_id, RunStatus.PENDING, RunStatus.RUNNING))
        self._db.commit()
        return cur.rowcount == 1

    def release(self, run_id: str, worker_id: str) -> bool:
        cur = self._db.execute(
            "UPDATE runs SET lease_owner=NULL, lease_until=NULL "
            "WHERE run_id=? AND lease_owner=?", (run_id, worker_id))
        self._db.commit()
        return cur.rowcount == 1

    def recover_expired(self, *, now: Optional[float] = None) -> Tuple[str, ...]:
        at = time.time() if now is None else float(now)
        rows = self._db.execute(
            "SELECT run_id FROM runs WHERE status IN (?, ?) "
            "AND lease_owner IS NOT NULL AND lease_until < ?",
            (RunStatus.PENDING, RunStatus.RUNNING, at)).fetchall()
        ids = tuple(row["run_id"] for row in rows)
        self._db.execute(
            "UPDATE runs SET lease_owner=NULL, lease_until=NULL "
            "WHERE status IN (?, ?) AND lease_owner IS NOT NULL AND lease_until < ?",
            (RunStatus.PENDING, RunStatus.RUNNING, at))
        self._db.commit()
        return ids

    def snapshot(self, run_id: str) -> RunSnapshot:
        row = self._db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id!r}")
        events = tuple(
            RunEvent(r["run_id"], r["seq"], r["kind"], r["at"],
                     json.loads(r["payload"]), r["idempotency_key"])
            for r in self._db.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY seq", (run_id,)))
        return RunSnapshot(
            run_id=row["run_id"], tenant=row["tenant"], workspace=row["workspace"],
            spec_hash=row["spec_hash"], status=row["status"],
            attempts=row["attempts"], created_at=row["created_at"],
            updated_at=row["updated_at"], lease_owner=row["lease_owner"],
            lease_until=row["lease_until"], error=row["error"],
            payload=json.loads(row["payload"]), result=json.loads(row["result"]),
            events=events)

    def list_runs(self, *, tenant: Optional[str] = None,
                  workspace: Optional[str] = None,
                  status: Optional[str] = None) -> Tuple[str, ...]:
        clauses, values = [], []
        for column, value in (("tenant", tenant), ("workspace", workspace),
                              ("status", status)):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._db.execute(
            "SELECT run_id FROM runs" + where + " ORDER BY created_at, run_id",
            values).fetchall()
        return tuple(row["run_id"] for row in rows)


class DurableRunService:
    """Lease, step, retry and complete a run against a :class:`SqliteRunStore`."""

    def __init__(self, store: SqliteRunStore) -> None:
        self.store = store

    def submit(self, tenant: str, workspace: str, *, spec_hash: str,
               run_id: Optional[str] = None,
               payload: Optional[Mapping[str, Any]] = None) -> str:
        return self.store.create(tenant, workspace, spec_hash, run_id=run_id,
                                 payload=payload)

    def snapshot(self, run_id: str) -> RunSnapshot:
        return self.store.snapshot(run_id)

    def cancel(self, run_id: str, *, reason: str = "operator cancel") -> RunSnapshot:
        snapshot = self.store.snapshot(run_id)
        if snapshot.terminal:
            return snapshot
        self.store.append_event(run_id, "cancelled", payload={"reason": reason},
                                idempotency_key="cancelled")
        return self.store.snapshot(run_id)

    def recover(self) -> Tuple[str, ...]:
        return self.store.recover_expired()

    @staticmethod
    def _consecutive_attempts(snapshot: RunSnapshot) -> int:
        count = 0
        for event in reversed(snapshot.events):
            if event.kind == "attempt":
                count += 1
            elif event.kind in ("step", "completed", "failed", "cancelled"):
                break
        return count

    def run_step(self, run_id: str, worker_id: str, step: StepFn, *,
                 lease_seconds: float = 30.0, max_steps: int = 100,
                 max_retries: int = 3) -> RunSnapshot:
        """Run one leased step loop until done, failed, cancelled or bounded.

        The step is at-least-once: a crash after the external effect but before
        the event is retried. Use ``StepOutcome.event_key`` (or the run's own
        idempotency key) for the external effect. Completion is exactly-once
        through the unique ``completed`` key. Every write is fenced on the live
        lease, so a stolen worker cannot commit after a recovery.
        """
        if max_steps < 1 or max_retries < 0:
            raise ValueError("max_steps >= 1 and max_retries >= 0")
        if not self.store.claim(run_id, worker_id, ttl=lease_seconds):
            return self.store.snapshot(run_id)
        try:
            for _ in range(max_steps):
                if not self.store.heartbeat(run_id, worker_id, ttl=lease_seconds):
                    return self.store.snapshot(run_id)
                snapshot = self.store.snapshot(run_id)
                if snapshot.terminal:
                    return snapshot
                try:
                    outcome = step(snapshot)
                    if not isinstance(outcome, StepOutcome):
                        raise TypeError(
                            f"step returned {type(outcome).__name__}, expected StepOutcome")
                except Exception as exc:  # noqa: BLE001 - a step failure is data
                    outcome = StepOutcome(error=f"{type(exc).__name__}: {exc}",
                                          retryable=True)
                if outcome.error:
                    consecutive = self._consecutive_attempts(snapshot) + 1
                    try:
                        self.store.append_event(
                            run_id, "attempt",
                            payload={"error": outcome.error, "cost": outcome.cost},
                            idempotency_key=f"attempt:{snapshot.attempts + 1}",
                            worker_id=worker_id, require_lease=True)
                    except (LeaseLost, RuntimeError):
                        return self.store.snapshot(run_id)
                    if not self.store.heartbeat(run_id, worker_id, ttl=lease_seconds):
                        return self.store.snapshot(run_id)
                    if not outcome.retryable or consecutive > max_retries:
                        try:
                            self.store.append_event(
                                run_id, "failed",
                                payload={"error": outcome.error},
                                idempotency_key="failed",
                                worker_id=worker_id, require_lease=True)
                        except (LeaseLost, RuntimeError):
                            pass
                        return self.store.snapshot(run_id)
                    continue
                if outcome.done:
                    if self.store.snapshot(run_id).terminal:
                        return self.store.snapshot(run_id)
                    try:
                        self.store.append_event(
                            run_id, "completed",
                            payload={"result": dict(outcome.payload),
                                     "cost": outcome.cost},
                            idempotency_key="completed",
                            worker_id=worker_id, require_lease=True)
                    except (LeaseLost, RuntimeError):
                        pass
                    return self.store.snapshot(run_id)
                key = outcome.event_key or f"step:{snapshot.attempts}:{len(snapshot.events)}"
                try:
                    self.store.append_event(
                        run_id, "step",
                        payload={"result": dict(outcome.payload), "cost": outcome.cost},
                        idempotency_key=key,
                        worker_id=worker_id, require_lease=True)
                except (LeaseLost, RuntimeError):
                    return self.store.snapshot(run_id)
                if not self.store.heartbeat(run_id, worker_id, ttl=lease_seconds):
                    return self.store.snapshot(run_id)
            return self.store.snapshot(run_id)
        finally:
            self.store.release(run_id, worker_id)
