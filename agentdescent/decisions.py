"""An append-only decision record: what was adopted, why, and at what cost.

The engine already keeps the artifact (ledger), the evidence (trajectory,
candidates, audit) and the run (runstore). A product also needs the *decision*
as one immutable object: the candidate, the guard that fired, the evidence it
saw, the cost it spent and its parent. ``DecisionLog`` stores those records as
JSONL so a run can be audited and replayed without reconstructing the story from
five directories.

    log = DecisionLog("~/.agentdescent/decisions.jsonl")
    log.record(kind="adoption", artifact_id="skill", diff_id="d1",
               accepted=False, category="leakage:gold",
               evidence={"leaked": ["321"]}, cost={"calls": 12})
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

__all__ = ["DecisionLog", "DecisionRecord"]


def _clean(value: Any) -> Any:
    """JSON-safe: non-finite floats become ``None``, unknown objects strings."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class DecisionRecord:
    """One immutable adoption/session decision."""

    record_id: str
    kind: str
    artifact_id: str
    diff_id: str
    accepted: bool
    category: str = ""
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "record_id": self.record_id,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "diff_id": self.diff_id,
            "accepted": bool(self.accepted),
            "category": self.category,
            "detail": self.detail,
            "evidence": _clean(self.evidence),
            "cost": _clean(self.cost),
            "parent_id": self.parent_id,
            "created_at": self.created_at,
        }

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", copy.deepcopy(dict(self.evidence or {})))
        object.__setattr__(self, "cost", copy.deepcopy(dict(self.cost or {})))
        object.__setattr__(
            self, "parent_id",
            (str(self.parent_id) if self.parent_id is not None else None))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DecisionRecord":
        if not isinstance(payload, dict):
            raise ValueError(
                f"a decision record must be a JSON object, got "
                f"{type(payload).__name__}")
        return cls(
            record_id=str(payload.get("record_id", "")),
            kind=str(payload.get("kind", "")),
            artifact_id=str(payload.get("artifact_id", "")),
            diff_id=str(payload.get("diff_id", "")),
            accepted=bool(payload.get("accepted", False)),
            category=str(payload.get("category", "")),
            detail=str(payload.get("detail", "")),
            evidence=dict(payload.get("evidence") or {}),
            cost=dict(payload.get("cost") or {}),
            parent_id=(str(payload["parent_id"])
                       if payload.get("parent_id") is not None else None),
            created_at=float(payload.get("created_at", 0.0)),
        )


_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


class DecisionLog:
    """Append-only JSONL decisions.

    Writers in one process are serialized per path (two ``DecisionLog`` objects
    on the same file share a lock). Separate processes are not serialized; use
    one writer per process or an external file lock. ``record_id`` is the
    primary key and ``append`` refuses a duplicate.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        self._lock = _lock_for(self.path)

    def record(self, *, kind: str, artifact_id: str, diff_id: str = "",
               accepted: bool, category: str = "", detail: str = "",
               evidence: Optional[Dict[str, Any]] = None,
               cost: Optional[Dict[str, Any]] = None,
               parent_id: Optional[str] = None,
               record_id: Optional[str] = None) -> DecisionRecord:
        record = DecisionRecord(
            record_id=record_id or uuid.uuid4().hex,
            kind=str(kind), artifact_id=str(artifact_id), diff_id=str(diff_id),
            accepted=bool(accepted), category=str(category), detail=str(detail),
            evidence=dict(evidence or {}), cost=dict(cost or {}),
            parent_id=parent_id, created_at=time.time())
        self.append(record)
        return record

    def append(self, record: DecisionRecord) -> None:
        data = json.dumps(record.to_dict(), sort_keys=True,
                          allow_nan=False) + "\n"
        with self._lock:
            if any(row.record_id == record.record_id for row in self):
                raise ValueError(
                    f"duplicate record_id {record.record_id!r}")
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            if self._needs_newline():
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write("\n")
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())

    def _needs_newline(self) -> bool:
        try:
            if os.path.getsize(self.path) == 0:
                return False
            with open(self.path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                return fh.read(1) != b"\n"
        except FileNotFoundError:
            return False

    def __iter__(self) -> Iterator[DecisionRecord]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            continue
                        yield DecisionRecord.from_dict(payload)
                    except (ValueError, TypeError, AttributeError):
                        continue
        except FileNotFoundError:
            return

    def all(self) -> List[DecisionRecord]:
        return list(self)

    def get(self, record_id: str) -> Optional[DecisionRecord]:
        return next((record for record in self if record.record_id == record_id),
                    None)

    def children(self, record_id: str) -> List[DecisionRecord]:
        return [record for record in self if record.parent_id == record_id]

    def count(self) -> int:
        return sum(1 for _ in self)
