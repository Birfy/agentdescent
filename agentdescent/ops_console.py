"""A read-only operations API for runs, approvals, QD, lineage and cost.

CLI/MCP is for the person running the loop. A team needs a control room: what is
running, what is blocked, what needs approval, how much of the behaviour space is
covered, where this artifact came from, and what it cost.

This module is the **API/read-model slice** behind that console. It does not
serve HTTP and it does not render HTML; it composes the existing read models
into stable JSON-shaped payloads with pagination, filters and an owned
approval-queue write path. ``decide_approval`` is the only mutation an HTTP
layer should expose as a POST; the run/QD/lineage/cost views are read-only.
A FastAPI/Flask/React frontend can sit on top without re-deriving the joins.

    console = OperationsConsole()
    console.upsert_run(RunSummary("run-1", "t1", "ws1", "running", reward=0.7))
    console.add_approval(ApprovalItem("ap-1", "artifact-1", risk=0.6))
    page = console.list_runs(status="running", limit=20)
    snapshot = console.snapshot()

The console is deliberately in-memory and role-agnostic. Authorization belongs
to `workspaces.WorkspaceRegistry`; this module only shapes the data the
authorized caller may see.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "RunSummary",
    "ApprovalItem",
    "QDSnapshot",
    "LineageSnapshot",
    "CostSnapshot",
    "RunPage",
    "ConsoleSnapshot",
    "OperationsConsole",
]

_RUN_STATUS = ("pending", "running", "completed", "failed", "cancelled",
               "deferred", "rolled_back")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    tenant: str
    workspace: str
    status: str
    reward: Optional[float] = None
    cost: float = 0.0
    updated_at: float = 0.0
    gate: str = ""
    risk: float = 0.0
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id or not self.tenant or not self.workspace:
            raise ValueError("run_id, tenant and workspace are required")
        if self.status not in _RUN_STATUS:
            raise ValueError(f"unknown run status {self.status!r}")
        if self.reward is not None:
            object.__setattr__(self, "reward", _finite(self.reward, "reward"))
        cost = _finite(self.cost, "cost")
        if cost < 0:
            raise ValueError("cost must be >= 0")
        risk = _finite(self.risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "updated_at", _finite(self.updated_at, "updated_at"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "tenant": self.tenant,
            "workspace": self.workspace, "status": self.status,
            "reward": self.reward, "cost": self.cost,
            "updated_at": self.updated_at, "gate": self.gate,
            "risk": self.risk, "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ApprovalItem:
    approval_id: str
    artifact_id: str
    risk: float = 0.0
    status: str = "pending"
    requested_at: float = 0.0
    detail: str = ""
    decided_by: str = ""
    decided_at: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.approval_id or not self.artifact_id:
            raise ValueError("approval_id and artifact_id are required")
        if self.status not in ("pending", "approved", "denied"):
            raise ValueError("status must be pending, approved or denied")
        risk = _finite(self.risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "requested_at", _finite(self.requested_at, "requested_at"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id, "artifact_id": self.artifact_id,
            "risk": self.risk, "status": self.status,
            "requested_at": self.requested_at, "detail": self.detail,
            "decided_by": self.decided_by, "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class QDSnapshot:
    coverage: float
    qd_score: float
    filled: int
    best: Optional[float] = None
    cells: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("coverage", "qd_score"):
            _finite(getattr(self, name), name)
        if not 0.0 <= float(self.coverage) <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if self.filled < 0:
            raise ValueError("filled must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {"coverage": self.coverage, "qd_score": self.qd_score,
                "filled": self.filled, "best": self.best,
                "cells": list(self.cells)}


@dataclass(frozen=True)
class LineageSnapshot:
    nodes: int
    edges: int
    roots: Tuple[str, ...] = ()
    cycles: int = 0

    def __post_init__(self) -> None:
        if self.nodes < 0 or self.edges < 0 or self.cycles < 0:
            raise ValueError("lineage counts must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges,
                "roots": list(self.roots), "cycles": self.cycles}


@dataclass(frozen=True)
class CostSnapshot:
    tokens: float = 0.0
    cost: float = 0.0
    by_model: Dict[str, float] = field(default_factory=dict)
    by_project: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tokens = _finite(self.tokens, "tokens")
        cost = _finite(self.cost, "cost")
        if tokens < 0 or cost < 0:
            raise ValueError("tokens and cost must be >= 0")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "cost", cost)

    def to_dict(self) -> Dict[str, Any]:
        return {"tokens": self.tokens, "cost": self.cost,
                "by_model": dict(self.by_model),
                "by_project": dict(self.by_project)}


@dataclass(frozen=True)
class RunPage:
    items: Tuple[RunSummary, ...]
    next_cursor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items],
                "next_cursor": self.next_cursor}


@dataclass(frozen=True)
class ConsoleSnapshot:
    runs: int
    by_status: Dict[str, int]
    pending_approvals: int
    qd: Optional[QDSnapshot]
    lineage: Optional[LineageSnapshot]
    cost: Optional[CostSnapshot]
    top_runs: Tuple[RunSummary, ...] = ()
    generated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs": self.runs, "by_status": dict(self.by_status),
            "pending_approvals": self.pending_approvals,
            "qd": None if self.qd is None else self.qd.to_dict(),
            "lineage": None if self.lineage is None else self.lineage.to_dict(),
            "cost": None if self.cost is None else self.cost.to_dict(),
            "top_runs": [run.to_dict() for run in self.top_runs],
            "generated_at": self.generated_at,
        }


class OperationsConsole:
    """In-memory read model behind the operations console."""

    def __init__(self, clock: Optional[Any] = None) -> None:
        self.clock = clock or time.time
        self._runs: Dict[str, RunSummary] = {}
        self._approvals: Dict[str, ApprovalItem] = {}
        self._qd: Optional[QDSnapshot] = None
        self._lineage: Optional[LineageSnapshot] = None
        self._cost: Optional[CostSnapshot] = None

    # -- ingest --------------------------------------------------------------

    def upsert_run(self, run: RunSummary) -> RunSummary:
        self._runs[run.run_id] = run
        return run

    def add_approval(self, approval: ApprovalItem) -> ApprovalItem:
        if approval.approval_id in self._approvals:
            raise ValueError(f"approval {approval.approval_id!r} already exists")
        self._approvals[approval.approval_id] = approval
        return approval

    def decide_approval(self, approval_id: str, decision: str, *,
                        actor: str) -> ApprovalItem:
        if decision not in ("approved", "denied"):
            raise ValueError("decision must be approved or denied")
        if not actor:
            raise ValueError("actor is required")
        try:
            item = self._approvals[approval_id]
        except KeyError:
            raise KeyError(f"unknown approval {approval_id!r}") from None
        if item.status != "pending":
            raise ValueError(f"approval is already {item.status}")
        updated = ApprovalItem(
            item.approval_id, item.artifact_id, item.risk, decision,
            item.requested_at, item.detail, actor, float(self.clock()))
        self._approvals[approval_id] = updated
        return updated

    def set_qd(self, qd: QDSnapshot) -> None:
        self._qd = qd

    def set_lineage(self, lineage: LineageSnapshot) -> None:
        self._lineage = lineage

    def set_cost(self, cost: CostSnapshot) -> None:
        self._cost = cost

    # -- read models ---------------------------------------------------------

    def get_run(self, run_id: str) -> RunSummary:
        try:
            return self._runs[run_id]
        except KeyError:
            raise KeyError(f"unknown run {run_id!r}") from None

    def list_runs(self, *, status: Optional[str] = None,
                  workspace: Optional[str] = None, tenant: Optional[str] = None,
                  limit: int = 50, cursor: Optional[str] = None) -> RunPage:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if status is not None and status not in _RUN_STATUS:
            raise ValueError(f"unknown run status {status!r}")
        rows = sorted(
            (run for run in self._runs.values()
             if (status is None or run.status == status)
             and (workspace is None or run.workspace == workspace)
             and (tenant is None or run.tenant == tenant)),
            key=lambda run: (-run.updated_at, run.run_id))
        if cursor is not None:
            if cursor not in self._runs:
                raise KeyError(f"unknown cursor {cursor!r}")
            if cursor not in {run.run_id for run in rows}:
                raise ValueError(
                    f"cursor {cursor!r} is not in the filtered result")
            index = next(i for i, run in enumerate(rows) if run.run_id == cursor)
            rows = rows[index + 1:]
        page = rows[:limit]
        next_cursor = page[-1].run_id if len(rows) > limit else None
        return RunPage(tuple(page), next_cursor)

    def pending_approvals(self) -> Tuple[ApprovalItem, ...]:
        return tuple(sorted(
            (item for item in self._approvals.values() if item.status == "pending"),
            key=lambda item: (-item.risk, item.requested_at, item.approval_id)))

    def snapshot(self) -> ConsoleSnapshot:
        by_status: Dict[str, int] = {}
        for run in self._runs.values():
            by_status[run.status] = by_status.get(run.status, 0) + 1
        top = sorted(self._runs.values(),
                     key=lambda run: (-(run.reward if run.reward is not None else -1.0),
                                      run.run_id))[:5]
        return ConsoleSnapshot(
            runs=len(self._runs), by_status=dict(sorted(by_status.items())),
            pending_approvals=len(self.pending_approvals()), qd=self._qd,
            lineage=self._lineage, cost=self._cost, top_runs=tuple(top),
            generated_at=float(self.clock()))

    def to_dict(self) -> Dict[str, Any]:
        return self.snapshot().to_dict()
