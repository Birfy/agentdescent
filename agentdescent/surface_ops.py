"""The ``agentdescent ops`` surface: publish finished runs to the control room.

`campaign_ops.CampaignOps` already turns a sequence of `LoopReport`s into the
console, lineage, telemetry, metrics and alerts. This module makes that work
from a durable run database, so a person or a host agent can publish a whole
campaign without writing Python. Each call builds a fresh in-memory read model
and returns it; nothing is persisted or deduplicated across separate calls.

    agentdescent ops plan    --runs-db platform.sqlite3 --tenant t1 --workspace ws
    agentdescent ops publish --runs-db platform.sqlite3 --tenant t1 --workspace ws \
                             --out ops.json --prometheus metrics.txt

The command reads only terminal runs. A missing database or an empty set of
terminal runs is a report, not a traceback: ``published: 0`` with a next step.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .campaign_ops import CampaignOps
from .durable_runs import SqliteRunStore
from .loop_kernel import LoopConfig, LoopKernel, LoopReport
from .ops_console import OperationsConsole
from .ops_fabric import OpsFabric
from .rsi_metrics import to_prometheus
from .rsi_telemetry import RSITelemetry
from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, atomic_write_json, atomic_write_text,
    require_text,
)

__all__ = ["ops_plan", "ops_publish"]


def _store(value: Any) -> str:
    path = require_text(value, "runs_db")
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise SurfaceError(
            f"runs_db {value!r} does not exist; pass the SQLite file a "
            "platform/loop run wrote")
    return resolved


def _loops(path: str, *, tenant: str, workspace: str) -> Tuple[LoopReport, ...]:
    store = SqliteRunStore(path)
    try:
        kernel = LoopKernel(
            store, LoopConfig(tenant=tenant, workspace=workspace,
                              spec_hash="ops"))
        reports: List[LoopReport] = []
        for run_id in store.list_runs(tenant=tenant, workspace=workspace):
            report = kernel.report(run_id)
            if report.terminal:
                reports.append(report)
        return tuple(reports)
    finally:
        store.close()


def ops_plan(*, runs_db: Any, tenant: Any, workspace: Any) -> Dict[str, Any]:
    path = _store(runs_db)
    tenant = require_text(tenant, "tenant")
    workspace = require_text(workspace, "workspace")
    loops = _loops(path, tenant=tenant, workspace=workspace)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "runs_db": path,
        "tenant": tenant,
        "workspace": workspace,
        "terminal_runs": len(loops),
        "generations": sum(len(loop.generations) for loop in loops),
        "will_publish": len(loops),
        "next": (
            ["agentdescent ops publish --runs-db <runs_db> ..."]
            if loops else
            ["run a platform/loop campaign first, then publish"]),
    }


def ops_publish(*, runs_db: Any, tenant: Any, workspace: Any,
                min_acceptance: float = 0.0,
                max_cost_per_accepted: Optional[float] = None,
                out: Optional[str] = None,
                prometheus: Optional[str] = None) -> Dict[str, Any]:
    path = _store(runs_db)
    tenant = require_text(tenant, "tenant")
    workspace = require_text(workspace, "workspace")
    loops = _loops(path, tenant=tenant, workspace=workspace)
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "runs_db": path,
        "tenant": tenant,
        "workspace": workspace,
        "published": len(loops),
    }
    if not loops:
        payload["detail"] = (
            "no terminal runs under this tenant/workspace; publish is a no-op")
        payload["next"] = ["run a platform/loop campaign first"]
    else:
        fabric = OpsFabric(OperationsConsole(), telemetry=RSITelemetry())
        report = CampaignOps(fabric).publish(
            loops, tenant=tenant, workspace=workspace,
            min_acceptance=min_acceptance,
            max_cost_per_accepted=max_cost_per_accepted)
        payload["report"] = report.to_dict()
        payload["metrics"] = report.metrics.to_dict()
        payload["alerts"] = list(report.alerts)
        payload["next"] = [
            "persist report/snapshot with the release record",
            "feed metrics to your Prometheus/OTel collector",
        ]
        if prometheus is not None:
            payload["prometheus"] = atomic_write_text(
                prometheus, to_prometheus(report.metrics))
    if out is not None:
        payload["out"] = atomic_write_json(out, payload)
    return payload
