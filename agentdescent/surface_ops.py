"""The ``agentdescent ops`` surface: publish finished runs to the control room.

`campaign_ops.CampaignOps` already turns a sequence of `LoopReport`s into the
console, lineage, telemetry, metrics and alerts. This module makes that work
from a durable run database, so a person or a host agent can publish a whole
campaign without writing Python. Each call builds a fresh in-memory read model
and returns it; pass ``history`` to also append one JSONL record, so separate
calls can be summarised across runs (``ops_history``).

    agentdescent ops plan    --runs-db platform.sqlite3 --tenant t1 --workspace ws
    agentdescent ops publish --runs-db platform.sqlite3 --tenant t1 --workspace ws \
                             --out ops.json --prometheus metrics.txt \
                             --history ops.jsonl
    agentdescent ops history --history ops.jsonl

The command reads only terminal runs. A missing database or an empty set of
terminal runs is a report, not a traceback: ``published: 0`` with a next step.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .campaign_ops import CampaignOps
from .durable_runs import SqliteRunStore
from .loop_kernel import LoopConfig, LoopKernel, LoopReport
from .ops_console import OperationsConsole
from .ops_fabric import OpsFabric
from .rsi_metrics import metrics_from_records, to_prometheus
from .rsi_telemetry import RSITelemetry
from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, atomic_write_json, atomic_write_text,
    require_text,
)

__all__ = ["ops_plan", "ops_publish", "ops_history"]


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


def _append_history(path: str, record: Dict[str, Any]) -> None:
    """Append one JSONL record and fsync; never rewrite the file."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                record, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise SurfaceError(
            f"history {path!r} cannot be written: {error}") from None


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _loop_metrics(loop: LoopReport) -> Dict[str, Any]:
    """The metrics for one run, using the same rows CampaignOps aggregates."""
    rows: List[Dict[str, Any]] = []
    calls = 0.0
    for generation in loop.generations:
        cost = generation.cost
        if (isinstance(cost, bool) or not isinstance(cost, (int, float))
                or not math.isfinite(float(cost)) or float(cost) < 0):
            raise SurfaceError(
                f"run {loop.run_id!r} generation {generation.generation}: "
                "cost must be finite and non-negative")
        calls += float(cost)
        # Charge the whole loop's cost to its first decision, matching
        # CampaignOps.publish, so per-run totals add up to the campaign total.
        rows.append({"accepted": generation.accepted,
                     "category": generation.action,
                     "cost": {"calls": 0}})
    if rows and loop.generations:
        rows[-len(loop.generations)]["cost"]["calls"] = int(math.ceil(calls))
    return metrics_from_records(rows).to_dict()


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(
            float(value), tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def ops_publish(*, runs_db: Any, tenant: Any, workspace: Any,
                min_acceptance: float = 0.0,
                max_cost_per_accepted: Optional[float] = None,
                out: Optional[str] = None,
                prometheus: Optional[str] = None,
                history: Optional[str] = None) -> Dict[str, Any]:
    path = _store(runs_db)
    tenant = require_text(tenant, "tenant")
    workspace = require_text(workspace, "workspace")
    history_path = (None if history is None else
                    os.path.abspath(os.path.expanduser(
                        require_text(history, "history"))))
    loops = _loops(path, tenant=tenant, workspace=workspace)
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "runs_db": path,
        "tenant": tenant,
        "workspace": workspace,
        "published": len(loops),
    }
    if history_path is not None:
        payload["history"] = history_path
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
        if history_path is not None:
            # Append before the report files: a failed report write must not
            # leave an unrecorded publish behind.
            _append_history(history_path, {
                "schema_version": SURFACE_SCHEMA_VERSION,
                "recorded_at": time.time(),
                "tenant": tenant,
                "workspace": workspace,
                "published": len(loops),
                "run_ids": [loop.run_id for loop in loops],
                "runs": [{"run_id": loop.run_id,
                          "metrics": _loop_metrics(loop)} for loop in loops],
                "metrics": payload["metrics"],
                "alerts": payload["alerts"],
            })
        if prometheus is not None:
            payload["prometheus"] = atomic_write_text(
                prometheus, to_prometheus(report.metrics))
    if out is not None:
        payload["out"] = atomic_write_json(out, payload)
    return payload


def ops_history(*, history: Any, tenant: Optional[str] = None,
                workspace: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate the JSONL history written by ``ops publish --history``.

    Each publish records a snapshot, so records are deduplicated **by run id**
    (the latest snapshot of a run wins) before totals are summed. Publishing the
    same database twice therefore does not double the counts. Alerts are
    deduplicated by message. A missing or empty file is an empty history, not an
    error; malformed lines and records without per-run metrics are skipped and
    counted in ``skipped``.
    """
    path = os.path.abspath(os.path.expanduser(require_text(history, "history")))
    tenant = None if tenant is None else require_text(tenant, "tenant")
    workspace = None if workspace is None else require_text(
        workspace, "workspace")
    selected: List[Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]],
                         List[str]]] = []
    skipped = 0
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as error:
            raise SurfaceError(
                f"history {history!r} cannot be read: {error}") from None
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            if tenant is not None and str(record.get("tenant", "")) != tenant:
                continue
            if (workspace is not None
                    and str(record.get("workspace", "")) != workspace):
                continue
            runs = record.get("runs")
            if not isinstance(runs, list) or not runs:
                skipped += 1
                continue
            parsed: List[Tuple[str, Dict[str, Any]]] = []
            bad = False
            for row in runs:
                if not isinstance(row, dict):
                    bad = True
                    break
                run_id = row.get("run_id")
                metrics = row.get("metrics")
                if (not isinstance(run_id, str) or not run_id
                        or not isinstance(metrics, dict)):
                    bad = True
                    break
                parsed.append((run_id, metrics))
            if bad:
                skipped += 1
                continue
            alerts = record.get("alerts")
            if not isinstance(alerts, list):
                alerts = []
            selected.append((record, parsed,
                             [alert for alert in alerts
                              if isinstance(alert, str)]))

    latest: Dict[str, Dict[str, Any]] = {}
    alert_messages: set = set()
    for _, runs, alerts in selected:
        for run_id, metrics in runs:
            latest[run_id] = metrics
        alert_messages.update(alerts)

    totals = {
        "decisions": 0, "accepted": 0, "rejected": 0, "calls": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "wallclock": 0.0,
    }
    categories: Dict[str, int] = {}
    for metrics in latest.values():
        for key in ("decisions", "accepted", "rejected", "calls",
                    "prompt_tokens", "completion_tokens"):
            totals[key] += int(round(_number(metrics.get(key))))
        totals["wallclock"] += _number(metrics.get("wallclock"))
        cats = metrics.get("categories")
        if isinstance(cats, dict):
            for name, count in cats.items():
                categories[str(name)] = (
                    categories.get(str(name), 0)
                    + int(round(_number(count))))
    accepted = totals["accepted"]
    decisions = totals["decisions"]
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "history": path,
        "tenant": tenant,
        "workspace": workspace,
        "publishes": len(selected),
        "published": len(latest),
        "runs": len(latest),
        "run_ids": sorted(latest),
        "decisions": decisions,
        "accepted": accepted,
        "rejected": totals["rejected"],
        "calls": totals["calls"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "wallclock": totals["wallclock"],
        "acceptance_rate": (accepted / decisions if decisions else 0.0),
        "cost_per_accepted": (totals["calls"] / accepted if accepted else None),
        "alerts": len(alert_messages),
        "categories": categories,
        "last": selected[-1][0] if selected else None,
        "last_recorded_at": (_iso(selected[-1][0].get("recorded_at"))
                             if selected else None),
        "skipped": skipped,
    }
    if not selected:
        payload["next"] = [
            "run ops publish --history <history> to record a campaign"]
    return payload
