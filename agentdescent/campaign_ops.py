"""Publish a multi-cycle campaign to the control room in one call.

``LoopKernel`` produces a ``LoopReport`` per durable loop, ``OpsFabric`` copies
one report into the console, lineage and telemetry read models, and
``rsi_metrics`` aggregates decision records into acceptance and cost alerts. A
campaign of several loops still had to wire those three steps together by hand,
and the read models could drift from the metrics.

``CampaignOps`` is that campaign-level write path. It records every loop through
``OpsFabric`` (idempotently, so a rerun does not duplicate runs or spans),
derives one decision row per generation for ``metrics_from_records``, applies
the same ``alerts`` thresholds, and returns a single ``CampaignOpsReport``:

    ops = OpsFabric(OperationsConsole(), telemetry=RSITelemetry())
    campaign = CampaignOps(ops)
    report = campaign.publish(
        [loop_a, loop_b], tenant="t1", workspace="ws1", min_acceptance=0.5)
    report.metrics.decisions, report.alerts, report.snapshot.console.runs

Validation is fail-closed: an empty campaign, a non-``LoopReport`` element, a
malformed generation cost or an invalid alert threshold raises before anything
is recorded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .loop_kernel import LoopGeneration, LoopReport
from .ops_fabric import OpsFabric, OpsRecord, OpsSnapshot
from .rsi_metrics import RSIMetrics, alerts, metrics_from_records

__all__ = ["CampaignOpsReport", "CampaignOps"]


@dataclass(frozen=True)
class CampaignOpsReport:
    """One campaign's write acknowledgements, read models, metrics and alerts."""

    records: Tuple[OpsRecord, ...]
    snapshot: OpsSnapshot
    metrics: RSIMetrics
    alerts: Tuple[str, ...]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [
                {
                    "run": record.run.to_dict(),
                    "lineage": record.lineage.to_dict(),
                    "span_id": record.span_id,
                    "detail": record.detail,
                }
                for record in self.records
            ],
            "snapshot": self.snapshot.to_dict(),
            "metrics": self.metrics.to_dict(),
            "alerts": list(self.alerts),
            "detail": self.detail,
        }


def _calls(generation: LoopGeneration, run_id: str) -> int:
    cost = generation.cost
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        raise ValueError(
            f"run {run_id!r} generation {generation.generation}: cost must be "
            "a number")
    number = float(cost)
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"run {run_id!r} generation {generation.generation}: cost must be "
            "finite and non-negative")
    return int(number)


class CampaignOps:
    """Record a whole campaign and return its consolidated control-room view."""

    def __init__(self, ops: OpsFabric) -> None:
        if not isinstance(ops, OpsFabric):
            raise TypeError("ops must be an OpsFabric")
        self.ops = ops

    def publish(
        self,
        loops: Sequence[LoopReport],
        *,
        tenant: str,
        workspace: str,
        min_acceptance: float = 0.0,
        max_cost_per_accepted: Optional[float] = None,
    ) -> CampaignOpsReport:
        """Record every loop, aggregate its decisions and return one report."""
        if isinstance(loops, (str, bytes)) or not isinstance(loops, _Sequence):
            raise TypeError("loops must be a non-empty sequence of LoopReport")
        items = tuple(loops)
        if not items:
            raise ValueError("loops must contain at least one LoopReport")
        for loop in items:
            if not isinstance(loop, LoopReport):
                raise TypeError("every loop must be a LoopReport")
        if not isinstance(tenant, str) or not tenant.strip():
            raise ValueError("tenant is required")
        if not isinstance(workspace, str) or not workspace.strip():
            raise ValueError("workspace is required")
        alerts(RSIMetrics(), min_acceptance=min_acceptance,
               max_cost_per_accepted=max_cost_per_accepted)

        rows = []
        for loop in items:
            for generation in loop.generations:
                if not isinstance(generation, LoopGeneration):
                    raise TypeError(
                        f"run {loop.run_id!r}: every generation must be a "
                        "LoopGeneration")
                rows.append({
                    "accepted": generation.accepted,
                    "category": generation.action,
                    "cost": {"calls": _calls(generation, loop.run_id)},
                })

        records = tuple(
            self.ops.record(loop, tenant=tenant, workspace=workspace)
            for loop in items)
        metrics = metrics_from_records(rows)
        alert_rows = alerts(
            metrics, min_acceptance=min_acceptance,
            max_cost_per_accepted=max_cost_per_accepted)
        snapshot = self.ops.snapshot()
        generations = sum(len(loop.generations) for loop in items)
        detail = (
            f"loops={', '.join(loop.run_id for loop in items)}; "
            f"generations={generations}; decisions={metrics.decisions}; "
            f"alerts={len(alert_rows)}")
        return CampaignOpsReport(
            records=records, snapshot=snapshot, metrics=metrics,
            alerts=tuple(alert_rows), detail=detail)
