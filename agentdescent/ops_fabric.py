"""Put a loop's run, lineage and cost into the operations console.

`ops_console` is the read model, `lineage` is the causal graph, `rsi_telemetry`
is the trace and `loop_kernel` is the durable run. They were useful separately;
a team still had to copy a finished `LoopReport` into all four by hand.
`OpsFabric` is that write path:

* one `RunSummary` per run in the console;
* one lineage node per generation, chained to the previous artifact;
* one `ops.record` span with an event per generation;
* an optional QD snapshot for the capability archive;
* an idempotent `run_id`: recording the same report twice does not duplicate
  lineage nodes or spans.

    ops = OpsFabric(OperationsConsole(), telemetry=RSITelemetry(),
                    qd=quality_diversity_archive)
    ops.record(loop_report, tenant="t1", workspace="ws1")
    ops.snapshot().console, ops.snapshot().lineage, ops.snapshot().telemetry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from .lineage import LineageGraph, LineageReport, lineage_report
from .loop_kernel import LoopReport
from .ops_console import (
    ConsoleSnapshot,
    LineageSnapshot,
    OperationsConsole,
    QDSnapshot,
    RunSummary,
    _RUN_STATUS,
)
from .quality_diversity import QualityDiversityArchive
from .rsi_telemetry import RSITelemetry, TelemetryReport, TelemetrySpan

__all__ = ["OpsRecord", "OpsSnapshot", "OpsFabric"]


@dataclass(frozen=True)
class OpsRecord:
    """The write acknowledgement for one loop report."""

    run: RunSummary
    lineage: LineageSnapshot
    span_id: str = ""
    detail: str = ""


@dataclass(frozen=True)
class OpsSnapshot:
    """The console, lineage, telemetry and QD read models in one object."""

    console: ConsoleSnapshot
    lineage: LineageReport
    telemetry: TelemetryReport
    recorded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "console": self.console.to_dict(),
            "lineage": {
                "nodes": self.lineage.nodes,
                "edges": self.lineage.edges,
                "depth": self.lineage.depth,
                "cycles": [list(cycle) for cycle in self.lineage.cycles],
                "healthy": self.lineage.healthy,
                "detail": self.lineage.detail,
            },
            "telemetry": {
                "spans": self.telemetry.spans,
                "traces": self.telemetry.traces,
                "errors": self.telemetry.errors,
                "wall_time": self.telemetry.wall_time,
                "detail": self.telemetry.detail,
            },
            "recorded": self.recorded,
        }


class OpsFabric:
    """Write a finished loop into the console/lineage/telemetry read models."""

    def __init__(self, console: OperationsConsole, *,
                 lineage: Optional[LineageGraph] = None,
                 telemetry: Optional[RSITelemetry] = None,
                 qd: Optional[QualityDiversityArchive] = None) -> None:
        if not isinstance(console, OperationsConsole):
            raise TypeError("console must be an OperationsConsole")
        if lineage is not None and not isinstance(lineage, LineageGraph):
            raise TypeError("lineage must be a LineageGraph or None")
        if telemetry is not None and not isinstance(telemetry, RSITelemetry):
            raise TypeError("telemetry must be an RSITelemetry or None")
        if qd is not None and not isinstance(qd, QualityDiversityArchive):
            raise TypeError("qd must be a QualityDiversityArchive or None")
        self.console = console
        self.lineage = lineage if lineage is not None else LineageGraph()
        self.telemetry = telemetry
        self.qd = qd
        self._recorded: Dict[str, OpsRecord] = {}
        self._keys: Dict[str, Tuple[str, int, str]] = {}

    def record(self, loop: LoopReport, *, tenant: str, workspace: str,
               tags: Iterable[str] = ()) -> OpsRecord:
        if not isinstance(loop, LoopReport):
            raise TypeError("loop must be a LoopReport")
        if not isinstance(tenant, str) or not tenant.strip():
            raise ValueError("tenant is required")
        if not isinstance(workspace, str) or not workspace.strip():
            raise ValueError("workspace is required")
        if loop.status not in _RUN_STATUS:
            raise ValueError(f"unknown run status {loop.status!r}")
        key = (loop.run_id, len(loop.generations), loop.status)
        cached = self._recorded.get(loop.run_id)
        if cached is not None and self._keys.get(loop.run_id) == key:
            return cached

        status = loop.status
        summary = RunSummary(
            run_id=loop.run_id, tenant=tenant, workspace=workspace,
            status=status, reward=loop.best_reward, cost=loop.total_cost,
            gate=loop.stopped_reason, tags=tuple(str(tag) for tag in tags))
        self.console.upsert_run(summary)

        last_accepted = ""
        span = self._start_span(loop)
        for generation in loop.generations:
            if span is not None and self.telemetry is not None:
                self.telemetry.event(
                    span, "generation",
                    {"generation": generation.generation,
                     "action": generation.action,
                     "reward": generation.reward,
                     "accepted": generation.accepted,
                     "artifact": generation.artifact_id})
            if not generation.artifact_id:
                continue
            if generation.artifact_id not in self.lineage:
                parents = (last_accepted,) if last_accepted else ()
                evidence = getattr(generation, "evidence", {}) or {}
                self.lineage.add_record(
                    generation.artifact_id, parent_ids=parents,
                    run_id=loop.run_id, score=generation.reward,
                    slot=str(evidence.get("slot", "")),
                    metadata={"generation": generation.generation,
                              "accepted": generation.accepted,
                              "applied": generation.applied})
            if generation.accepted:
                last_accepted = generation.artifact_id
        if span is not None and self.telemetry is not None:
            self.telemetry.end_span(
                span, "ok" if loop.terminal else "unset")

        lineage = lineage_report(self.lineage)
        self.console.set_lineage(LineageSnapshot(
            nodes=lineage.nodes, edges=lineage.edges,
            roots=lineage.roots, cycles=len(lineage.cycles)))
        if self.qd is not None:
            best = self.qd.best
            self.console.set_qd(QDSnapshot(
                coverage=self.qd.coverage, qd_score=self.qd.qd_score,
                filled=self.qd.filled,
                best=None if best is None else best.score,
                cells=tuple(str(elite.descriptor)
                            for elite in self.qd.elites)))
        detail = (f"run={loop.run_id}, {len(loop.generations)} generation(s), "
                  f"lineage={len(self.lineage)} node(s)")
        record = OpsRecord(
            run=summary,
            lineage=LineageSnapshot(nodes=lineage.nodes, edges=lineage.edges,
                                    roots=lineage.roots,
                                    cycles=len(lineage.cycles)),
            span_id="" if span is None else span.span_id, detail=detail)
        self._recorded[loop.run_id] = record
        self._keys[loop.run_id] = key
        return record

    def _start_span(self, loop: LoopReport) -> Optional[TelemetrySpan]:
        if self.telemetry is None:
            return None
        return self.telemetry.start_span(
            "ops.record", kind="run",
            attributes={"run_id": loop.run_id, "status": loop.status,
                        "best_reward": loop.best_reward,
                        "total_cost": loop.total_cost})

    def snapshot(self) -> OpsSnapshot:
        telemetry = (self.telemetry.summary() if self.telemetry is not None
                     else TelemetryReport(spans=0, traces=0, roots=0,
                                          errors=0, kinds={}, wall_time=0.0,
                                          span_time=0.0))
        return OpsSnapshot(
            console=self.console.snapshot(),
            lineage=lineage_report(self.lineage),
            telemetry=telemetry, recorded=len(self._recorded))
