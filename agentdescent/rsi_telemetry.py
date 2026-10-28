"""A small OpenTelemetry-shaped trace model for RSI runs.

Metrics answer "how many"; traces answer **where the time and money went**.
The framework already has `rsi_metrics`, `manifest`, `diagnose`, `rsi_report`
and `chain_audit`; they are separate read models with no shared span tree. This
module is the common trace/event shape:

``run -> round -> rollout -> merge -> gate -> commit``

A span carries attributes (model, slot, artifact id, parent id, risk, verdict),
events (proposal, reject, adopt, rollback, human-defer), status and timestamps.
``to_otel()`` emits OTel-inspired dictionaries (not OTLP JSON: IDs are hex but
the kind/status/attribute encoding is simplified); ``to_prometheus()`` emits
counter/summary-style samples with escaped labels. No exporter is bundled, so
the module stays standard-library-only and testable with a fake clock.

    telemetry = RSITelemetry()
    with telemetry.span("run", kind="run") as run:
        with telemetry.span("round", kind="round", parent=run) as rnd:
            telemetry.event(rnd, "proposal", {"artifact_id": "a1"})
    report = telemetry.summary()
    otel = telemetry.to_otel()
"""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "TelemetryEvent",
    "TelemetrySpan",
    "TelemetryReport",
    "RSITelemetry",
]


@dataclass
class TelemetryEvent:
    """One point-in-time event inside a span."""

    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TelemetryEvent.name must be non-empty")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("TelemetryEvent.timestamp must be finite")


@dataclass
class TelemetrySpan:
    """One span in the RSI trace tree."""

    trace_id: str
    span_id: str
    name: str
    kind: str
    start: float
    parent_id: Optional[str] = None
    end: Optional[float] = None
    status: str = "unset"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[TelemetryEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TelemetrySpan.name must be non-empty")
        if not self.kind:
            raise ValueError("TelemetrySpan.kind must be non-empty")
        if not math.isfinite(float(self.start)):
            raise ValueError("TelemetrySpan.start must be finite")

    @property
    def duration(self) -> Optional[float]:
        if self.end is None:
            return None
        return max(0.0, float(self.end) - float(self.start))

    @property
    def finished(self) -> bool:
        return self.end is not None

    def set_attribute(self, key: str, value: Any) -> "TelemetrySpan":
        self.attributes[str(key)] = value
        return self

    def add_event(self, name: str, attributes: Optional[Mapping[str, Any]] = None,
                  *, timestamp: Optional[float] = None) -> TelemetryEvent:
        event = TelemetryEvent(name, time.time() if timestamp is None else timestamp,
                               dict(attributes or {}))
        self.events.append(event)
        return event

    def finish(self, status: str = "ok", *, timestamp: Optional[float] = None) -> "TelemetrySpan":
        if status not in ("ok", "error", "cancelled", "unset"):
            raise ValueError("status must be ok, error, cancelled or unset")
        if self.end is not None:
            raise ValueError("span is already finished")
        end = time.time() if timestamp is None else float(timestamp)
        if not math.isfinite(end):
            raise ValueError("end timestamp must be finite")
        if end < self.start:
            raise ValueError("end timestamp must be >= start")
        self.status = status
        self.end = end
        return self


@dataclass(frozen=True)
class TelemetryReport:
    """A read model over a completed (or in-flight) trace set."""

    spans: int
    traces: int
    roots: int
    errors: int
    kinds: Dict[str, int]
    wall_time: float
    span_time: float
    detail: str = ""


class RSITelemetry:
    """Create spans and export them in OTel/Prometheus-friendly shapes."""

    def __init__(self, clock: Optional[Callable[[], float]] = None,
                 id_factory: Optional[Callable[[str], str]] = None) -> None:
        self.clock = clock or time.time
        self.id_factory = id_factory or (
            lambda kind: secrets.token_hex(16 if kind == "trace" else 8))
        self._spans: List[TelemetrySpan] = []

    @property
    def spans(self) -> Tuple[TelemetrySpan, ...]:
        return tuple(self._spans)

    def start_span(self, name: str, kind: str = "span", *,
                   parent: Optional[TelemetrySpan] = None,
                   attributes: Optional[Mapping[str, Any]] = None,
                   trace_id: Optional[str] = None) -> TelemetrySpan:
        if parent is not None:
            if parent not in self._spans:
                raise ValueError("parent span belongs to a different RSITelemetry")
            if trace_id is not None and trace_id != parent.trace_id:
                raise ValueError("trace_id conflicts with the parent span")
            trace_id = parent.trace_id
        if trace_id is None:
            trace_id = self.id_factory("trace")
        span = TelemetrySpan(
            trace_id=trace_id, span_id=self.id_factory("span"),
            name=name, kind=kind, start=float(self.clock()),
            parent_id=parent.span_id if parent is not None else None,
            attributes=dict(attributes or {}))
        self._spans.append(span)
        return span

    def end_span(self, span: TelemetrySpan, status: str = "ok") -> TelemetrySpan:
        return span.finish(status, timestamp=float(self.clock()))

    def event(self, span: TelemetrySpan, name: str,
              attributes: Optional[Mapping[str, Any]] = None) -> TelemetryEvent:
        return span.add_event(name, attributes, timestamp=float(self.clock()))

    def span(self, name: str, kind: str = "span", *,
             parent: Optional[TelemetrySpan] = None,
             attributes: Optional[Mapping[str, Any]] = None) -> "_SpanContext":
        return _SpanContext(self, name, kind, parent, attributes)

    def summary(self) -> TelemetryReport:
        traces = {span.trace_id for span in self._spans}
        kinds: Dict[str, int] = {}
        span_time = 0.0
        wall_time = 0.0
        for span in self._spans:
            kinds[span.kind] = kinds.get(span.kind, 0) + 1
            if span.duration is not None:
                span_time += span.duration
                if span.parent_id is None:
                    wall_time += span.duration
        roots = sum(1 for span in self._spans if span.parent_id is None)
        errors = sum(1 for span in self._spans if span.status == "error")
        return TelemetryReport(
            spans=len(self._spans), traces=len(traces), roots=roots,
            errors=errors, kinds=dict(sorted(kinds.items())),
            wall_time=wall_time, span_time=span_time,
            detail=(f"{len(self._spans)} span(s) over {len(traces)} trace(s), "
                    f"{errors} error(s), wall={wall_time:.6g}s, "
                    f"span-time={span_time:.6g}s"))

    def to_otel(self) -> List[Dict[str, Any]]:
        """OTel-GenAI-shaped dictionaries (no exporter dependency)."""
        out = []
        for span in self._spans:
            out.append({
                "traceId": span.trace_id,
                "spanId": span.span_id,
                "parentSpanId": span.parent_id,
                "name": span.name,
                "kind": span.kind,
                "startTimeUnixNano": int(float(span.start) * 1_000_000_000),
                "endTimeUnixNano": (None if span.end is None
                                    else int(float(span.end) * 1_000_000_000)),
                "attributes": {**dict(span.attributes), "rsi.span.kind": span.kind},
                "events": [
                    {"name": event.name,
                     "timeUnixNano": int(float(event.timestamp) * 1_000_000_000),
                     "attributes": dict(event.attributes)}
                    for event in span.events
                ],
                "status": span.status,
            })
        return out

    @staticmethod
    def _label(value: str) -> str:
        return (str(value).replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n"))

    def to_prometheus(self) -> List[str]:
        """Counter-style samples with escaped labels; no client library."""
        report = self.summary()
        lines = [
            "# TYPE rsi_traces_total counter",
            f"rsi_traces_total {report.traces}",
            "# TYPE rsi_span_errors_total counter",
            f"rsi_span_errors_total {report.errors}",
            "# TYPE rsi_span_seconds summary",
            f"rsi_span_seconds_sum {report.span_time:.6f}",
            f"rsi_span_seconds_count {report.spans}",
            "# TYPE rsi_spans_total counter",
        ]
        for kind, count in report.kinds.items():
            lines.append(f'rsi_spans_total{{kind="{self._label(kind)}"}} {count}')
        return lines


class _SpanContext:
    def __init__(self, telemetry: RSITelemetry, name: str, kind: str,
                 parent: Optional[TelemetrySpan],
                 attributes: Optional[Mapping[str, Any]]) -> None:
        self.telemetry = telemetry
        self.span = telemetry.start_span(name, kind, parent=parent,
                                         attributes=attributes)

    def __enter__(self) -> TelemetrySpan:
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.telemetry.end_span(self.span, "ok")
        else:
            cancelled = not issubclass(exc_type, Exception)
            self.telemetry.end_span(self.span, "cancelled" if cancelled else "error")
            if not cancelled:
                self.telemetry.event(self.span, "exception",
                                     {"type": exc_type.__name__, "message": str(exc)})
        return False
