"""A trace must answer where a run's time and errors went."""

import pytest

from agentdescent.rsi_telemetry import (
    RSITelemetry,
    TelemetryEvent,
    TelemetrySpan,
)


def _ids():
    state = {"trace": 0, "span": 0}

    def make(kind):
        state[kind] += 1
        return f"{kind}-{state[kind]}"

    return make


def test_context_manager_builds_a_parent_child_tree():
    ticks = iter(range(100))
    telemetry = RSITelemetry(clock=lambda: float(next(ticks)),
                             id_factory=_ids())
    with telemetry.span("run", kind="run") as run:
        with telemetry.span("round", kind="round", parent=run) as rnd:
            telemetry.event(rnd, "proposal", {"artifact_id": "a1"})
            with telemetry.span("gate", kind="gate", parent=rnd):
                pass
    assert len(telemetry.spans) == 3
    assert run.trace_id == rnd.trace_id == "trace-1"
    assert rnd.parent_id == run.span_id
    assert run.parent_id is None
    assert run.status == "ok"
    assert rnd.events[0].name == "proposal"
    assert run.duration is not None and run.duration >= 0


def test_report_separates_wall_time_from_span_time():
    ticks = iter(range(100))
    telemetry = RSITelemetry(clock=lambda: float(next(ticks)),
                             id_factory=_ids())
    with pytest.raises(RuntimeError):
        with telemetry.span("run", kind="run") as run:
            with telemetry.span("bad", kind="rollout", parent=run):
                raise RuntimeError("boom")
    report = telemetry.summary()
    assert report.spans == 2
    assert report.traces == 1
    assert report.roots == 1
    assert report.errors == 2
    assert report.kinds == {"rollout": 1, "run": 1}
    assert report.span_time > report.wall_time > 0
    assert "wall=" in report.detail and "span-time=" in report.detail


def test_otel_shape_and_prometheus_samples():
    ticks = iter(range(100))
    telemetry = RSITelemetry(clock=lambda: float(next(ticks)),
                             id_factory=_ids())
    with telemetry.span("run", kind="run", attributes={"slot": "selection"}):
        pass
    otel = telemetry.to_otel()
    assert otel[0]["traceId"] == "trace-1"
    assert otel[0]["parentSpanId"] is None
    assert otel[0]["attributes"] == {"slot": "selection", "rsi.span.kind": "run"}
    assert otel[0]["status"] == "ok"
    assert otel[0]["startTimeUnixNano"] < otel[0]["endTimeUnixNano"]
    lines = telemetry.to_prometheus()
    assert "rsi_spans_total 1" not in lines
    assert 'rsi_spans_total{kind="run"} 1' in lines
    assert "# TYPE rsi_spans_total counter" in lines
    assert any(line.startswith("rsi_span_seconds_sum") for line in lines)


def test_ids_are_unique_across_instances():
    one = RSITelemetry()
    two = RSITelemetry()
    first = one.start_span("a")
    second = two.start_span("b")
    assert first.trace_id != second.trace_id
    assert first.span_id != second.span_id


def test_prometheus_label_value_is_escaped():
    telemetry = RSITelemetry(id_factory=_ids())
    with telemetry.span("x", kind='ev"il\nfoo 1'):
        pass
    text = "\n".join(telemetry.to_prometheus())
    assert '\\"il\\nfoo 1' in text
    assert 'kind="ev"il' not in text


def test_manual_span_event_and_finish():
    telemetry = RSITelemetry(clock=lambda: 1.0, id_factory=_ids())
    span = telemetry.start_span("manual", kind="merge")
    telemetry.event(span, "adopt", {"artifact_id": "a"})
    telemetry.end_span(span, "cancelled")
    assert span.finished and span.status == "cancelled"
    assert span.events[0].attributes["artifact_id"] == "a"


def test_cancellation_is_not_recorded_as_error():
    telemetry = RSITelemetry(id_factory=_ids())
    with pytest.raises(KeyboardInterrupt):
        with telemetry.span("cancelled") as span:
            raise KeyboardInterrupt()
    assert span.status == "cancelled"
    assert not any(event.name == "exception" for event in span.events)


def test_exception_is_recorded_as_an_event():
    telemetry = RSITelemetry(id_factory=_ids())
    with pytest.raises(RuntimeError):
        with telemetry.span("bad") as span:
            raise RuntimeError("boom")
    assert span.status == "error"
    assert span.events[-1].name == "exception"
    assert span.events[-1].attributes["type"] == "RuntimeError"


def test_invalid_names_statuses_and_timestamps_are_refused():
    with pytest.raises(ValueError):
        TelemetryEvent("", 1.0)
    with pytest.raises(ValueError):
        TelemetryEvent("e", float("nan"))
    with pytest.raises(ValueError):
        TelemetrySpan("t", "s", "", "run", 1.0)
    with pytest.raises(ValueError):
        TelemetrySpan("t", "s", "n", "", 1.0)
    with pytest.raises(ValueError):
        TelemetrySpan("t", "s", "n", "run", float("inf"))
    telemetry = RSITelemetry(clock=lambda: 1.0, id_factory=_ids())
    span = telemetry.start_span("x")
    with pytest.raises(ValueError):
        telemetry.end_span(span, "magic")
    with pytest.raises(ValueError):
        span.finish(timestamp=float("nan"))
    with pytest.raises(ValueError):
        span.finish(timestamp=0.5)
    telemetry.end_span(span)
    with pytest.raises(ValueError):
        telemetry.end_span(span)


def test_mismatched_parent_or_trace_is_refused():
    one = RSITelemetry(id_factory=_ids())
    two = RSITelemetry(id_factory=_ids())
    parent = one.start_span("parent")
    with pytest.raises(ValueError):
        two.start_span("child", parent=parent)
    with pytest.raises(ValueError):
        one.start_span("child", parent=parent, trace_id="other")
