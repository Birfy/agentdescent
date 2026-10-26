"""Metrics must aggregate decisions deterministically and export a stable text."""

from __future__ import annotations

from dataclasses import dataclass

from agentdescent import (RSIMetrics, alerts, metrics_from_log,
                          metrics_from_records, to_prometheus)


@dataclass
class _Record:
    accepted: bool
    category: str = ""
    cost: dict = None


def test_aggregate_counts_and_cost():
    rows = [
        _Record(True, "session", {"calls": 10, "prompt_tokens": 100,
                                  "completion_tokens": 20, "wallclock": 1.5}),
        _Record(False, "leakage:gold", {"calls": 5}),
        _Record(True, "session", {"calls": 10}),
    ]
    metrics = metrics_from_records(rows)
    assert isinstance(metrics, RSIMetrics)
    assert (metrics.decisions, metrics.accepted, metrics.rejected) == (3, 2, 1)
    assert metrics.acceptance_rate == 2 / 3
    assert metrics.calls == 25 and metrics.cost_per_accepted == 12.5
    assert metrics.categories["leakage:gold"] == 1


def test_dict_records_and_a_log_like_object():
    rows = [{"accepted": True, "category": "x",
             "cost": {"calls": 4, "wallclock": 0.5}}]

    class Log:
        def __iter__(self):
            return iter(rows)

    metrics = metrics_from_log(Log())
    assert metrics.accepted == 1 and metrics.calls == 4


def test_prometheus_output_is_stable_and_names_categories():
    metrics = metrics_from_records([_Record(True, "session", {"calls": 2})])
    text = to_prometheus(metrics)
    assert "agentdescent_rsi_acceptance_rate 1.000000" in text
    assert 'agentdescent_rsi_decisions_by_category_total{category="session"} 1' in text
    assert text == to_prometheus(metrics)


def test_empty_metrics_and_alerts():
    metrics = metrics_from_records([])
    assert metrics.cost_per_accepted is None and alerts(metrics) == []
    assert "agentdescent_rsi_cost_per_accepted NaN" in to_prometheus(metrics)
    assert alerts(metrics_from_records([_Record(False, "x")]),
                  min_acceptance=0.5) == [
        "acceptance rate 0.000 is below 0.500"]


def test_labels_escape_backslash_quote_and_newline():
    text = to_prometheus(metrics_from_records([
        _Record(False, 'a"b\\c\nd', {"calls": 1})]))
    assert 'category="a\\"b\\\\c\\nd"' in text


def test_prometheus_output_round_trips_through_a_real_parser():
    import pytest
    parser = pytest.importorskip("prometheus_client.parser")
    metrics = metrics_from_records([
        _Record(True, "session", {"calls": 2, "prompt_tokens": 3,
                                  "completion_tokens": 4, "wallclock": 0.5}),
        _Record(False, "leakage:gold", {"calls": 1}),
    ])
    families = {f.name: f for f in
                parser.text_string_to_metric_families(to_prometheus(metrics))}
    assert families["agentdescent_rsi_decisions"].type == "counter"
    assert families["agentdescent_rsi_decisions_by_category"].type == "counter"
    assert families["agentdescent_rsi_prompt_tokens"].samples[0].value == 3
    assert families["agentdescent_rsi_accepted"].samples[0].value == 1


def test_malformed_present_fields_are_refused():
    import pytest
    bad = [
        {"accepted": "false", "cost": {}},
        {"accepted": True, "cost": {"calls": "abc"}},
        {"accepted": True, "cost": {"calls": 2.9}},
        {"accepted": True, "cost": {"calls": -5}},
        {"accepted": True, "cost": {"wallclock": float("nan")}},
        {"accepted": True, "cost": 1.0},
    ]
    for row in bad:
        with pytest.raises(ValueError, match="record 0: "):
            metrics_from_records([row])


def test_missing_fields_are_zero_and_objects_are_supported():
    class Cost:
        calls = 4

    @dataclass
    class Row:
        accepted: bool
        category: str = "x"
        cost: object = None

    metrics = metrics_from_records([Row(True, cost=Cost())])
    assert metrics.calls == 4 and metrics.prompt_tokens == 0


def test_alerts_fire_when_nothing_was_accepted_but_calls_were_spent():
    metrics = metrics_from_records([_Record(False, "x", {"calls": 300})])
    assert alerts(metrics, max_cost_per_accepted=1) == [
        "no accepted decisions after 300 calls"]


def test_alert_thresholds_are_validated():
    import pytest
    metrics = metrics_from_records([])
    for value in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError):
            alerts(metrics, min_acceptance=value)
    for value in (-1, float("inf"), "1"):
        with pytest.raises(ValueError):
            alerts(metrics, max_cost_per_accepted=value)
