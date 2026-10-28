"""A proxy that inflates without ordering candidates is a Goodhart alarm."""

import math

import pytest

from agentdescent.goodhart import (
    GoodhartMonitor,
    detect_goodhart,
    spearman,
)


def test_aligned_proxy_and_oracle_are_aligned():
    report = detect_goodhart([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert report.verdict == "aligned"
    assert report.inflation == pytest.approx(0.0)
    assert report.rank_correlation == pytest.approx(1.0)
    assert report.shortcut_rate == 0.0
    assert report.aligned


def test_inflated_proxy_with_no_ordering_is_gaming():
    report = detect_goodhart([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert report.verdict == "gaming"
    assert report.inflation == pytest.approx(1.0)
    assert report.rank_correlation is None
    assert report.shortcut_rate == pytest.approx(1.0)
    assert report.gaming
    assert len(report.shortcuts) == 3


def test_inflation_alone_is_suspicious_not_proof_of_gaming():
    report = detect_goodhart([0.2, 0.3, 0.4], [0.1, 0.2, 0.3])
    assert report.verdict == "suspicious"
    assert report.inflation == pytest.approx(0.1)
    assert report.rank_correlation == pytest.approx(1.0)


def test_rank_inversion_is_caught_even_without_inflation():
    report = detect_goodhart([0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
    assert report.rank_correlation == pytest.approx(-1.0)
    assert report.verdict == "suspicious"


def test_tiny_or_constant_batches_do_not_raise_a_rank_alarm():
    tiny = detect_goodhart([0.1], [0.1])
    assert tiny.verdict == "insufficient"
    assert tiny.rank_correlation is None
    constant = detect_goodhart([0.1] * 3, [0.1] * 3)
    assert constant.verdict == "aligned"
    assert constant.rank_correlation is None
    assert "rank=n/a" in constant.detail


def test_specific_shortcuts_are_reported_sorted_and_truncated():
    report = detect_goodhart([0.2, 0.5, -0.1, 0.3], [0.0, 0.0, 0.1, 0.0],
                             max_shortcuts=2)
    assert report.shortcut_rate == pytest.approx(0.75)
    assert [row.proxy_gain for row in report.shortcuts] == [0.5, 0.3]


def test_shortcuts_keep_the_task_label_through_the_monitor():
    monitor = GoodhartMonitor()
    monitor.observe(0.5, 0.0, "taskA")
    monitor.observe(0.1, 0.1, "taskB")
    monitor.observe(0.2, 0.0, "taskC")
    report = monitor.report()
    assert report.shortcuts[0].task == "taskA"
    assert {row.task for row in report.shortcuts} == {"taskA", "taskC"}
    with pytest.raises(ValueError):
        detect_goodhart([0.1, 0.2], [0.1, 0.2], tasks=["only-one"])


def test_spearman_is_tie_aware_and_constant_safe():
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        spearman([1], [1, 2])
    with pytest.raises(ValueError):
        spearman([], [])


def test_mismatched_empty_and_non_finite_pairs_are_refused():
    with pytest.raises(ValueError):
        detect_goodhart([0.1], [])
    with pytest.raises(ValueError):
        detect_goodhart([], [])
    with pytest.raises(ValueError):
        detect_goodhart([math.nan], [0.1])
    with pytest.raises(ValueError):
        detect_goodhart([0.1], [math.inf])


def test_thresholds_are_validated():
    with pytest.raises(ValueError):
        detect_goodhart([0.1], [0.1], min_rank_correlation=2)
    with pytest.raises(ValueError):
        detect_goodhart([0.1], [0.1], max_shortcut_rate=1.5)
    with pytest.raises(ValueError):
        detect_goodhart([0.1], [0.1], inflation_tolerance=-0.1)
    with pytest.raises(ValueError):
        detect_goodhart([0.1], [0.1], max_shortcuts=-1)


def test_monitor_accumulates_reports_a_window_and_validates_at_observe():
    monitor = GoodhartMonitor()
    monitor.observe(0.2, 0.2, "a")
    monitor.observe(0.4, 0.3, "b")
    monitor.observe(0.6, 0.1, "c")
    report = monitor.report()
    assert report.n == 3
    assert report.inflation == pytest.approx(0.2)
    window = monitor.report(last=2)
    assert window.n == 2
    assert window.inflation == pytest.approx(0.3)
    with pytest.raises(ValueError):
        monitor.report(last=0)
    with pytest.raises(ValueError):
        monitor.observe(math.nan, 0.1)
    monitor.clear()
    with pytest.raises(ValueError):
        monitor.report()
