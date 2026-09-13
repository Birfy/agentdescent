"""A control chart that does not cry wolf, over inputs that are estimates.

`test_a_naive_test_per_generation_cries_wolf` is why this module is an EWMA
chart and not a test per generation: it measures both against a hundred
in-control generations and counts the alarms.

`test_overlapping_windows_are_the_normal_case_and_invalidate_the_chart` is the
one that catches a real deployment mistake -- `Calibrator` recomputes from the
whole store, so consecutive rectifications share most of their labels and the
limits below stop meaning anything.
"""

import math
import random

import pytest

from agentdescent.audit.calibrator import Rectification
from agentdescent.audit.drift import (DEFAULT_L, DEFAULT_LAMBDA, MIN_GAIN,
                                      DriftKind, DriftMonitor, DriftPoint,
                                      EWMA)


def _rect(delta, se=0.03, *, gain=1.4, covers=(0.0, 0.0), stale=False, n=200):
    return Rectification(
        verifier_version="v1", delta_hat=delta, delta_se=se, theta=0.5,
        theta_ci=(0.4, 0.6), se=se, n=n, n_unlab=1000, gain_factor=gain,
        is_stale=stale, stale_reason="changed" if stale else None,
        resid_sd=0.38, covers=covers)


def _window(i, width=100.0):
    """Disjoint per-generation label windows, which is what a valid chart needs."""
    return (i * width, i * width + width - 1.0)


# -- the smoother ------------------------------------------------------------

def test_the_variance_is_carried_forward_rather_than_assumed_constant():
    """Each `delta_hat` arrives with its own `se`; the closed-form chart cannot
    express that and this one does."""
    ewma = EWMA(lam=0.2)
    _, narrow = ewma.observe(0.0, 0.01)
    assert narrow == pytest.approx(DEFAULT_L * 0.2 * 0.01)

    ewma = EWMA(lam=0.2)
    ewma.observe(0.0, 0.01)
    _, after = ewma.observe(0.0, 0.10)
    expected = DEFAULT_L * math.sqrt((0.2 * 0.10) ** 2 + (0.8 * 0.2 * 0.01) ** 2)
    assert after == pytest.approx(expected)
    assert after > narrow, "a noisy generation widens the band, as it should"


def test_the_band_matches_the_textbook_when_every_point_is_the_same():
    """The closed form is the special case, so it had better fall out."""
    lam, sigma, n = 0.2, 0.05, 12
    ewma = EWMA(lam=lam, L=1.0)
    for _ in range(n):
        _, band = ewma.observe(0.0, sigma)
    closed = sigma * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * n)))
    assert band == pytest.approx(closed, rel=1e-12)


def test_a_lambda_of_one_is_no_smoothing_at_all():
    ewma = EWMA(lam=1.0, L=2.0)
    z, band = ewma.observe(0.3, 0.05)
    assert z == pytest.approx(0.3) and band == pytest.approx(0.1)


def test_a_nonsense_smoother_is_refused_rather_than_run():
    with pytest.raises(ValueError):
        EWMA(lam=0.0)
    with pytest.raises(ValueError):
        EWMA(lam=1.5)
    with pytest.raises(ValueError):
        EWMA(L=0.0)


# -- the reason for the whole module -----------------------------------------

def test_a_naive_test_per_generation_cries_wolf():
    """The plan says do not run a test per generation. This is by how much.

    A hundred in-control generations, five hundred runs. A two-sided test at
    alpha = 0.05 per generation alarms on about five generations per run *by
    construction* -- that is what the alpha means -- and an operator who has
    seen five false alarms will not act on the sixth.
    """
    rng = random.Random(20260910)
    lam, L, se, gens, runs = DEFAULT_LAMBDA, DEFAULT_L, 0.03, 100, 500
    naive_runs = ewma_runs = 0
    naive_alarms = 0
    for _ in range(runs):
        monitor = DriftMonitor(lam=lam, L=L, centre=0.0)
        naive_here = 0
        for g in range(gens):
            value = rng.gauss(0.0, se)
            if abs(value) > 1.96 * se:
                naive_here += 1
            monitor.observe(_rect(value, se, covers=_window(g)))
        naive_alarms += naive_here
        naive_runs += naive_here > 0
        ewma_runs += monitor.report.alarming

    assert naive_alarms / runs == pytest.approx(5.0, abs=1.0), (
        "premise: alpha = 0.05 means five false alarms per hundred generations")
    assert naive_runs / runs > 0.98, "almost every clean run gets a false alarm"
    assert ewma_runs / runs < 0.35, (
        f"the EWMA chart alarms on {ewma_runs / runs:.0%} of clean runs, "
        f"against {naive_runs / runs:.0%} for a test per generation")


def test_a_real_drift_is_still_caught():
    """Smoothing is only worth having if it does not smooth away the thing."""
    rng = random.Random(7)
    monitor = DriftMonitor(centre=0.0)
    caught = None
    for g in range(40):
        drift = 0.004 * g                      # a slow walk upward
        monitor.observe(_rect(rng.gauss(drift, 0.03), 0.03, covers=_window(g)))
        if caught is None and monitor.report.alarming:
            caught = g
    assert caught is not None and caught < 25, (
        "a drift of one standard error over eight generations has to be seen")
    assert monitor.report.of_kind(DriftKind.BIAS_UP)


def test_the_direction_of_the_drift_is_named():
    monitor = DriftMonitor(centre=0.0)
    for g in range(20):
        monitor.observe(_rect(-0.05, 0.02, covers=_window(g)))
    assert monitor.report.of_kind(DriftKind.BIAS_DOWN)
    assert not monitor.report.of_kind(DriftKind.BIAS_UP)
    signal = monitor.report.of_kind(DriftKind.BIAS_DOWN)[0]
    assert "harsher" in signal.detail


def test_the_centre_is_where_watching_started_not_zero():
    """"Has the bias moved" is the question a chart answers well. "Is the bias
    zero" is one the calibrator already answers, with an interval."""
    monitor = DriftMonitor()
    for g in range(20):
        monitor.observe(_rect(0.25, 0.02, covers=_window(g)))
    assert monitor.report.centre == pytest.approx(0.25)
    assert not monitor.report.alarming, (
        "a large but *steady* bias is not drift; it is what the calibrator "
        "subtracts")


# -- validity ----------------------------------------------------------------

def test_overlapping_windows_are_the_normal_case_and_invalidate_the_chart():
    """`Calibrator` recomputes from the whole store every time.

    So consecutive rectifications share most of their labels, are strongly
    positively correlated, and the recursion underestimates the spread of `z` --
    the limits are too tight and the chart alarms on a verifier that never
    moved. The monitor says so rather than charting silently.
    """
    monitor = DriftMonitor(centre=0.0)
    for g in range(5):
        monitor.observe(_rect(0.1, 0.02, covers=(0.0, 100.0 * (g + 1))))
    assert monitor.report.overlapping
    assert monitor.report.of_kind(DriftKind.NOT_INDEPENDENT)
    assert "share labels" in monitor.report.of_kind(DriftKind.NOT_INDEPENDENT)[0].detail
    assert "control limits do not apply" in monitor.report.to_markdown()


def test_the_independence_warning_is_raised_once_not_every_generation():
    monitor = DriftMonitor(centre=0.0)
    for g in range(10):
        monitor.observe(_rect(0.1, 0.02, covers=(0.0, 100.0 * (g + 1))))
    assert len(monitor.report.of_kind(DriftKind.NOT_INDEPENDENT)) == 1


def test_disjoint_windows_raise_no_validity_warning():
    monitor = DriftMonitor(centre=0.0)
    for g in range(10):
        monitor.observe(_rect(0.0, 0.02, covers=_window(g)))
    assert not monitor.report.overlapping


def test_an_unrecorded_window_is_not_evidence_of_overlap():
    """A caller charting hand-built points should not be told its chart is
    invalid because it left a field blank."""
    monitor = DriftMonitor(centre=0.0)
    for _ in range(5):
        monitor.observe(_rect(0.0, 0.02))
    assert not monitor.report.overlapping


def test_the_numbers_are_still_reported_when_the_chart_is_invalid():
    """Withholding them would hide a real drift as thoroughly as a false alarm
    would bury it."""
    monitor = DriftMonitor(centre=0.0)
    for g in range(4):
        monitor.observe(_rect(0.1 * g, 0.02, covers=(0.0, 100.0 * (g + 1))))
    assert len(monitor.report.points) == 4
    assert all(z == z for z in monitor.report.z_bias)


# -- what is skipped ---------------------------------------------------------

def test_a_stale_rectification_is_skipped_rather_than_plotted_twice():
    """It carries the previous generation's numbers forward, so charting it
    would plot the same point again and read as a verifier gone quiet."""
    monitor = DriftMonitor(centre=0.0)
    monitor.observe(_rect(0.1, 0.02, covers=_window(0)))
    assert monitor.observe(_rect(0.1, 0.02, covers=_window(1), stale=True)) == []
    assert len(monitor.report.points) == 1


def test_a_nan_delta_is_not_a_point():
    monitor = DriftMonitor(centre=0.0)
    assert monitor.observe(_rect(float("nan"))) == []
    assert monitor.report.points == []


# -- the gain chart ----------------------------------------------------------

def test_a_gain_factor_falling_to_one_says_replace_the_verifier():
    """The one signal whose remedy is not "buy more labels"."""
    monitor = DriftMonitor(centre=0.0)
    for g in range(15):
        monitor.observe(_rect(0.0, 0.02, gain=1.0, covers=_window(g)))
    signals = monitor.report.of_kind(DriftKind.SIGNAL_LOST)
    assert signals
    assert "Replace the verifier" in signals[0].detail
    assert "more labels do not fix this" in signals[0].detail


def test_the_gain_signal_fires_once():
    monitor = DriftMonitor(centre=0.0)
    for g in range(30):
        monitor.observe(_rect(0.0, 0.02, gain=1.0, covers=_window(g)))
    assert len(monitor.report.of_kind(DriftKind.SIGNAL_LOST)) == 1


def test_a_healthy_gain_factor_raises_nothing():
    monitor = DriftMonitor(centre=0.0)
    for g in range(20):
        monitor.observe(_rect(0.0, 0.02, gain=1.8, covers=_window(g)))
    assert not monitor.report.of_kind(DriftKind.SIGNAL_LOST)


def test_the_gain_chart_is_a_smoothed_threshold_not_a_test():
    """It has no standard error, so there is nothing to put limits around.

    One bad generation must not fire it; a sustained fall must.
    """
    monitor = DriftMonitor(centre=0.0)
    for g in range(10):
        gain = 1.0 if g == 4 else 2.0
        monitor.observe(_rect(0.0, 0.02, gain=gain, covers=_window(g)))
    assert not monitor.report.of_kind(DriftKind.SIGNAL_LOST)
    assert monitor.report.z_gain[-1] > MIN_GAIN


# -- reporting ---------------------------------------------------------------

def test_a_quiet_chart_says_so_rather_than_showing_an_empty_section():
    monitor = DriftMonitor(centre=0.0)
    for g in range(6):
        monitor.observe(_rect(0.0, 0.02, covers=_window(g)))
    text = monitor.report.to_markdown()
    assert "No signal" in text and "6 generations" in text


def test_a_hand_built_point_can_be_charted():
    monitor = DriftMonitor(centre=0.0)
    monitor.observe_point(DriftPoint("gen-1", 0.1, 0.02, 1.5, 100))
    assert monitor.report.points[0].label == "gen-1"


def test_observe_all_returns_the_report():
    rects = [_rect(0.0, 0.02, covers=_window(g)) for g in range(5)]
    report = DriftMonitor(centre=0.0).observe_all(rects)
    assert len(report.points) == 5
