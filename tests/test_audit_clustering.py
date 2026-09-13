"""Audited units are not independent draws, and the interval said they were.

A run scores the same task again for every artifact version. A task the verifier
is generous about, it is generous about every time. Treating those as separate
observations makes the correction's interval too narrow -- and the acceptance
gate spends that width as `drift`.

`test_coverage_without_clusters_is_79_not_95` is the failure. The two after it
are the fix and the part of it no variance formula can reach.
"""

import math
import statistics

import numpy as np
import pytest

from agentdescent.audit.ppi import (Stratum, cluster_var_of_mean,
                                    ppi_mean_stratified)

TRUTH = 0.5
REPS = 400


def _run(rep, *, n_tasks=200, versions=4, bias=0.3, disjoint=False,
         clustered=True):
    """One audit of a run that scored `n_tasks` tasks under `versions` artifacts.

    The estimand is the **superpopulation** mean: the correction is applied to
    future decisions on other tasks, not to the units already scored. That is
    what makes the clustering bite -- 800 units carry the information of 200.
    """
    rng = np.random.default_rng(rep)
    y_task = (rng.random(n_tasks) < TRUTH).astype(float)
    tasks = np.repeat(np.arange(n_tasks), versions)
    y = np.repeat(y_task, versions)
    # a generous judge, and how generous is a property of the *task*
    forgive = np.repeat(rng.random(n_tasks) < bias, versions)
    f = np.where((y == 0.0) & forgive, 1.0, y)

    if disjoint:
        order = rng.permutation(n_tasks)
        labelled_tasks = set(order[:n_tasks // 3])
        lab = np.array([i for i, t in enumerate(tasks) if t in labelled_tasks])
        unlab = np.array([i for i, t in enumerate(tasks)
                          if t not in labelled_tasks])
    else:
        idx = rng.permutation(y.size)
        cut = y.size // 3
        lab, unlab = idx[:cut], idx[cut:]

    stratum = Stratum("all", 1.0, f[lab], y[lab], f[unlab],
                      clusters_lab=list(tasks[lab]) if clustered else None)
    return ppi_mean_stratified([stratum], alpha=0.05, seed=rep)


def _coverage(**kw):
    hits = sum(1 for rep in range(REPS)
               if (lambda r: r.ci[0] <= TRUTH <= r.ci[1])(_run(rep, **kw)))
    return hits / REPS


# -- the failure -------------------------------------------------------------

def test_coverage_without_clusters_is_79_not_95():
    """The estimator this package shipped with, on data this package produces.

    Nothing about the interval looks wrong. It is simply too narrow, by enough
    that one nominal 95% interval in five misses.
    """
    assert _coverage(clustered=False) < 0.85


def test_cluster_robust_variance_recovers_most_of_it():
    assert _coverage(clustered=True) > 0.88


def test_disjoint_halves_reach_the_nominal_coverage():
    """And this is a sampling design, not an arithmetic fix.

    PPI assumes the labelled and unlabelled halves are independent samples. A
    per-unit inclusion draw puts the same task in both, and no variance formula
    recovers the covariance that omits. Auditing a task whole or not at all does
    -- which is why `AuditedReward` defaults to `draw_by="task"`.
    """
    assert _coverage(clustered=True, disjoint=True) > 0.93


def test_the_three_treatments_are_ordered():
    """Run together so a change that helps one and hurts another is visible."""
    naive = _coverage(clustered=False)
    robust = _coverage(clustered=True)
    designed = _coverage(clustered=True, disjoint=True)
    assert naive < robust < designed, (
        f"{naive:.3f} -> {robust:.3f} -> {designed:.3f}")
    assert designed - naive > 0.10


def test_clustering_moves_the_interval_and_barely_moves_the_estimate():
    """What clustering is for is the width, and it is not *quite* only the width.

    The cross-fitting folds hold out whole groups when clusters are given, so
    `lam` differs and the point estimate moves with it -- by a fraction of a
    standard error, which is the size of "the same estimator, split differently"
    rather than of a change in what is being estimated.
    """
    plain, grouped = _run(0, clustered=False), _run(0, clustered=True)
    assert abs(grouped.theta - plain.theta) < 0.1 * plain.se
    assert grouped.se > plain.se * 1.2
    assert grouped.clustered and not plain.clustered


# -- the variance itself -----------------------------------------------------

def test_singleton_clusters_reproduce_the_independent_estimate():
    """Exactly, not approximately.

    This is what makes the feature safe to switch on: the estimator that ran
    before clusters existed is the degenerate case of this one. The golden
    vectors did not move a digit when it landed -- only two new fields appeared.
    """
    rng = np.random.default_rng(4)
    values = rng.normal(size=40)
    got, groups = cluster_var_of_mean(values, list(range(40)))
    assert groups == 40
    assert got == float(np.var(values, ddof=1)) / 40


def test_a_group_that_is_uniformly_high_counts_once_not_once_per_member():
    """Summing within a group before squaring is the whole difference."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=10)
    values = np.repeat(base, 4)                    # four copies of each task
    clusters = [f"t{i}" for i in range(10) for _ in range(4)]
    robust, groups = cluster_var_of_mean(values, clusters)
    naive = float(np.var(values, ddof=1)) / values.size
    assert groups == 10
    assert robust / naive == pytest.approx(4.0, rel=0.15), (
        "four identical copies per task is a design effect of four")


def test_one_group_is_nan_rather_than_a_confident_zero():
    """Everything measured on one task says nothing about another task."""
    got, groups = cluster_var_of_mean(np.array([1.0, 2.0, 3.0]), ["t", "t", "t"])
    assert groups == 1 and got != got


def test_a_single_cluster_falls_back_and_says_so():
    stratum = Stratum("all", 1.0, [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0],
                      [1.0] * 20, clusters_lab=["t"] * 4)
    got = ppi_mean_stratified([stratum])
    assert any("one cluster" in w for w in got.warnings)
    assert got.se > 0.0


def test_a_negative_design_effect_never_narrows_the_interval():
    """Negative intra-cluster correlation is real and rare. Narrowing an
    interval on an estimate from a handful of clusters is the wrong direction to
    be wrong in, so the effect is floored at 1."""
    # alternating residuals inside each pair cancel, giving a robust variance
    # below the independent one
    f = np.array([1.0, 0.0] * 10)
    y = np.array([0.0, 1.0] * 10)
    clusters = [f"t{i // 2}" for i in range(20)]
    plain = ppi_mean_stratified([Stratum("all", 1.0, f, y, [0.5] * 40)])
    grouped = ppi_mean_stratified(
        [Stratum("all", 1.0, f, y, [0.5] * 40, clusters_lab=clusters)])
    assert grouped.per_stratum["all"]["design_effect"] == 1.0
    assert grouped.se >= plain.se


def test_the_degrees_of_freedom_count_groups_not_units():
    """Forty units from five tasks is five draws' worth of evidence, and the t
    quantile has to know it."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=5)
    f = np.repeat(base, 8) + 0.01 * rng.normal(size=40)
    y = f + 0.3 * np.repeat(rng.normal(size=5), 8)
    grouped = ppi_mean_stratified(
        [Stratum("all", 1.0, f, y, list(rng.normal(size=100)),
                 clusters_lab=[f"t{i // 8}" for i in range(40)])])
    assert grouped.per_stratum["all"]["n_clusters"] == 5.0
    assert grouped.df < 20, "not 39"


def test_cluster_ids_must_be_one_per_labelled_unit():
    from agentdescent.audit.ppi import PPIError

    with pytest.raises(PPIError, match="cluster ids"):
        Stratum("all", 1.0, [1.0, 0.0], [1.0, 1.0], clusters_lab=["t"])


def test_n_clusters_falls_back_to_n_when_there_are_none():
    assert Stratum("all", 1.0, [1.0, 0.0], [1.0, 1.0]).n_clusters == 2


# -- the wiring --------------------------------------------------------------

def test_the_calibrator_clusters_by_task_by_default():
    import uuid

    from agentdescent.audit import AuditRecord, AuditStore, Purpose
    from agentdescent.audit.calibrator import Calibrator

    rng = np.random.default_rng(0)
    store = AuditStore()
    for t in range(40):
        y = 1.0 if rng.random() < 0.5 else 0.0
        forgive = rng.random() < 0.3
        for _ in range(4):                          # four artifact versions
            f = 1.0 if (y == 0.0 and forgive) else y
            store.append(AuditRecord(
                record_id=uuid.uuid4().hex, task_id=f"t{t}",
                artifact_signature="a", output="o", verifier_version="v1",
                verifier_score=f, inclusion_prob=1.0,
                purpose=Purpose.CALIBRATION, oracle_score=y, resolved_at=1.0))
    for _ in range(400):
        store.observe_unlabelled("v1", "all", float(rng.random() < 0.6))

    grouped = Calibrator(store).current("v1")
    independent = Calibrator(store, cluster_by=None).current("v1")
    assert abs(grouped.delta_hat - independent.delta_hat) < 0.1 * independent.se
    assert grouped.se > independent.se * 1.2
    assert not grouped.is_stale


def test_the_tap_audits_a_task_whole_or_not_at_all():
    """`draw_by="task"` is what makes the halves disjoint."""
    from agentdescent.audit import AuditedReward
    from agentdescent.evolution import Task

    audited = AuditedReward(lambda task, out: 1.0, sample_rate=0.5, seed=3)
    task = Task("t1", "q")
    drawn = {audited._draw(task, f"answer {i}") for i in range(20)}
    assert len(drawn) == 1, "every unit of a task draws the same number"

    per_unit = AuditedReward(lambda task, out: 1.0, draw_by="output",
                             sample_rate=0.5, seed=3)
    assert len({per_unit._draw(task, f"answer {i}") for i in range(20)}) > 1


def test_the_purpose_split_is_task_level_too():
    """Otherwise one task's units land in both pools, and the improvement pool's
    edits are informed by a task the calibration pool also measures."""
    from agentdescent.audit import AuditedReward
    from agentdescent.evolution import Task

    audited = AuditedReward(lambda task, out: 1.0, seed=1)
    task = Task("t1", "q")
    assert len({audited._purpose_draw(task, f"a{i}") for i in range(20)}) == 1


def test_an_unknown_draw_by_is_refused():
    from agentdescent.audit import AuditedReward

    with pytest.raises(ValueError, match="draw_by"):
        AuditedReward(lambda task, out: 1.0, draw_by="unit")


def _split_tap(rates, *, draw_by="task", seed=0):
    """A tap whose stratum depends on the score, as the shipped one does."""
    from agentdescent.audit import AuditedReward

    scores = {"hi": 1.0, "lo": 0.0}
    tap = AuditedReward(lambda task, out: scores[out], draw_by=draw_by,
                        sample_rate=min(rates.values()), rates=rates, seed=seed,
                        stratify=lambda t, o, s: "hi" if s >= 0.5 else "lo")
    seen = []
    inner = tap.store.observe_unlabelled
    tap.store.observe_unlabelled = (
        lambda v, s, sc: (seen.append(s), inner(v, s, sc))[1])
    return tap, seen


def test_a_task_is_never_in_both_halves_when_the_strata_differ_in_rate():
    """The draw is task-level; the threshold it meets was not.

    `_key` guarantees every unit of a task draws the same number, and the test
    above pins it. But the number is compared against `rate_for(stratum)`, and
    the shipped `boundary_stratifier` bands by *score* -- so one task scored
    under several artifact versions meets a different threshold each time. A
    task could be labelled in the high-rate band and unlabelled in the low-rate
    one: both halves at once, which is what `draw_by="task"` exists to prevent
    and what costs the interval its coverage. Measured before the fix: 33 of the
    35 eligible tasks out of 40.
    """
    from agentdescent.evolution import Task

    tap, seen = _split_tap({"lo": 0.05, "hi": 0.90})
    both = 0
    for i in range(40):
        task = Task(f"t{i}", "q")
        if not 0.05 <= tap._draw(task, "") < 0.90:
            continue                       # outside the band, never ambiguous
        audited, unlabelled = tap.audited, len(seen)
        tap(task, "hi")
        tap(task, "lo")
        if tap.audited > audited and len(seen) > unlabelled:
            both += 1
    assert both == 0, f"{both} tasks landed in both halves"


def test_the_partition_costs_neither_the_allocation_nor_the_weights():
    """Dropping a unit is only safe if the frame still counts it.

    A unit is labelled exactly when `draw < rate[stratum]`, so Neyman allocation
    still spends labels where it planned to, and it is recorded unlabelled
    exactly when `draw >= max_rate`, which does not mention the stratum.

    Neither of those is the property that matters most, and checking only them
    is how the first version of this fix shipped a worse bug than the one it
    cured. The weights are not read off the unlabelled counts: they are a count
    of **every unit the run scored**, labelled and unlabelled together. Drop a
    unit from that frame and a 50/50 population reads as 83/17 at rates 0.9 and
    0.1 -- a correction that moves rather than one that widens.
    `test_a_skipped_unit_stays_in_the_population_frame` is the one that counts.
    """
    from collections import Counter

    from agentdescent.evolution import Task

    rates = {"lo": 0.05, "hi": 0.90}
    tap, seen = _split_tap(rates)
    labelled = Counter()
    n = 4000
    for i in range(n):
        task = Task(f"t{i}", "q")
        for out in ("hi", "lo"):
            before = tap.audited
            tap(task, out)
            if tap.audited > before:
                labelled[out] += 1
    unlabelled = Counter(seen)

    for stratum, rate in rates.items():
        assert abs(labelled[stratum] / n - rate) < 0.02, stratum
    for stratum in rates:
        assert abs(unlabelled[stratum] / n - 0.10) < 0.02, stratum


def test_one_rate_everywhere_changes_nothing():
    """The drop is a fix for unequal rates and must be inert without them."""
    from agentdescent.evolution import Task

    tap, seen = _split_tap({"lo": 0.5, "hi": 0.5})
    for i in range(200):
        task = Task(f"t{i}", "q")
        tap(task, "hi")
        tap(task, "lo")
    assert tap.skipped == 0
    assert tap.audited + len(seen) == tap.seen, "every unit was accounted for"


def test_drawing_per_output_is_left_alone():
    """`draw_by="output"` never promised whole-task inclusion, and a per-unit
    draw against a per-unit threshold is already internally consistent."""
    from agentdescent.evolution import Task

    tap, _ = _split_tap({"lo": 0.05, "hi": 0.90}, draw_by="output")
    for i in range(200):
        task = Task(f"t{i}", "q")
        tap(task, "hi")
        tap(task, "lo")
    assert tap.skipped == 0


def test_a_skipped_unit_stays_in_the_population_frame():
    """The frame is a census, not a sample, and a dropped unit still happened.

    The first version of this fix dropped the unit outright, on the reasoning
    that a unit is recorded unlabelled at `1 - max_rate` in every stratum, so
    the relative weights were safe. That reasoning was about the wrong counts.
    `observed_weights` and `Calibrator._estimate` both size a stratum as
    labelled + unlabelled -- "the tap saw every unit the run scored" -- so the
    high-rate stratum kept its labelled units while the low-rate one lost the
    same draws to the drop. At rates 0.9 and 0.1 a 50/50 population read as
    83/17, which moves the correction and the next plan toward the high-rate
    stratum rather than costing precision.

    The store counts skipped units per stratum. They belong in the frame and
    nowhere else: they have no score, so they must stay out of `n_unlab` and out
    of the mean PPI borrows, which the next test pins.
    """
    from agentdescent.audit.sampler import observed_weights
    from agentdescent.evolution import Task

    tap, _ = _split_tap({"lo": 0.1, "hi": 0.9})
    n = 6000
    for i in range(n):
        for out in ("hi", "lo"):
            tap(Task(f"t{i}", "q"), out)

    assert tap.skipped > 0, "nothing was dropped; the test proves nothing"
    weights = observed_weights(tap.store, tap.verifier_version)
    for stratum in ("hi", "lo"):
        assert abs(weights[stratum] - 0.5) < 0.01, weights


def test_a_skipped_unit_never_reaches_the_moments_ppi_borrows():
    """It has no score. Folding it into `n` would claim one."""
    from agentdescent.evolution import Task

    tap, seen = _split_tap({"lo": 0.1, "hi": 0.9})
    for i in range(2000):
        for out in ("hi", "lo"):
            tap(Task(f"t{i}", "q"), out)

    moments = tap.store.unlabelled_moments(tap.verifier_version)
    recorded = sum(int(m["n"]) for m in moments.values())
    assert recorded == len(seen), "n counts observed scores and nothing else"
    assert sum(int(m["skipped"]) for m in moments.values()) == tap.skipped
