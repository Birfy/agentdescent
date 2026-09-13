"""The estimator has to recover a bias it was given, and stay quiet when there is none.

Two failure directions, and they are not symmetric in cost. An estimator that
misses a real bias leaves a loop optimising a proxy nobody is checking. One that
invents a bias sends a team to build a correction layer for a number that was
noise -- which is exactly the outcome the plan's Phase 0 kill test exists to
prevent, so the false positive is the one the gate is built around.

The pitfall these are written against is IMPLEMENTATION.md's fourth: **symmetric
noise on balanced binary outcomes is unbiased**, so a test built from symmetric
flips lets a wrong estimator pass with no sign of trouble. Every test here that
checks bias detection uses a **directional** perturbation.
"""

import random
import statistics

import pytest

from agentdescent.audit import (AuditRecord, Purpose, bootstrap_ci, hajek_mean,
                                residual_bias, standard_error)


def _rec(f, y, *, prob=1.0, purpose=Purpose.CALIBRATION, rid=None):
    import uuid
    return AuditRecord(
        record_id=rid or uuid.uuid4().hex, task_id="t", artifact_signature="s",
        output="o", verifier_version="v1", verifier_score=f,
        inclusion_prob=prob, purpose=purpose,
        oracle_score=y, resolved_at=(None if y is None else 1.0))


# -- the weighted mean --------------------------------------------------------

def test_with_one_probability_it_is_the_plain_mean():
    vals = [0.0, 1.0, 1.0, 0.5]
    assert hajek_mean(vals, [0.25] * 4) == pytest.approx(statistics.fmean(vals))
    assert hajek_mean(vals, [1.0] * 4) == pytest.approx(statistics.fmean(vals))


def test_stratified_sampling_is_where_the_weighting_earns_its_keep():
    """The unweighted mean is *wrong* here, and plausibly so.

    Population: 100 units with residual 0 and 100 with residual 1, so the true
    mean is 0.5. Sample the second stratum ten times as heavily. An unweighted
    mean of what comes back reads about 0.91; the Hajek mean recovers 0.5.
    """
    values = [0.0] * 10 + [1.0] * 100
    probs = [0.1] * 10 + [1.0] * 100

    assert statistics.fmean(values) == pytest.approx(0.909, abs=0.01)   # the trap
    assert hajek_mean(values, probs) == pytest.approx(0.5, abs=0.01)


def test_a_zero_probability_is_refused_rather_than_dividing():
    with pytest.raises(ValueError, match="strictly positive"):
        hajek_mean([1.0], [0.0])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="probabilities"):
        hajek_mean([1.0, 2.0], [1.0])


# -- the interval -------------------------------------------------------------

def _draw(n, bias_dir, base_rate, rng):
    """One audited sample. `bias_dir` scores *up* only -- never down.

    Directional on purpose: a symmetric flip is unbiased on balanced binary
    outcomes, so a sample built from one would let a broken estimator through.
    """
    res = []
    for _ in range(n):
        y = 1.0 if rng.random() < base_rate else 0.0
        f = 1.0 if (y == 0.0 and rng.random() < bias_dir) else y
        res.append(f - y)
    return res


def test_the_interval_covers_the_truth_about_95_percent_of_the_time():
    """A coverage test, which is the only thing that makes an interval a claim.

    True bias is `bias_dir * P(Y = 0)`. 200 replications at n = 120; the
    bootstrap should cover it close to 95% of the time. The band is wide because
    200 replications have a Monte-Carlo sd of about 1.5pp themselves -- narrower
    would make this flaky rather than strict.
    """
    bias_dir, base_rate, n = 0.4, 0.6, 120
    truth = bias_dir * (1 - base_rate)
    covered = 0
    for rep in range(200):
        rng = random.Random(rep)
        res = _draw(n, bias_dir, base_rate, rng)
        lo, hi = bootstrap_ci(res, [1.0] * n, draws=400, seed=rep)
        covered += lo <= truth <= hi
    assert 0.88 <= covered / 200 <= 1.0, f"coverage {covered / 200}"


def test_a_directional_bias_is_detected():
    rng = random.Random(7)
    n = 200
    res = _draw(n, 0.4, 0.6, rng)
    lo, hi = bootstrap_ci(res, [1.0] * n, seed=7)
    assert lo > 0.0, "a real, one-sided bias must exclude zero"
    assert hajek_mean(res, [1.0] * n) == pytest.approx(0.4 * 0.4, abs=0.08)


def test_symmetric_noise_is_not_reported_as_a_bias():
    """The false positive the kill test is built around.

    A verifier that is merely *noisy* -- wrong in both directions equally -- has
    no bias to correct, and an estimator that reports one would send a team to
    build the whole calibration layer for nothing.
    """
    rng = random.Random(11)
    n = 400
    res = []
    for _ in range(n):
        y = 1.0 if rng.random() < 0.5 else 0.0
        f = (1.0 - y) if rng.random() < 0.3 else y      # flips both ways
        res.append(f - y)
    lo, hi = bootstrap_ci(res, [1.0] * n, seed=11)
    assert lo <= 0.0 <= hi, f"symmetric noise reported as bias: [{lo}, {hi}]"


def test_an_unbiased_verifier_gives_an_interval_containing_zero():
    n = 300
    lo, hi = bootstrap_ci([0.0] * n, [1.0] * n, seed=3)
    assert lo == 0.0 and hi == 0.0


def test_below_two_units_it_says_nan_rather_than_raising():
    lo, hi = bootstrap_ci([1.0], [1.0])
    assert lo != lo and hi != hi          # nan
    assert standard_error([1.0], [1.0]) != standard_error([1.0], [1.0])


def test_the_standard_error_shrinks_with_the_square_root_of_n():
    rng = random.Random(5)
    small = _draw(50, 0.4, 0.6, rng)
    large = _draw(800, 0.4, 0.6, rng)
    se_small = standard_error(small, [1.0] * 50, seed=1)
    se_large = standard_error(large, [1.0] * 800, seed=1)
    assert se_large < se_small
    assert se_small / se_large == pytest.approx((800 / 50) ** 0.5, rel=0.45)


# -- reading it off records ---------------------------------------------------

def test_residual_bias_reads_the_pairs_off_the_store():
    recs = [_rec(1.0, 0.0), _rec(1.0, 1.0), _rec(1.0, 0.0), _rec(0.0, 0.0)]
    got = residual_bias(recs, draws=500)
    assert got["n"] == 4
    assert got["delta"] == pytest.approx(0.5)
    assert got["f_mean"] == pytest.approx(0.75)
    assert got["y_mean"] == pytest.approx(0.25)
    assert got["disagree"] == pytest.approx(0.5)


def test_an_unresolved_record_is_skipped_not_counted_as_agreement():
    """"The oracle has not answered" and "the oracle agreed" are different facts,
    and only one of them is a measurement."""
    recs = [_rec(1.0, 0.0), _rec(1.0, None), _rec(1.0, None)]
    got = residual_bias(recs, draws=500)
    assert got["n"] == 1
    assert got["delta"] == pytest.approx(1.0)


def test_no_resolved_records_reports_nothing_rather_than_zero():
    got = residual_bias([_rec(1.0, None)])
    assert got["n"] == 0
    assert got["delta"] != got["delta"]       # nan, not 0.0


def test_records_carry_their_own_inclusion_probabilities_into_the_estimate():
    """The whole reason `inclusion_prob` is persisted per record."""
    heavy = [_rec(1.0, 0.0, prob=1.0) for _ in range(100)]
    light = [_rec(0.0, 0.0, prob=0.1) for _ in range(10)]
    got = residual_bias(heavy + light, draws=500)
    # unweighted this would read 100/110 = 0.909; weighted it is 0.5
    assert got["delta"] == pytest.approx(0.5, abs=0.01)


# -- clustering ---------------------------------------------------------------
#
# A run scores the same task again for every artifact version, so the audited
# units are not independent draws. Treating them as if they were is the quiet
# way to report an interval that is too narrow, which on this gate means
# "PROCEED" on evidence that did not support it.


def _clustered_sample(n_tasks=80, per_task=3, share_wrong=0.25, seed=0):
    """Residuals that are constant within a task -- maximal dependence."""
    rng = random.Random(seed)
    values, tasks = [], []
    for t in range(n_tasks):
        shared = 1.0 if rng.random() < share_wrong else 0.0
        values.extend([shared] * per_task)
        tasks.extend([t] * per_task)
    return values, tasks


def test_the_clustered_interval_is_wider_than_the_unit_one():
    values, tasks = _clustered_sample()
    probs = [1.0] * len(values)
    n_lo, n_hi = bootstrap_ci(values, probs, seed=0)
    c_lo, c_hi = bootstrap_ci(values, probs, seed=0, clusters=tasks)
    assert (c_hi - c_lo) > (n_hi - n_lo), (
        "resampling tasks must not report a tighter interval than resampling "
        "units -- the whole point is that the units carry less information than "
        "their count suggests")


def test_repeating_every_unit_does_not_make_the_clustered_interval_shrink():
    """The failure the unit bootstrap has: duplicate the data, halve the width.

    Scoring one task under three artifact versions is close to duplication, and a
    unit bootstrap reads the duplicates as three times the evidence.
    """
    base, tasks = _clustered_sample(per_task=1)
    tripled = [v for v in base for _ in range(3)]
    tripled_tasks = [t for t in tasks for _ in range(3)]

    once = bootstrap_ci(base, [1.0] * len(base), seed=1)
    unit = bootstrap_ci(tripled, [1.0] * len(tripled), seed=1)
    clustered = bootstrap_ci(tripled, [1.0] * len(tripled), seed=1,
                             clusters=tripled_tasks)

    assert (unit[1] - unit[0]) < (once[1] - once[0]) * 0.75, "premise"
    assert (clustered[1] - clustered[0]) == pytest.approx(once[1] - once[0], abs=0.05)


def test_independent_units_make_the_two_intervals_agree():
    """No dependence to correct for, so clustering must cost nothing."""
    rng = random.Random(4)
    values = _draw(300, 0.4, 0.6, rng)
    probs = [1.0] * 300
    tasks = list(range(300))                    # one unit per cluster
    n = bootstrap_ci(values, probs, seed=2)
    c = bootstrap_ci(values, probs, seed=2, clusters=tasks)
    assert (c[1] - c[0]) == pytest.approx(n[1] - n[0], abs=0.03)


def test_residual_bias_reports_both_intervals_and_the_cluster_count():
    recs = ([_rec(1.0, 0.0, rid=f"a{i}") for i in range(30)]
            + [_rec(0.0, 0.0, rid=f"b{i}") for i in range(30)])
    got = residual_bias(recs, draws=500)
    assert got["n"] == 60
    assert got["n_tasks"] == 1              # every _rec shares task_id "t"
    assert "ci_clustered" in got and "se_clustered" in got
