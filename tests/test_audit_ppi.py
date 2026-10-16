"""The statistical core: it must be unbiased, and its interval must mean what it says.

A point estimate is easy to check and easy to get right. The interval is neither,
and it is the part a decision rests on -- the acceptance gate subtracts a
correction *and adds its uncertainty to the variance*, so an interval that is 5%
too narrow makes the gate confident about a number it should be hedging.

So the load-bearing tests here are the two nobody writes by default:
`test_the_interval_covers_at_the_nominal_rate` and
`test_the_reported_standard_error_matches_the_sampling_sd`. Everything else is a
guard around them.

Every test that checks *bias* uses a **directional** perturbation, never a
symmetric flip. Symmetric noise is unbiased on balanced binary outcomes, so a
suite built from it passes against an estimator that has no idea what it is
doing -- the fourth pitfall in `ppi.py`'s module docstring.
"""

import math
from statistics import NormalDist

import numpy as np
import pytest

from agentdescent.audit.ppi import (MIN_N_DOMINANT, PPIError, Stratum,
                                    ppi_mean_stratified, t_ppf)

# ---------------------------------------------------------------------------
# A workload with a verifier that is generous in one direction only.
# ---------------------------------------------------------------------------

WEIGHTS = (0.5, 0.3, 0.2)
RATES = (0.8, 0.5, 0.2)
TRUTH = sum(w * p for w, p in zip(WEIGHTS, RATES))


def sample(rep, *, n_per=100, n_unlab=300, bias_dir=0.3,
           weights=WEIGHTS, rates=RATES):
    """One stratified audit. `bias_dir` only ever scores **up**.

    A wrong answer is forgiven with probability `bias_dir`; a right one is never
    marked down. So `E[f - Y] = bias_dir * P(Y = 0) > 0` and the sign is known.
    """
    rng = np.random.default_rng(rep)
    strata, truth = [], 0.0
    for h, (w, p) in enumerate(zip(weights, rates)):
        y = (rng.random(n_per + n_unlab) < p).astype(float)
        forgiven = (y == 0) & (rng.random(n_per + n_unlab) < bias_dir)
        f = np.where(forgiven, 1.0, y)
        strata.append(Stratum(f"s{h}", w, f[:n_per], y[:n_per], f[n_per:]))
        truth += w * p
    return strata, truth


# ---------------------------------------------------------------------------
# The two that matter
# ---------------------------------------------------------------------------

def test_the_interval_covers_at_the_nominal_rate():
    """400 replications; a 95% interval must contain the truth about 95% of the time.

    This is the only test that makes the interval a claim rather than a
    decoration. The band allows for the ~1.1pp Monte-Carlo sd of 400
    replications; tightening it would buy flakiness, not rigour.
    """
    covered = sum(
        ppi_mean_stratified(sample(rep)[0], seed=rep).ci[0] <= TRUTH
        <= ppi_mean_stratified(sample(rep)[0], seed=rep).ci[1]
        for rep in range(400))
    assert 0.92 <= covered / 400 <= 0.98, f"coverage {covered / 400}"


def test_the_reported_standard_error_matches_the_sampling_sd():
    """What cross-fitting buys, stated as the thing it fixes.

    Fit `lam` on the labels it is then applied to and the residuals look smaller
    than they are, because `lam` was chosen to make them small: the reported SE
    comes in ~7% under the truth and coverage drops to about 0.91. The check is
    the ratio of the mean reported SE to the actual spread of the estimates.
    """
    errs, ses = [], []
    for rep in range(400):
        strata, truth = sample(rep)
        r = ppi_mean_stratified(strata, seed=rep)
        errs.append(r.theta - truth)
        ses.append(r.se)
    ratio = float(np.mean(ses)) / float(np.std(errs))
    assert 0.93 <= ratio <= 1.07, f"reported SE / true sd = {ratio}"


def test_the_estimate_is_unbiased_even_though_the_verifier_is_not():
    """The whole point: `f` is generous by construction, `theta` is not."""
    errs = [ppi_mean_stratified(s, seed=rep).theta - t
            for rep in range(300) for s, t in [sample(rep)]]
    assert abs(float(np.mean(errs))) < 0.01, f"bias {np.mean(errs)}"


def test_cross_fitting_is_not_decoration():
    """Refit `lam` on everything and apply it to everything: the SE shrinks.

    Not a coverage run -- just the mechanism, measured directly, so that a change
    to `_lambda_crossfit` that quietly reverts to a single in-sample coefficient
    fails here in a second rather than in a 400-replication coverage test.
    """
    from agentdescent.audit.ppi import _lambda_from

    tighter = 0
    for rep in range(60):
        strata, _ = sample(rep)
        honest = ppi_mean_stratified(strata, seed=rep)
        # the in-sample version, by hand
        var = 0.0
        for s in strata:
            lam = _lambda_from(s.f_lab, s.y_lab, s.n, s.n_unlab)
            resid = s.y_lab - lam * s.f_lab
            var += s.weight ** 2 * (
                float(np.var(resid, ddof=1)) / s.n
                + lam ** 2 * float(np.var(s.f_unlab, ddof=1)) / s.n_unlab)
        tighter += math.sqrt(var) < honest.se
    assert tighter > 45, (
        f"the in-sample SE was smaller in only {tighter}/60 draws -- "
        "cross-fitting is not doing anything")


# ---------------------------------------------------------------------------
# Degenerate cases the estimator must survive
# ---------------------------------------------------------------------------

def test_a_useless_verifier_costs_nothing():
    """`lam` goes to zero and the answer falls back to the labelled mean.

    The guarantee that makes PPI safe to switch on: a verifier carrying no
    signal cannot make the estimate worse than not having used it.
    """
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.5).astype(float)
    f = rng.random(200)                       # pure noise, unrelated to y
    s = Stratum("only", 1.0, f, y, rng.random(600))
    r = ppi_mean_stratified([s], seed=0)
    assert r.lambda_ < 0.15
    assert r.theta == pytest.approx(float(np.mean(y)), abs=0.03)
    assert 0.95 <= r.gain_factor <= 1.15
    assert any("buying almost nothing" in w for w in r.warnings)


def test_a_perfect_verifier_buys_a_much_tighter_interval():
    rng = np.random.default_rng(1)
    y = (rng.random(120) < 0.4).astype(float)
    s = Stratum("only", 1.0, y.copy(), y, (rng.random(880) < 0.4).astype(float))
    r = ppi_mean_stratified([s], seed=0)
    assert r.lambda_ > 0.8
    assert r.gain_factor > 3.0


def test_no_unlabelled_units_degenerates_to_the_classical_mean():
    """Nothing to borrow, so `lam` must be 0 rather than dividing by zero."""
    rng = np.random.default_rng(2)
    y = (rng.random(50) < 0.6).astype(float)
    f = np.where((y == 0) & (rng.random(50) < 0.3), 1.0, y)
    r = ppi_mean_stratified([Stratum("only", 1.0, f, y, np.array([]))], seed=0)
    assert r.lambda_ == 0.0
    assert r.theta == pytest.approx(float(np.mean(y)))
    assert r.gain_factor == pytest.approx(1.0)


def test_a_constant_verifier_is_survived_rather_than_dividing_by_zero():
    rng = np.random.default_rng(3)
    y = (rng.random(40) < 0.5).astype(float)
    f = np.ones(40)
    r = ppi_mean_stratified([Stratum("only", 1.0, f, y, np.ones(200))], seed=0)
    assert r.lambda_ == 0.0
    assert r.theta == pytest.approx(float(np.mean(y)))


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_weights_that_do_not_sum_to_one_are_refused():
    """Silently normalising would answer a different question, plausibly."""
    strata, _ = sample(0)
    bad = [Stratum(s.name, s.weight * 0.5, s.f_lab, s.y_lab, s.f_unlab)
           for s in strata]
    with pytest.raises(PPIError, match="sum to"):
        ppi_mean_stratified(bad)


def test_a_stratum_with_fewer_than_two_labels_is_refused():
    """The caller turns this into a stale rectifier; it is an early state, not a bug."""
    thin = Stratum("thin", 1.0, np.array([1.0]), np.array([1.0]), np.zeros(10))
    with pytest.raises(PPIError, match="at least 2"):
        ppi_mean_stratified([thin])


def test_unpaired_scores_are_refused():
    with pytest.raises(PPIError, match="paired"):
        Stratum("bad", 1.0, np.zeros(5), np.zeros(4), np.zeros(3))


def test_no_strata_is_refused():
    with pytest.raises(PPIError, match="no strata"):
        ppi_mean_stratified([])


def test_the_same_labels_give_the_same_interval_twice():
    """The fold split is seeded, so a replay is a replay."""
    strata, _ = sample(5)
    a = ppi_mean_stratified(strata, seed=7)
    b = ppi_mean_stratified(strata, seed=7)
    assert (a.theta, a.ci, a.se, a.df) == (b.theta, b.ci, b.se, b.df)
    c = ppi_mean_stratified(strata, seed=8)
    assert c.theta != a.theta, "a different fold split must give a different fit"


# ---------------------------------------------------------------------------
# The reported diagnostics
# ---------------------------------------------------------------------------

def test_resid_sd_is_the_residual_not_the_outcome():
    """The silent trap in Neyman allocation.

    `sd(f - Y)` says where the verifier is least trustworthy; `sd(Y)` says where
    the outcome varies most. A sampler that reads the wrong one sends its oracle
    budget to the wrong layer and nothing errors. This builds a stratum where
    the two point in opposite directions.
    """
    y = np.array([0.0, 1.0] * 30)                 # sd(Y) is large
    f = y.copy()                                  # sd(f - Y) is zero
    r = ppi_mean_stratified([Stratum("h", 1.0, f, y, y.copy())], seed=0)
    assert r.per_stratum["h"]["resid_sd"] == pytest.approx(0.0)
    assert float(np.std(y, ddof=1)) > 0.4         # the value it must not be


def test_the_gain_factor_is_the_effective_sample_size_multiplier():
    strata, _ = sample(0)
    r = ppi_mean_stratified(strata, seed=0)
    classical = 0.0
    for s in strata:
        classical += s.weight ** 2 * float(np.var(s.y_lab, ddof=1)) / s.n
    assert r.gain_factor == pytest.approx(classical / r.se ** 2, rel=1e-9)
    assert r.gain_factor > 1.5


def test_known_limitation_small_dominant_stratum():
    """Locked, because it is a limitation rather than a defect.

    With fewer than `MIN_N_DOMINANT` labels in the heaviest stratum, measured
    coverage is about 0.92 against a nominal 0.95. That is the skew of a binary
    outcome at small n, not an error in the estimator -- but it is exactly the
    kind of thing that gets discovered by someone lowering the sampling floor
    "because the tests still pass". This is the test that stops them.
    """
    covered = 0
    reps = 300
    for rep in range(reps):
        strata, truth = sample(rep, n_per=25, n_unlab=300)
        r = ppi_mean_stratified(strata, seed=rep)
        covered += r.ci[0] <= truth <= r.ci[1]
        if rep == 0:
            assert any("MIN_N_DOMINANT" in w for w in r.warnings), (
                "the shortfall must be reported, not merely suffered")
    rate = covered / reps
    assert 0.86 <= rate <= 0.945, (
        f"coverage {rate} at n=25 -- expected the documented ~0.92 shortfall. "
        "Above the band, the limitation may have been fixed (good: update "
        "MIN_N_DOMINANT and this test). Below it, something else broke.")
    assert MIN_N_DOMINANT == 80


# ---------------------------------------------------------------------------
# t_ppf
# ---------------------------------------------------------------------------

def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _t_cdf(t, df):
    """Exact within floating point, from the regularized incomplete beta."""
    lb = math.lgamma(df / 2 + 0.5) - math.lgamma(df / 2) - math.lgamma(0.5)
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    if x >= 1.0:                      # t == 0 exactly; log1p(-1) is a domain error
        return 0.5
    if x <= 0.0:
        return 1.0 if t > 0 else 0.0
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(lb + a * math.log(x) + b * math.log1p(-x)) * _betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(lb + b * math.log1p(-x) + a * math.log(x)) * \
            _betacf(b, a, 1.0 - x) / b
    p = 0.5 * ib
    return 1.0 - p if t > 0 else p


def _t_ppf_exact(p, df):
    """Independent reference: bisection on `_t_cdf` (kept apart from the library
    on purpose, so the implementation is not checked against itself). The upper
    bracket grows rather than saturating -- at df < 1 the quantile is in the
    hundreds, and a fixed ceiling silently returned the ceiling."""
    lo, hi = -60.0, 60.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    while _t_cdf(lo, df) > p:
        lo *= 2.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_t_ppf_is_accurate_where_the_estimator_uses_it():
    """Checked against the exact quantile, not a three-decimal book table.

    A table lookup would pin this to 1e-3 and call a 1e-3 regression a pass. The
    documented claim is 1.4e-5 for df >= 9, and that is what is asserted.
    """
    worst = max(abs(t_ppf(0.975, df) - _t_ppf_exact(0.975, df))
                for df in (9, 10, 12, 15, 20, 30, 50, 100, 500))
    assert worst < 1.4e-5, f"worst error {worst:.2e}"


@pytest.mark.parametrize("p", [0.9, 0.975, 0.995])
@pytest.mark.parametrize("df", [0.5, 1, 1.5, 2, 3, 5, 8, 9, 30, 1000])
def test_t_ppf_matches_the_exact_quantile_at_every_df_the_estimator_can_reach(p, df):
    """Cluster-robust intervals have `groups - 1` degrees of freedom, so df 1-3
    is an ordinary early state. The expansion alone was 11% narrow at df = 1
    (36% at p = 0.995) and returned 63.9 for an exact 164.6 at df = 0.5."""
    exact = _t_ppf_exact(p, df)
    assert t_ppf(p, df) == pytest.approx(exact, rel=1e-6 if df >= 30 else 1e-9)
    assert t_ppf(1.0 - p, df) == pytest.approx(-exact, rel=1e-6 if df >= 30 else 1e-9)


def test_t_ppf_closed_forms_and_symmetry():
    assert t_ppf(0.975, 1.0) == pytest.approx(12.706204736174694, rel=1e-12)
    assert t_ppf(0.995, 1.0) == pytest.approx(63.65674116287, rel=1e-11)
    assert t_ppf(0.975, 2.0) == pytest.approx(4.302652729749462, rel=1e-12)
    for df in (0.3, 1.0, 2.0, 4.5):
        assert t_ppf(0.5, df) == 0.0
        assert t_ppf(0.2, df) == pytest.approx(-t_ppf(0.8, df), rel=1e-12)


def test_t_ppf_does_not_saturate_for_tiny_df():
    """The bracket grows: df = 0.1 has a 97.5% quantile far above 60."""
    q = t_ppf(0.975, 0.1)
    assert q > 1e4
    assert _t_cdf(q, 0.1) == pytest.approx(0.975, abs=1e-9)


def test_t_ppf_is_continuous_across_the_series_threshold():
    below, above = t_ppf(0.975, 29.999999), t_ppf(0.975, 30.0)
    assert abs(below - above) < 1e-7
    assert above == pytest.approx(_t_ppf_exact(0.975, 30.0), rel=1e-7)


@pytest.mark.parametrize("n_clusters", [2, 3])
def test_interval_half_width_is_the_exact_t_quantile_at_the_reported_df(n_clusters):
    """End to end: the estimator's own `halfwidth / se` at the small df that
    clustering produces equals the exact quantile, not the expansion's."""
    rng = np.random.default_rng(1)
    n = 10
    f = rng.random(n)
    y = f + rng.normal(0, 0.2, n)
    groups = [i * n_clusters // n for i in range(n)]
    s = Stratum("a", 1.0, f, y, f_unlab=rng.random(200), clusters_lab=groups)
    r = ppi_mean_stratified([s])
    assert r.df < 5
    half = 0.5 * (r.ci[1] - r.ci[0])
    assert half / r.se == pytest.approx(_t_ppf_exact(0.975, r.df), rel=1e-9)


def test_t_ppf_is_wider_than_the_normal_quantile_and_converges_like_one_over_df():
    """`t` approaches `z` from above, but slowly -- O(1/df), not fast.

    Worth pinning because the slowness is the reason the `z` shortcut is wrong
    at audit sizes: the leading correction is `(z**3 + z) / (4 df)`, still 2.4e-4
    at df = 10000 and a full 1.2% at df = 40. An earlier version of this test
    asserted convergence to 1e-6 by df = 10000 and failed against a correct
    implementation.
    """
    z = NormalDist().inv_cdf(0.975)
    assert t_ppf(0.975, 10) > t_ppf(0.975, 100) > t_ppf(0.975, 10000) > z
    leading = (z ** 3 + z) / 4.0
    for df in (1000, 10000, 100000):
        assert (t_ppf(0.975, df) - z) == pytest.approx(leading / df, rel=0.02)


def test_a_z_quantile_would_have_been_too_narrow_at_audit_sizes():
    """Pitfall 2, quantified: the shortfall this avoids."""
    z = NormalDist().inv_cdf(0.975)
    for df, floor in ((40, 0.01), (80, 0.005)):
        assert (t_ppf(0.975, df) - z) / z > floor


# ---------------------------------------------------------------------------
# input validation: nothing non-finite becomes a result (#314)
# ---------------------------------------------------------------------------

def _good(n=12, n_unlab=50, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.random(n)
    y = f + rng.normal(0, 0.1, n)
    return f, y, rng.random(n_unlab)


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("weight", [NAN, INF, -INF, -0.5])
def test_a_non_finite_or_negative_weight_is_refused(weight):
    """`abs(nan - 1) > 1e-6` is False, so a NaN weight used to skip the check."""
    f, y, fu = _good()
    with pytest.raises(PPIError):
        Stratum("a", weight, f, y, f_unlab=fu)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
@pytest.mark.parametrize("which", ["f_lab", "y_lab", "f_unlab"])
def test_non_finite_scores_are_refused(which, bad):
    f, y, fu = _good()
    arrs = {"f_lab": f.copy(), "y_lab": y.copy(), "f_unlab": fu.copy()}
    arrs[which][0] = bad
    with pytest.raises(PPIError):
        Stratum("a", 1.0, arrs["f_lab"], arrs["y_lab"], f_unlab=arrs["f_unlab"])


@pytest.mark.parametrize("kwargs", [
    dict(mean_unlab=NAN, var_unlab=0.05),
    dict(mean_unlab=INF, var_unlab=0.05),
    dict(mean_unlab=0.5, var_unlab=NAN),
    dict(mean_unlab=0.5, var_unlab=INF),
    dict(mean_unlab=0.5, var_unlab=0.05, n_unlab=NAN),
    dict(mean_unlab=0.5, var_unlab=0.05, n_unlab=INF),
    dict(mean_unlab=0.5, var_unlab=0.05, n_unlab=2.5),
    dict(mean_unlab=0.5, var_unlab=0.05, n_unlab=-1),
])
def test_non_finite_moments_and_bad_counts_are_refused(kwargs):
    f, y, _ = _good()
    kwargs = {"n_unlab": 50, **kwargs}
    with pytest.raises(PPIError):
        Stratum.from_moments("a", 1.0, f, y, **kwargs)


def test_an_integral_float_count_is_still_accepted_and_normalised():
    f, y, _ = _good()
    s = Stratum.from_moments("a", 1.0, f, y, n_unlab=50.0,
                             mean_unlab=0.5, var_unlab=0.05)
    assert s.n_unlab == 50 and isinstance(s.n_unlab, int)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5, NAN, INF, -INF, "0.05", None])
def test_alpha_outside_the_open_unit_interval_is_a_ppi_error(alpha):
    """Used to be StatisticsError (<= 0), a zero-width interval (1.0) or NaN."""
    f, y, fu = _good()
    with pytest.raises(PPIError):
        ppi_mean_stratified([Stratum("a", 1.0, f, y, f_unlab=fu)], alpha=alpha)


@pytest.mark.parametrize("k", [1, 0, -3, 2.5, None])
def test_k_folds_below_two_is_a_ppi_error(k):
    f, y, fu = _good()
    with pytest.raises(PPIError):
        ppi_mean_stratified([Stratum("a", 1.0, f, y, f_unlab=fu)], k_folds=k)


def test_a_nan_weight_that_bypasses_the_constructor_still_fails_the_sum_check():
    """The check itself is written so NaN fails it, not only the constructor."""
    f, y, fu = _good()
    ok = Stratum("a", 0.5, f, y, f_unlab=fu)
    bad = Stratum("b", 0.5, f, y, f_unlab=fu)
    object.__setattr__(bad, "weight", NAN)
    with pytest.raises(PPIError, match="sum"):
        ppi_mean_stratified([ok, bad])


def test_valid_inputs_are_unchanged_by_the_validation():
    f, y, fu = _good()
    r = ppi_mean_stratified([Stratum("a", 1.0, f, y, f_unlab=fu)], alpha=0.1)
    assert np.isfinite(r.theta) and r.ci[0] < r.theta < r.ci[1]
    assert r.alpha == 0.1
