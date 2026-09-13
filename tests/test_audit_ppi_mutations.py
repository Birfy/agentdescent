"""Deliberately wrong estimators, and the coverage they must destroy.

These do not test `ppi.py`. They test **the rest of the suite** -- specifically
`test_audit_ppi.py`'s coverage run, by proving it can tell right from wrong. A
coverage test that passes at 0.95 tells you nothing until you know it would fail
at 0.50, and the cheapest way to know that is to write the wrong estimator and
watch it fail.

The three mutations are chosen because each is a mistake somebody actually
makes, and because they span the range of how loud a statistical error is:

    A  ignore the stratum weights          catastrophic and obvious
    B  impute: treat `f` as truth          total, and the most tempting
    C  drop lam**2 Var(f_unlab) / N        a few points, and invisible

C is the important one. It is a single missing term, it breaks no invariant, the
point estimate is untouched, and the interval it produces looks entirely normal.
Nothing but a coverage run finds it.

Measured here (400 replications, three strata, n=100 and N=300 each):

    correct                                 0.940
    A  ignore stratum weights               0.010
    B  naive imputation                     0.000
    C  drop the unlabelled variance term    0.907

The plan's own figures were 0.937 -> 0.178 / 0.000 / 0.915 on its author's
setup. B matches exactly, C to within a point; A is more violent here because
these strata means are further apart. The numbers are properties of the
simulation, not constants of the method -- what has to hold is the *ordering*
and the size of the collapse, which is what the assertions below check.
"""

import numpy as np
import pytest

from agentdescent.audit.ppi import Stratum, ppi_mean_stratified, t_ppf

REPS = 400
WEIGHTS = (0.5, 0.3, 0.2)
RATES = (0.8, 0.5, 0.2)


def sample(rep, *, n_per=100, n_unlab=300, bias_dir=0.3):
    """Directional bias, and strata whose means are far apart.

    Both are load-bearing. Symmetric noise would be unbiased and B would look
    fine; equal stratum means would make the weights irrelevant and A would look
    fine. A mutation suite is only as good as the data it is run on.
    """
    rng = np.random.default_rng(rep)
    strata, truth = [], 0.0
    for h, (w, p) in enumerate(zip(WEIGHTS, RATES)):
        y = (rng.random(n_per + n_unlab) < p).astype(float)
        f = np.where((y == 0) & (rng.random(n_per + n_unlab) < bias_dir), 1.0, y)
        strata.append(Stratum(f"s{h}", w, f[:n_per], y[:n_per], f[n_per:]))
        truth += w * p
    return strata, truth


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def mutation_a_pool_the_strata(strata, alpha=0.05):
    """Throw the weights away and treat the labels as one simple random sample.

    The mistake: reading "we sampled 100 from each layer" as "we have 300 units".
    The layers were sampled at different rates *on purpose*, so pooling them
    estimates the mean of the sample rather than of the population.
    """
    pooled = Stratum(
        "pooled", 1.0,
        np.concatenate([s.f_lab for s in strata]),
        np.concatenate([s.y_lab for s in strata]),
        np.concatenate([s.f_unlab for s in strata]))
    return ppi_mean_stratified([pooled], alpha=alpha).ci


def mutation_b_impute(strata, alpha=0.05):
    """Score everything with `f` and call it the answer.

    The most tempting mistake, because it uses all the data and looks like it
    has the largest sample. It is exactly what the audit exists to prevent: it
    reports the verifier's own opinion of itself.
    """
    theta = var = 0.0
    n = 0
    for s in strata:
        allf = np.concatenate([s.f_lab, s.f_unlab])
        theta += s.weight * float(np.mean(allf))
        var += s.weight ** 2 * float(np.var(allf, ddof=1)) / allf.size
        n += allf.size
    se = var ** 0.5
    crit = t_ppf(1 - alpha / 2, n - 1)
    return (theta - crit * se, theta + crit * se)


def mutation_c_drop_unlabelled_variance(strata, alpha=0.05):
    """Keep everything, lose one term of the variance.

    `Var = lam**2 Var(f_unlab)/N + Var(y - lam f)/n`. Drop the first. The point
    estimate does not move, the interval narrows a little, and nothing anywhere
    complains.
    """
    r = ppi_mean_stratified(strata, alpha=alpha)
    var = 0.0
    for s in strata:
        cell = r.per_stratum[s.name]
        unlab = (cell["lambda"] ** 2
                 * float(np.var(s.f_unlab, ddof=1)) / s.n_unlab)
        var += s.weight ** 2 * (cell["var"] - unlab)
    se = var ** 0.5
    crit = t_ppf(1 - alpha / 2, r.df)
    return (r.theta - crit * se, r.theta + crit * se)


def _coverage(interval_fn):
    covered = 0
    for rep in range(REPS):
        strata, truth = sample(rep)
        lo, hi = interval_fn(strata)
        covered += lo <= truth <= hi
    return covered / REPS


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------

def test_the_correct_estimator_covers():
    """The baseline the three below are measured against."""
    got = _coverage(lambda s: ppi_mean_stratified(s).ci)
    assert 0.92 <= got <= 0.97, f"coverage {got}"


def test_mutation_a_ignoring_stratum_weights_destroys_coverage():
    got = _coverage(mutation_a_pool_the_strata)
    assert got < 0.30, (
        f"pooling the strata still covered {got:.3f} of the time. Either the "
        "stratum means were made too similar for the weights to matter, or the "
        "estimator is not using them.")


def test_mutation_b_imputing_the_verifier_as_truth_never_covers():
    got = _coverage(mutation_b_impute)
    assert got < 0.05, (
        f"treating the verifier as ground truth covered {got:.3f} of the time. "
        "On a directionally biased verifier it must never cover -- if it does, "
        "the test data is symmetric and proves nothing (pitfall 4).")


def test_mutation_c_dropping_the_unlabelled_variance_term_is_detectable():
    """The quiet one, and the reason a coverage test exists at all.

    A single missing term. The estimate is unchanged, the interval looks normal,
    and coverage falls a few points -- far too little for anything but a
    replication study to notice, and far too much to ship.
    """
    correct = _coverage(lambda s: ppi_mean_stratified(s).ci)
    got = _coverage(mutation_c_drop_unlabelled_variance)
    assert got < correct - 0.015, (
        f"correct {correct:.3f} vs mutated {got:.3f}: dropping the unlabelled "
        "variance term made no measurable difference. Most likely the "
        "unlabelled set is too large relative to the labelled one for the term "
        "to matter here -- lower `n_unlab` in `sample()` until it does, or this "
        "test is asleep.")
    assert 0.85 <= got <= 0.94, f"expected the documented ~0.91, got {got:.3f}"


def test_the_mutations_are_run_on_data_that_can_expose_them():
    """A mutation suite is only as good as its data; this pins both properties.

    Symmetric noise makes B look correct. Equal stratum means make A look
    correct. Neither failure announces itself, so both premises are asserted
    rather than assumed.
    """
    strata, _ = sample(0)
    residuals = np.concatenate([s.f_lab - s.y_lab for s in strata])
    assert float(np.mean(residuals)) > 0.05, "the verifier must be biased UP"
    assert (residuals >= 0).all(), "the bias must be one-directional"

    means = [float(np.mean(s.y_lab)) for s in strata]
    assert max(means) - min(means) > 0.4, "strata must differ for weights to matter"
