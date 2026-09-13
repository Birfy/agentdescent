"""Deliberately wrong versions of the statistics added after `ppi.py`.

The plan's rule: every new piece of statistical logic gets a test that fails
when the logic is written wrong. Not a test of the logic -- a test of the rest
of the suite, proving it can tell right from wrong. A number that looks fine
tells you nothing until you have seen what a broken one looks like.

Each mutation below is a mistake somebody makes, and three of the four were made
here and caught:

    A  discount a *posterior* variance, prior included    made, and fixed
    B  average the strata's residual sds                  the tempting one
    C  forget to divide sigma_eps**2 by n                 silent, scale-free
    D  a fixed-sigma control band                         made, in the plan

They span how loud a statistical error is. A and D produce plausible numbers
that are wrong in a direction nothing downstream notices; B produces **zero**
for a verifier that is badly biased, and C produces a discount independent of
how much evidence there is.
"""

import math
import random

import numpy as np
import pytest

from agentdescent.audit.calibrator import Rectification, population_resid_sd
from agentdescent.audit.drift import DEFAULT_L, DriftMonitor, EWMA
from agentdescent.audit.gate import RectifiedAcceptance, discount_for
from agentdescent.defaults import DefaultAcceptance
from agentdescent.policies import MergeContext
from agentdescent.stats import BetaPosterior


class _S:
    def __init__(self, weight, f, y):
        self.weight, self.f_lab, self.y_lab = weight, f, y


# ---------------------------------------------------------------------------
# A: discount a posterior variance instead of the measurement's
# ---------------------------------------------------------------------------

def _discount_posterior(counts, prior, extra_var, min_kappa=1e-3):
    """The mutation: solve for the variance of the *posterior*, prior included.

    Plausible -- the posterior is what the gate reads, so widening it is what
    you want. The flaw is that the prior cannot be scaled: it is not something
    the verifier measured. Past a certain amount of history the prior alone is
    already narrower than the target, and no discount of the measurement can
    reach it.
    """
    successes, failures = counts
    prior_s, prior_f = prior
    n = successes + failures
    a, b = prior_s + successes + 1.0, prior_f + failures + 1.0
    total = a + b
    p = a / total
    var = p * (1.0 - p) / (total + 1.0)
    if extra_var <= 0.0 or n <= 0.0:
        return 1.0
    wanted = p * (1.0 - p) / (var + extra_var) - 1.0
    return min(1.0, max(min_kappa, (wanted - (prior_s + prior_f + 2.0)) / n))


def test_a_posterior_target_makes_a_long_lived_artifact_immune_to_the_audit():
    """The mutation's discount depends on how many commits an artifact has.

    Which is the tell: the verifier's noise is a fact about the *verifier*, so a
    correction that fades as an artifact accumulates history is correcting the
    wrong thing. Past about forty commits the audit stops working entirely, and
    silently -- the arithmetic returns a number either way.
    """
    counts, extra = (20.0, 12.0), 0.0045
    correct_thin, _, _ = discount_for(counts, extra)
    correct_fat, _, _ = discount_for(counts, extra)
    assert correct_thin == correct_fat, "the real one does not consult a prior"

    mutant_thin = _discount_posterior(counts, (0.0, 0.0), extra)
    mutant_fat = _discount_posterior(counts, (40.0, 40.0), extra)
    assert mutant_thin > 0.5, "premise: with no history the mutation looks fine"
    assert mutant_fat <= 1e-3, (
        "and with forty commits behind it, the mutation annihilates the "
        "measurement instead of widening it")


def test_a_posterior_target_turns_the_audit_into_a_blanket_veto():
    """The same mutation, at the level where it would have been noticed -- late.

    An artifact with forty commits behind it has its measurement discounted to
    the floor, so the posterior is the prior on both sides, `p_improve` is 0.51,
    and **every** candidate is refused however good it is. The audit stops being
    a calibrated discount and becomes a veto that does not read the data -- and
    from outside it looks exactly like a mature artifact that has stopped
    improving, which is the state everyone expects to see.
    """
    inner = DefaultAcceptance(base_delta=0.2, anneal_half_life=64,
                              accept_samples=4000)
    rect = Rectification(
        verifier_version="v1", delta_hat=0.1751, delta_se=0.0287, theta=0.51,
        theta_ci=(0.45, 0.57), se=0.0287, n=177, n_unlab=1000, gain_factor=1.2,
        is_stale=False, stale_reason=None, resid_sd=0.3812)
    prior = BetaPosterior(40.0, 40.0)
    # a candidate nobody would call marginal: 0.50 -> 0.94 on 32 held-out tasks
    base, cand = (16.0, 16.0), (30.0, 2.0)
    ctx = MergeContext(artifact=None, candidate=None, cards=[],
                       base_counts=base, cand_counts=cand, prior=prior)

    assert RectifiedAcceptance(inner, rectification=rect).accept(ctx).accept, (
        "the real gate discounts this candidate's evidence and still commits it")

    shifted = cand[0] - 0.1751 * sum(cand)
    kappa = _discount_posterior((shifted, sum(cand) - shifted), (40.0, 40.0),
                                0.0045)
    mutant = inner.accept(MergeContext(
        artifact=None, candidate=None, cards=[],
        base_counts=(base[0] * kappa, base[1] * kappa),
        cand_counts=(cand[0] * kappa, cand[1] * kappa), prior=prior))
    assert kappa <= 1e-3 and not mutant.accept
    assert mutant.p_improve == pytest.approx(0.5, abs=0.05), (
        "the posterior is the prior on both sides, so the gate is flipping a "
        "coin and calling it a measurement")


# ---------------------------------------------------------------------------
# B: average the strata's residual sds
# ---------------------------------------------------------------------------

def _resid_sd_within_only(strata):
    """The mutation: weight the strata's own sds and stop there.

    It reads as obviously right -- a weighted average of the within-stratum
    spread -- and it throws away every bit of variation *between* strata, which
    on a stratified sample is where a directional bias lives.
    """
    parts = [(s.weight, np.asarray(s.f_lab, float) - np.asarray(s.y_lab, float))
             for s in strata]
    total = sum(w for w, _ in parts)
    return math.sqrt(sum(w * (float(r.var(ddof=1)) if r.size >= 2 else 0.0)
                         for w, r in parts) / total)


def test_averaging_the_within_stratum_sds_reports_zero_for_a_biased_verifier():
    """A verifier +0.4 generous on half the population and exact on the other.

    Every within-stratum sd is zero. The mutation reports **0.0** -- which the
    acceptance gate reads as "this proxy is a measurement" and spends the whole
    held-out set as if an oracle had scored it.
    """
    strata = [_S(0.5, [1.0] * 8, [0.6] * 8), _S(0.5, [1.0] * 8, [1.0] * 8)]
    assert population_resid_sd(strata) == pytest.approx(0.2)
    assert _resid_sd_within_only(strata) == pytest.approx(0.0)

    kappa_real, _, _ = discount_for((20.0, 12.0), 0.2 ** 2 / 32)
    kappa_mutant, _, _ = discount_for((20.0, 12.0), 0.0)
    assert kappa_real < 0.9 and kappa_mutant == 1.0, (
        "the mutation hands the gate every observation as though it were true")


def test_the_two_agree_when_the_strata_do():
    """So the mutation is not caught by a test that forgot to vary the strata --
    which is the shape of test that lets it through."""
    strata = [_S(0.5, [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
              _S(0.5, [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0])]
    assert population_resid_sd(strata) == pytest.approx(
        _resid_sd_within_only(strata))


# ---------------------------------------------------------------------------
# C: forget that sigma_eps is per unit
# ---------------------------------------------------------------------------

def test_forgetting_to_divide_by_n_makes_more_evidence_count_for_less():
    """`sigma_eps ** 2 / n`, not `sigma_eps ** 2`.

    The mutation is invisible in any single measurement: one number, positive,
    of a plausible size. What gives it away is how it moves with the amount of
    evidence.

    The real discount is **scale-free** -- it is the fraction of each
    observation that is signal rather than verifier noise, a per-unit property,
    so a thousand held-out tasks and ten give the same fraction and differ in
    the *effective* count `kappa * n`. The mutation makes the fraction fall as
    `n` grows: the more the run measured, the less of it the gate is allowed to
    believe.
    """
    sigma = 0.3812
    small, big = (6.0, 4.0), (600.0, 400.0)

    real_small, _, _ = discount_for(small, sigma ** 2 / sum(small))
    real_big, _, _ = discount_for(big, sigma ** 2 / sum(big))
    assert real_small == pytest.approx(real_big, abs=0.01), (
        "the same verifier keeps the same fraction of every observation")
    assert real_big * sum(big) > 50 * real_small * sum(small), (
        "what grows with n is the effective count -- a hundredfold larger "
        "held-out set is worth about a hundred times as much")

    mutant_small, _, _ = discount_for(small, sigma ** 2)
    mutant_big, _, _ = discount_for(big, sigma ** 2)
    assert mutant_small > 20 * mutant_big, (
        "under the mutation a hundredfold larger held-out set is worth a "
        "twentieth as much per task")
    assert mutant_big * sum(big) < 2.0, (
        "a thousand held-out tasks reduced to the evidence of two")


# ---------------------------------------------------------------------------
# D: a control band from one fixed sigma
# ---------------------------------------------------------------------------

def _fixed_band(lam, L, sigma, t):
    """The textbook EWMA band: one sigma for every point.

    Correct when the chart's inputs are measurements. Here they are estimates
    whose standard error moves with how many labels a generation bought, and a
    generation that bought few has a `delta_hat` the fixed band calls a shift.
    """
    return L * sigma * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))


def test_a_fixed_band_alarms_on_a_generation_that_merely_bought_fewer_labels():
    """Nothing drifted. One generation was audited a tenth as hard.

    The recursive band widens to absorb it; the fixed band, calibrated on the
    well-audited generations, calls it a shift in the verifier.
    """
    lam, se_good, se_thin = 0.2, 0.01, 0.10
    ewma = EWMA(lam=lam, L=DEFAULT_L, centre=0.0)
    for _ in range(8):
        ewma.observe(0.0, se_good)
    z, band = ewma.observe(0.09, se_thin)      # a thin generation, off-centre

    assert abs(z) < band, "the real band absorbs it"
    assert abs(z) > _fixed_band(lam, DEFAULT_L, se_good, 9), (
        "the fixed band calls the same point a drift")


def test_charting_cumulative_recomputations_is_caught_rather_than_charted():
    """The deployment mistake, which no arithmetic can detect after the fact.

    `Calibrator` recomputes from the whole store, so consecutive points share
    labels and the chart's limits stop applying. Without the `covers` check the
    monitor charts them and the alarms read as drift.
    """
    rng = random.Random(4)
    monitor = DriftMonitor(centre=0.0)
    for g in range(12):
        # every point is computed on all the labels so far
        monitor.observe(Rectification(
            verifier_version="v1", delta_hat=rng.gauss(0.0, 0.01),
            delta_se=0.01, theta=0.5, theta_ci=(0.4, 0.6), se=0.01, n=50 * g + 50,
            n_unlab=1000, gain_factor=1.4, is_stale=False, stale_reason=None,
            resid_sd=0.38, covers=(0.0, 100.0 * (g + 1))))
    assert monitor.report.overlapping, (
        "the one thing that can be checked is whether the windows overlap, and "
        "it is checked")
    assert "control limits do not apply" in monitor.report.to_markdown()
