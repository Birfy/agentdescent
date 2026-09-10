"""The audit reaching the only decision it is allowed to change.

Two claims carry this file.

`test_a_twelve_point_gain_is_inside_the_verifiers_own_noise` is the point of
the module: a candidate that scores 0.625 -> 0.750 on 32 held-out tasks commits
today and does not commit once the verifier's measured disagreement with ground
truth is spent as uncertainty rather than as evidence.

`test_disabled_is_the_same_call_not_an_equivalent_one` is the constraint that
makes that safe to turn on mid-run.
"""

import math
import random

import pytest

from agentdescent.audit.calibrator import (STALE_INFLATION, Rectification,
                                           population_resid_sd)
from agentdescent.audit.gate import (MIN_KAPPA, Adjustment, RectifiedAcceptance,
                                     VerifierWatch, discount_for,
                                     rectified_counts)
from agentdescent.defaults import DefaultAcceptance
from agentdescent.policies import AcceptDecision, MergeContext
from agentdescent.stats import BetaPosterior

# the real Phase 0 numbers, so the tests fail if the arithmetic drifts away from
# what was actually measured
SIGMA_EPS = 0.3812
DELTA_HAT = 0.1751
SE_DELTA = 0.0287
N_HELD = 32


def _rect(**kw):
    base = dict(verifier_version="v1", delta_hat=DELTA_HAT, delta_se=SE_DELTA,
                theta=0.51, theta_ci=(0.45, 0.57), se=SE_DELTA, n=177,
                n_unlab=1000, gain_factor=1.2, is_stale=False,
                stale_reason=None, resid_sd=SIGMA_EPS)
    base.update(kw)
    return Rectification(**base)


def _ctx(base=(20.0, 12.0), cand=(24.0, 8.0), **kw):
    return MergeContext(artifact=None, candidate=None, cards=[],
                        base_counts=base, cand_counts=cand, **kw)


def _inner(**kw):
    kw.setdefault("base_delta", 0.2)
    kw.setdefault("anneal_half_life", 64)
    kw.setdefault("accept_samples", 4000)
    return DefaultAcceptance(**kw)


class _Recording:
    """Remembers the context it was handed, and whether it was the same object."""

    def __init__(self, decision=None):
        self.ctx = None
        self.calls = 0
        self.decision = decision or AcceptDecision(True, "committed", "")

    def accept(self, ctx):
        self.ctx, self.calls = ctx, self.calls + 1
        return self.decision


# -- constraint 1: off is off ------------------------------------------------

def test_disabled_is_the_same_call_not_an_equivalent_one():
    """`enabled=False` must be indistinguishable from not installing the gate.

    Not "produces the same numbers" -- the same object reaches the inner rule
    and the same object comes back, so there is nothing left for a difference to
    hide in. This is what lets the audit be switched on during a production run.
    """
    inner = _Recording()
    ctx = _ctx()
    got = RectifiedAcceptance(inner, rectification=_rect(),
                              enabled=False).accept(ctx)
    assert inner.ctx is ctx
    assert got is inner.decision
    assert inner.calls == 1


def test_disabled_reproduces_the_shipped_gate_exactly_over_many_draws():
    rng = random.Random(20260909)
    off = RectifiedAcceptance(_inner(), rectification=_rect(), enabled=False)
    plain = _inner()
    for _ in range(200):
        n = rng.randint(4, 60)
        ctx = _ctx(base=(float(rng.randint(0, n)), 0.0),
                   cand=(float(rng.randint(0, n)), 0.0))
        ctx = MergeContext(
            artifact=None, candidate=None, cards=[],
            base_counts=(ctx.base_counts[0], n - ctx.base_counts[0]),
            cand_counts=(ctx.cand_counts[0], n - ctx.cand_counts[0]))
        assert off.accept(ctx) == plain.accept(ctx)


def test_a_calibrator_that_has_nothing_to_say_can_still_be_switched_off():
    """The pass-through choice, spelled out.

    `inflate_when_stale=1.0` says "not having measured the verifier" and "the
    verifier is unbiased" are the same claim. They are not, which is why it is
    not the default -- but a run migrating onto the audit needs the option.
    """
    inner = _Recording()
    gate = RectifiedAcceptance(inner, rectification=None,
                               inflate_when_stale=1.0)
    ctx = _ctx()
    gate.accept(ctx)
    assert inner.ctx is ctx
    assert not gate.explain(ctx).applied


# -- the headline ------------------------------------------------------------

def test_a_twelve_point_gain_is_inside_the_verifiers_own_noise():
    """20/32 -> 24/32 commits today. It should not.

    `sigma_eps = 0.381` over 32 held-out tasks is 0.0045 of variance -- 68% as
    large again as the binomial term the gate already carries -- so a little
    over half the apparent evidence for this candidate is the verifier
    disagreeing with the truth rather than the candidate being better.
    """
    ctx = _ctx(base=(20.0, 12.0), cand=(24.0, 8.0))
    plain = _inner().accept(ctx)
    audited = RectifiedAcceptance(_inner(), rectification=_rect()).accept(ctx)

    assert plain.accept, "premise: the shipped gate commits this"
    assert not audited.accept
    assert audited.p_improve < plain.p_improve
    assert "refused by the audit" in audited.detail
    assert "sigma_eps" in audited.detail


def test_two_fifths_of_the_held_out_evidence_is_verifier_noise():
    adj = RectifiedAcceptance(_inner(), rectification=_rect()).explain(_ctx())
    assert 0.55 < adj.evidence_kept < 0.65, (
        "32 tasks scored by this verifier carry about the information of 19 "
        "scored by an oracle")


def test_a_clear_win_still_commits():
    """The audit is a discount on evidence, not a veto."""
    ctx = _ctx(base=(16.0, 16.0), cand=(30.0, 2.0))
    assert RectifiedAcceptance(_inner(), rectification=_rect()).accept(ctx).accept


# -- the cancellation, which is the surprising half --------------------------

def test_the_bias_cancels_out_of_the_comparison():
    """`delta_hat` alone changes almost nothing, and that is the finding.

    One number subtracted from both sides leaves their difference untouched.
    A gate asking "is this better than that" is nearly immune to a uniformly
    generous verifier; what it is not immune to is the *spread*, which is why
    `sigma_eps` and not `delta_hat` is what makes the test above fire.
    """
    ctx = _ctx(base=(20.0, 12.0), cand=(26.0, 6.0))
    plain = _inner().accept(ctx)
    bias_only = RectifiedAcceptance(
        _inner(), rectification=_rect(resid_sd=0.0, se=0.0)).accept(ctx)

    assert bias_only.accept == plain.accept
    assert bias_only.observed_delta == pytest.approx(plain.observed_delta,
                                                     abs=1e-9)
    # not *identical*, because the Beta spread p(1-p) is read at the corrected
    # rate -- the second reason the correction is applied at all
    assert bias_only.p_improve != plain.p_improve


def test_a_positive_bias_alone_can_raise_the_acceptance_rate():
    """The plan's acceptance criterion is "delta_hat > 0 -> fewer commits". Half true.

    `delta_hat` cancels out of the comparison, so its only route to the verdict
    is the Beta spread `p(1-p)`. Subtracting it moves rates **towards** a half
    when they were above it -- wider posterior, fewer commits -- and **away**
    from a half when they were below it, which narrows the posterior and commits
    *more*. Measured over 600 random pairs on 32 held-out tasks:

    ======================  =======  ==================  =====================
    measured rates          plain    delta_hat = 0.175   plus sigma_eps = 0.38
    ======================  =======  ==================  =====================
    0.55 - 0.85              0.608          0.542                0.450
    0.10 - 0.35              0.602        **0.685**              0.257
    ======================  =======  ==================  =====================

    The criterion is really about `resid_sd`, which lowers the rate in both
    regimes because it is uncertainty rather than a shift. That is the whole
    argument of this module in one table.
    """
    rng = random.Random(11)
    inner, n = _inner(), 32

    def acceptance_rate(gate, lo, hi):
        rng.seed(11)
        accepted = 0
        for _ in range(600):
            base = rng.randint(int(lo * n), int(hi * n))
            cand = min(n, base + rng.randint(0, 8))
            accepted += gate.accept(_ctx(
                base=(float(base), float(n - base)),
                cand=(float(cand), float(n - cand)))).accept
        return accepted / 600

    bias_only = RectifiedAcceptance(inner, rectification=_rect(resid_sd=0.0, se=0.0))
    full = RectifiedAcceptance(inner, rectification=_rect())

    high = [acceptance_rate(g, 0.55, 0.85) for g in (inner, bias_only, full)]
    low = [acceptance_rate(g, 0.10, 0.35) for g in (inner, bias_only, full)]

    plain_high, bias_high, full_high = high
    plain_low, bias_low, full_low = low

    assert bias_high < plain_high, "above a half, the bias alone commits less"
    assert bias_low > plain_low, (
        "below a half it commits MORE -- the criterion as written is not what "
        "the correction does")
    assert full_high < plain_high and full_low < plain_low, (
        "sigma_eps lowers the rate in both regimes, because it is uncertainty "
        "and not a shift -- which is what the criterion was reaching for")


def test_the_regression_guard_reads_the_same_rates_after_correction():
    """Shifting both sides by one constant cannot reorder them."""
    gate = RectifiedAcceptance(_inner(), rectification=_rect())
    for base_s, cand_s in [(20, 24), (24, 20), (16, 16), (0, 1), (31, 32)]:
        ctx = _ctx(base=(float(base_s), 32.0 - base_s),
                   cand=(float(cand_s), 32.0 - cand_s))
        adj = gate.explain(ctx)
        got = gate._adjusted(ctx, adj)
        before = MergeContext.rate(ctx.cand_counts) < MergeContext.rate(ctx.base_counts)
        after = MergeContext.rate(got.cand_counts) < MergeContext.rate(got.base_counts)
        assert before == after or not after, (
            "clipping may withdraw a regression flag, never invent one")


def test_a_clipped_shift_narrows_the_gap_rather_than_widening_it():
    """The one asymmetry the correction can introduce, in the safe direction."""
    for delta in (0.4, -0.4):
        (b_s, b_f), _ = rectified_counts((2.0, 30.0), delta)
        (c_s, c_f), _ = rectified_counts((8.0, 24.0), delta)
        gap_before = 8.0 / 32 - 2.0 / 32
        gap_after = c_s / (c_s + c_f) - b_s / (b_s + b_f)
        assert abs(gap_after) <= abs(gap_before) + 1e-12


# -- the arithmetic ----------------------------------------------------------

def test_shifting_preserves_the_number_of_observations():
    (s, f), moved = rectified_counts((22.0, 10.0), 0.175)
    assert s + f == pytest.approx(32.0)
    assert s / 32.0 == pytest.approx(22.0 / 32.0 - 0.175)
    assert moved == pytest.approx(0.175)


def test_shifting_reports_the_shift_it_could_not_make():
    (s, _), moved = rectified_counts((2.0, 30.0), 0.4)
    assert s == pytest.approx(0.0)
    assert moved == pytest.approx(2.0 / 32.0), "clipped, and says so"


def test_the_discount_hits_the_target_variance_exactly():
    """Not an approximation: the equivalent sample size is solved, not fitted.

    `kappa * n` observations of the same rate must have exactly the variance the
    original `n` have *about the truth* -- binomial plus the verifier's own.
    """
    counts, extra = (22.0, 10.0), 0.004
    kappa, var, target = discount_for(counts, extra)
    n = sum(counts)
    p = (counts[0] + 1.0) / (n + 2.0)
    assert var == pytest.approx(p * (1.0 - p) / n)
    assert target == pytest.approx(var + extra)
    assert p * (1.0 - p) / (n * kappa) == pytest.approx(target, rel=1e-12)


def test_the_discount_leaves_the_rate_alone():
    counts = (22.0, 10.0)
    kappa, _, _ = discount_for(counts, 0.01)
    assert kappa < 1.0
    scaled = (counts[0] * kappa, counts[1] * kappa)
    assert MergeContext.rate(scaled) == pytest.approx(MergeContext.rate(counts))


def test_nothing_to_add_means_nothing_is_taken_away():
    kappa, var, target = discount_for((22.0, 10.0), 0.0)
    assert kappa == 1.0 and var == target


def test_the_gates_prior_does_not_enter_the_discount():
    """The measurement is discounted; the artifact's history is not consulted.

    An earlier version of this solved for a *posterior* variance, which made the
    achievable widening depend on how many commits an artifact already had
    behind it: past about forty, the prior alone was narrower than the target
    and the audit could not make the gate doubt its verifier at all. Silently --
    the arithmetic returned a number either way.

    Discounting the measurement alone has no such ceiling. The prior is separate
    evidence and still speaks; it is just not evidence the verifier produced, so
    the verifier's noise says nothing about it.
    """
    counts, extra = (20.0, 12.0), 0.01
    kappa, _, _ = discount_for(counts, extra)
    assert kappa < 1.0

    # the same discount reaches the gate whatever the prior says
    gate = RectifiedAcceptance(_Recording(), rectification=_rect())
    thin = gate.explain(_ctx(prior=BetaPosterior(0.0, 0.0)))
    fat = gate.explain(_ctx(prior=BetaPosterior(400.0, 400.0)))
    assert thin.kappa_cand == pytest.approx(fat.kappa_cand)


def test_the_discount_never_reaches_zero_counts():
    kappa, _, _ = discount_for((20.0, 12.0), 10.0)
    assert kappa == MIN_KAPPA
    scaled = (20.0 * kappa, 12.0 * kappa)
    assert MergeContext.rate(scaled) == pytest.approx(20.0 / 32.0), (
        "an annihilated posterior still has to report an honest rate")


def test_the_drift_allowance_is_carried_once_not_twice():
    """`se(delta)` cancels between the two sides, so adding it to both would be
    double counting; it is split so the *difference* carries it exactly once."""
    gate = RectifiedAcceptance(_inner(), rectification=_rect(resid_sd=0.0))
    adj = gate.explain(_ctx())
    # each side received drift**2 / 2
    _, var, target = discount_for((20.0, 12.0), SE_DELTA ** 2 / 2.0)
    assert target - var == pytest.approx(SE_DELTA ** 2 / 2.0)
    assert adj.drift == pytest.approx(SE_DELTA)


def test_a_caller_can_name_its_own_drift():
    gate = RectifiedAcceptance(_inner(), rectification=_rect(),
                               drift_allowance=0.2)
    assert gate.explain(_ctx()).drift == pytest.approx(0.2)


def test_audit_limited_fires_when_the_correction_is_the_weaker_measurement():
    loose = RectifiedAcceptance(_inner(), rectification=_rect(se=0.4))
    tight = RectifiedAcceptance(_inner(), rectification=_rect())
    assert loose.explain(_ctx()).audit_limited
    assert not tight.explain(_ctx()).audit_limited
    assert "AUDIT-LIMITED" in loose.explain(_ctx()).to_detail()


# -- stale -------------------------------------------------------------------

def test_stale_widens_rather_than_corrects():
    """There is no number to correct with, so the gate spends less instead."""
    ctx = _ctx()
    gate = RectifiedAcceptance(_inner(),
                               rectification=_rect(is_stale=True,
                                                   stale_reason="verifier changed"))
    adj = gate.explain(ctx)
    assert adj.stale and adj.applied
    assert adj.delta_hat == 0.0, "a stale correction must not be applied"
    assert adj.var_after == pytest.approx(adj.var_before * STALE_INFLATION)
    assert adj.kappa_cand == pytest.approx(1.0 / STALE_INFLATION), (
        "the multiplier means exactly this: half the evidence, at any rate "
        "and any sample size")
    assert "verifier changed" in adj.to_detail()


def test_a_rectification_without_a_residual_sd_is_treated_as_stale():
    """The missing term is the one the variance is mostly made of.

    A rectification from before `resid_sd` existed carries `delta_hat` and `se`
    and nothing else. Applying those two and calling it calibrated would be the
    plan's own mistake, made silently.
    """
    adj = RectifiedAcceptance(
        _inner(), rectification=_rect(resid_sd=float("nan"))).explain(_ctx())
    assert adj.stale and adj.delta_hat == 0.0
    assert "residual sd" in adj.reason


def test_no_source_at_all_is_stale_not_a_free_pass():
    adj = RectifiedAcceptance(_inner()).explain(_ctx())
    assert adj.stale and adj.applied


def test_inflate_below_one_is_refused():
    with pytest.raises(ValueError):
        RectifiedAcceptance(_inner(), inflate_when_stale=0.5)


# -- attribution -------------------------------------------------------------

def test_the_refusal_names_the_audit_only_when_the_audit_caused_it():
    marginal = _ctx(base=(20.0, 12.0), cand=(24.0, 8.0))
    hopeless = _ctx(base=(24.0, 8.0), cand=(20.0, 12.0))
    gate = RectifiedAcceptance(_inner(), rectification=_rect())

    assert "refused by the audit" in gate.accept(marginal).detail
    assert "refused by the audit" not in gate.accept(hopeless).detail
    assert "audit" in gate.accept(hopeless).detail, (
        "still says what was applied -- just not that it was the cause")


def test_attribution_is_exact_rather_than_a_second_noisy_draw():
    """The shipped gate seeds its sampler per candidate, so re-running it on the
    same context returns the same number. The attribution is therefore a fact
    about this decision and not a coin flip near the threshold."""
    inner = _inner()
    ctx = _ctx(base=(20.0, 12.0), cand=(24.0, 8.0))
    assert inner.accept(ctx).p_improve == inner.accept(ctx).p_improve


def test_an_accepted_decision_is_returned_untouched():
    inner = _Recording(AcceptDecision(True, "committed", "", p_improve=0.99))
    got = RectifiedAcceptance(inner, rectification=_rect()).accept(_ctx())
    assert got is inner.decision and inner.calls == 1, (
        "no second draw when there is nothing to attribute"
    )


def test_the_wrapper_forwards_install_hooks_to_the_rule_that_decides():
    class _Config:
        base_delta, anneal_half_life, accept_samples = 0.3, 32, 100

    inner = DefaultAcceptance()
    gate = RectifiedAcceptance(inner)
    gate.configure(_Config())
    assert inner.base_delta == 0.3 and inner.accept_samples == 100


def test_the_version_may_be_a_callable_for_a_verifier_that_moves():
    seen = []

    class _Cal:
        def current(self, version):
            seen.append(version)
            return _rect(verifier_version=version)

    gate = RectifiedAcceptance(_inner(), calibrator=_Cal(),
                               verifier_version=lambda: f"v{len(seen)}")
    gate.explain(_ctx())
    gate.explain(_ctx())
    assert seen == ["v0", "v1"]


# -- the population residual sd ---------------------------------------------

class _S:
    def __init__(self, weight, f, y):
        self.weight, self.f_lab, self.y_lab = weight, f, y


def test_the_residual_sd_counts_the_spread_between_strata_too():
    """Two strata each internally uniform, at different offsets.

    Every within-stratum sd is zero. A verifier that is +0.4 generous on half
    the population and exact on the other half has plenty of spread, and an
    average of the strata's own sds would report none of it -- telling the gate a
    proxy was a measurement.
    """
    strata = [_S(0.5, [1.0] * 8, [0.6] * 8), _S(0.5, [1.0] * 8, [1.0] * 8)]
    assert population_resid_sd(strata) == pytest.approx(0.2)


def test_the_residual_sd_matches_the_pooled_sample_when_strata_agree():
    f = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    y = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    one = _S(1.0, f, y)
    resid = [a - b for a, b in zip(f, y)]
    mean = sum(resid) / len(resid)
    biased = math.sqrt(sum((r - mean) ** 2 for r in resid) / len(resid))
    assert population_resid_sd([one]) == pytest.approx(
        biased * math.sqrt(len(resid) / (len(resid) - 1)))


def test_no_strata_is_nan_rather_than_a_confident_zero():
    got = population_resid_sd([])
    assert got != got


# -- the staleness hook ------------------------------------------------------

class _Cal:
    def __init__(self):
        self.reasons = []

    def mark_stale(self, reason):
        self.reasons.append(reason)


class _Art:
    def __init__(self, artifact_id="skills", blast_radius=0.2):
        self.id, self.blast_radius = artifact_id, blast_radius


class _Diff:
    def __init__(self, target="skills", ops=None):
        self.target, self.ops = target, ops or {}


def test_the_first_fingerprint_is_not_a_change():
    cal = _Cal()
    watch = VerifierWatch(cal, fingerprint=lambda: "abc")
    assert not watch.check() and not watch.check()
    assert cal.reasons == []


def test_a_changed_fingerprint_withdraws_the_calibration():
    cal, seen = _Cal(), ["abc"]
    watch = VerifierWatch(cal, fingerprint=lambda: seen[-1])
    watch.check()
    seen.append("def")
    assert watch.check()
    assert "abc" in cal.reasons[0] and "def" in cal.reasons[0]


def test_the_verifier_named_by_id_is_the_precise_signal():
    cal = _Cal()
    watch = VerifierWatch(cal, artifact_ids=["judge_prompt"])
    assert not watch.on_merge(_Art("skills"), _Diff("skills"))
    assert watch.on_merge(_Art("judge_prompt"), _Diff("judge_prompt"))
    assert "it is the verifier" in cal.reasons[0]


def test_a_glob_catches_the_keys_the_verifier_reads():
    cal = _Cal()
    watch = VerifierWatch(cal, key_globs=["rubric.*"])
    assert not watch.on_merge(_Art(), _Diff(ops={"style.tone": "x"}))
    assert watch.on_merge(_Art(), _Diff(ops={"rubric.strictness": "high"}))
    assert "rubric.strictness" in cal.reasons[0]


def test_a_governance_layer_can_be_watched_wholesale():
    from agentdescent.governance import Layer

    cal = _Cal()
    watch = VerifierWatch(cal, layers=[Layer.L1_SLOW])
    assert not watch.on_merge(_Art(blast_radius=0.2), _Diff())
    assert watch.on_merge(_Art(blast_radius=0.6), _Diff())
    assert "L1_SLOW" in cal.reasons[0]


def test_watching_nothing_marks_nothing():
    """The default, and the one that needs saying out loud.

    A run whose verifier is a fixed function is correctly served by watching
    nothing. A run that evolves its own judge and watches nothing gets a
    confident correction for a verifier that no longer exists, with no symptom.
    """
    cal = _Cal()
    watch = VerifierWatch(cal)
    assert not watch.check()
    assert not watch.on_merge(_Art(blast_radius=0.9), _Diff(ops={"a": 1}))
    assert cal.reasons == []
