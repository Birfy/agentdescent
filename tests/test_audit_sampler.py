"""Allocation: does the budget go where the verifier is unreliable?

The failure this guards against does not raise and does not look wrong. A plan
that allocates by population share alone, or by the sd of the *outcome* instead
of the *residual*, produces perfectly reasonable rates and a correction that is
wider than it needed to be. Nobody notices, because there is nothing to compare
against.

So the tests are about *ordering* and *ratios* rather than absolute numbers: the
unreliable layer must get more than the reliable one, and by roughly the factor
Neyman says.
"""

import math

import pytest

from agentdescent.audit import AuditRecord, AuditStore, Purpose
from agentdescent.audit.ppi import MIN_N_DOMINANT
from agentdescent.audit.sampler import (AuditPolicy, boundary_stratifier,
                                        observed_weights, plan, resid_sd_from)

ON = AuditPolicy(enabled=True, min_per_stratum=1, min_dominant=1, max_labels=10_000)


# -- stratifying -------------------------------------------------------------

def test_the_boundary_band_is_where_the_verifier_is_least_sure():
    s = boundary_stratifier(threshold=0.5, width=0.05)
    assert s(None, "", 0.50) == "boundary"
    assert s(None, "", 0.55) == "boundary"
    assert s(None, "", 0.45) == "boundary"
    assert s(None, "", 0.56) == "accepted"
    assert s(None, "", 0.44) == "rejected"
    assert s(None, "", 1.0) == "accepted" and s(None, "", 0.0) == "rejected"


# -- allocation --------------------------------------------------------------

def test_the_unreliable_stratum_gets_more_of_the_budget():
    """Two layers of equal size; one is three times as unreliable.

    Neyman says the labels split in proportion to `W_h * sd_h`, so 3:1.
    """
    p = plan(ON, weights={"a": 0.5, "b": 0.5}, expected_units=10_000,
             resid_sd={"a": 0.3, "b": 0.1})
    assert p.target_n["a"] / p.target_n["b"] == pytest.approx(3.0, rel=0.05)
    assert p.rates["a"] > p.rates["b"]


def test_a_bigger_stratum_gets_more_when_they_are_equally_unreliable():
    p = plan(ON, weights={"big": 0.8, "small": 0.2}, expected_units=10_000,
             resid_sd={"big": 0.2, "small": 0.2})
    assert p.target_n["big"] / p.target_n["small"] == pytest.approx(4.0, rel=0.05)
    # equal reliability means equal *rates* -- proportional allocation
    assert p.rates["big"] == pytest.approx(p.rates["small"], rel=0.05)


def test_size_and_unreliability_trade_off_against_each_other():
    """A small unreliable layer can outrank a large reliable one, and should."""
    p = plan(ON, weights={"big": 0.9, "small": 0.1}, expected_units=10_000,
             resid_sd={"big": 0.02, "small": 0.5})
    assert p.target_n["small"] > p.target_n["big"]


def test_allocating_by_the_outcome_sd_would_be_the_silent_mistake():
    """`resid_sd` and `sd(Y)` can point in opposite directions.

    Layer `a` has a wildly variable outcome that the verifier tracks perfectly;
    layer `b` has a near-constant outcome the verifier keeps getting wrong. A
    plan built on the outcome sd sends the budget to `a`, where a label buys
    nothing.
    """
    p = plan(ON, weights={"a": 0.5, "b": 0.5}, expected_units=10_000,
             resid_sd={"a": 0.01, "b": 0.4})       # residual, not outcome
    assert p.target_n["b"] > 10 * p.target_n["a"]


def test_no_history_allocates_proportionally_and_says_so():
    """Which is right: equal residuals is exactly what proportional assumes."""
    p = plan(ON, weights={"a": 0.7, "b": 0.3}, expected_units=10_000)
    assert p.target_n["a"] / p.target_n["b"] == pytest.approx(7 / 3, rel=0.05)
    assert any("no residual history" in w for w in p.warnings)


def test_a_stratum_with_no_history_is_over_sampled_not_under_sampled():
    """The fill-in value has to be on the measured scale, not a constant.

    An earlier version filled missing strata with 1.0. Against a stratum measured
    at 0.4 that is not a neutral stand-in -- it hands the unmeasured layer two
    and a half times the allocation for no reason but the units the constant
    happened to be written in.

    The largest measured sd instead, and the asymmetry is why: under-sampling a
    stratum nobody has measured keeps it unmeasured, while over-sampling costs
    budget once and self-corrects as soon as there is a real number.
    """
    p = plan(ON, weights={"a": 0.5, "b": 0.5}, expected_units=10_000,
             resid_sd={"a": 0.4})
    assert any("no residual history for ['b']" in w for w in p.warnings)
    assert p.target_n["b"] == p.target_n["a"], (
        "filled at the largest measured sd, so an equally-weighted unmeasured "
        "stratum gets the same allocation")


def test_the_fill_in_never_under_ranks_a_measured_stratum():
    """Three equal layers, two measured. The unknown one is treated as the worst.

    Equal weights on purpose: an earlier version of this test gave the unknown
    stratum a weight of 0.0001, where `min_per_stratum` binds and the assertion
    measured the floor rather than the fill-in.
    """
    third = 1.0 / 3.0
    p = plan(ON, weights={"lo": third, "hi": third, "unknown": third},
             expected_units=100_000, resid_sd={"lo": 0.05, "hi": 0.5})
    assert p.target_n["unknown"] == pytest.approx(p.target_n["hi"], rel=0.05)
    assert p.target_n["unknown"] > 5 * p.target_n["lo"]


# -- the total ---------------------------------------------------------------

def test_the_total_is_the_sample_size_the_requested_halfwidth_needs():
    """`target_halfwidth` is a specification, not a wish.

    Under Neyman allocation `se = sum(W_h sd_h) / sqrt(n)`, so the total follows
    from the half-width directly. Checked against the closed form.
    """
    from statistics import NormalDist
    policy = AuditPolicy(enabled=True, target_halfwidth=0.02,
                         min_per_stratum=1, min_dominant=1, max_labels=10 ** 6)
    weights, sd = {"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.5}
    p = plan(policy, weights=weights, expected_units=10 ** 6, resid_sd=sd)

    z = NormalDist().inv_cdf(0.975)
    wanted = (sum(weights[k] * sd[k] for k in weights) * z / 0.02) ** 2
    assert p.total_n == pytest.approx(wanted, rel=0.02)


def test_a_tighter_halfwidth_costs_quadratically_more_labels():
    def total(halfwidth):
        policy = AuditPolicy(enabled=True, target_halfwidth=halfwidth,
                             min_per_stratum=1, min_dominant=1, max_labels=10 ** 7)
        return plan(policy, weights={"a": 1.0}, expected_units=10 ** 7,
                    resid_sd={"a": 0.4}).total_n

    assert total(0.025) / total(0.05) == pytest.approx(4.0, rel=0.02)


def test_an_unreachable_halfwidth_is_capped_and_flagged_rather_than_billed():
    policy = AuditPolicy(enabled=True, target_halfwidth=0.001, max_labels=300,
                         min_per_stratum=1, min_dominant=1)
    p = plan(policy, weights={"a": 1.0}, expected_units=10 ** 6,
             resid_sd={"a": 0.4})
    assert p.total_n == 300
    assert any("capped at max_labels" in w for w in p.warnings)


# -- floors ------------------------------------------------------------------

def test_the_floors_are_applied_after_the_allocation_not_before():
    """A bound layer must not shrink the others proportionally.

    The tiny layer is raised to its floor; the big one keeps the allocation
    Neyman gave it rather than being scaled down to fit.
    """
    policy = AuditPolicy(enabled=True, target_halfwidth=0.05,
                         min_per_stratum=20, min_dominant=1, max_labels=10 ** 6)
    unfloored = plan(AuditPolicy(enabled=True, target_halfwidth=0.05,
                                 min_per_stratum=1, min_dominant=1,
                                 max_labels=10 ** 6),
                     weights={"big": 0.99, "tiny": 0.01}, expected_units=10 ** 5,
                     resid_sd={"big": 0.4, "tiny": 0.4})
    floored = plan(policy, weights={"big": 0.99, "tiny": 0.01},
                   expected_units=10 ** 5, resid_sd={"big": 0.4, "tiny": 0.4})
    assert floored.target_n["tiny"] == 20
    assert floored.target_n["big"] == unfloored.target_n["big"]


def test_the_dominant_stratum_floor_is_the_coverage_floor():
    """The same number as `MIN_N_DOMINANT`, and for the same reason.

    Below it the reported interval covers about 0.92 rather than 0.95. A plan
    that allocates under the floor and a warning that fires at the floor would
    otherwise be two knobs describing one fact, free to drift apart.
    """
    assert AuditPolicy().min_dominant == MIN_N_DOMINANT
    p = plan(AuditPolicy(enabled=True, target_halfwidth=0.5),
             weights={"a": 0.9, "b": 0.1}, expected_units=10 ** 5,
             resid_sd={"a": 0.01, "b": 0.01})
    assert p.target_n["a"] >= MIN_N_DOMINANT


def test_floors_that_exceed_the_cap_win_and_say_why():
    policy = AuditPolicy(enabled=True, min_per_stratum=50, min_dominant=50,
                         max_labels=60)
    p = plan(policy, weights={"a": 0.5, "b": 0.5}, expected_units=10 ** 5,
             resid_sd={"a": 0.4, "b": 0.4})
    assert p.total_n == 100, "the floors are what make a stratum readable at all"
    assert any("floors win" in w for w in p.warnings)


# -- rates -------------------------------------------------------------------

def test_rates_are_the_allocation_divided_by_the_expected_population():
    p = plan(ON, weights={"a": 0.5, "b": 0.5}, expected_units=1000,
             resid_sd={"a": 0.3, "b": 0.1})
    for k in ("a", "b"):
        assert p.rates[k] == pytest.approx(p.target_n[k] / (0.5 * 1000))


def test_a_rate_is_capped_at_one_and_the_shortfall_is_reported():
    p = plan(ON, weights={"a": 1.0}, expected_units=10,
             resid_sd={"a": 0.4})
    assert p.rates["a"] == 1.0
    assert any("still falling short" in w for w in p.warnings)


def test_an_unplanned_stratum_gets_a_rate_of_zero_not_a_default():
    """Sampling a layer the plan never saw would record an inclusion probability
    nobody chose, which is the one field that cannot be reconstructed later."""
    p = plan(ON, weights={"a": 1.0}, expected_units=1000, resid_sd={"a": 0.4})
    assert p.rate_for("a") > 0.0
    assert p.rate_for("surprise") == 0.0


def test_a_disabled_policy_plans_nothing_rather_than_a_little():
    p = plan(AuditPolicy(enabled=False), weights={"a": 1.0}, expected_units=1000)
    assert p.rates == {} and p.rate_for("a") == 0.0
    assert any("disabled" in w for w in p.warnings)


def test_no_strata_is_an_empty_plan_rather_than_an_error():
    p = plan(ON, weights={}, expected_units=1000)
    assert p.rates == {} and any("no strata" in w for w in p.warnings)


# -- reading the inputs off a run --------------------------------------------

def _rec(stratum, version="v1"):
    import uuid
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id="t", artifact_signature="s",
        output="o", verifier_version=version, verifier_score=1.0,
        inclusion_prob=1.0, purpose=Purpose.CALIBRATION, stratum=stratum,
        oracle_score=1.0, resolved_at=1.0)


def test_weights_come_from_both_halves_of_a_previous_run():
    """Labelled records plus unlabelled moments account for every unit scored."""
    store = AuditStore()
    for _ in range(10):
        store.append(_rec("a"))
    for _ in range(5):
        store.append(_rec("b"))
    for _ in range(90):
        store.observe_unlabelled("v1", "a", 1.0)
    for _ in range(95):
        store.observe_unlabelled("v1", "b", 0.0)

    w = observed_weights(store, "v1")
    assert w["a"] == pytest.approx(0.5) and w["b"] == pytest.approx(0.5)
    assert sum(w.values()) == pytest.approx(1.0)


def test_weights_ignore_other_verifier_versions():
    store = AuditStore()
    store.append(_rec("a", version="v1"))
    for _ in range(99):
        store.observe_unlabelled("v1", "a", 1.0)
    store.append(_rec("b", version="v2"))
    for _ in range(500):
        store.observe_unlabelled("v2", "b", 1.0)
    assert set(observed_weights(store, "v1")) == {"a"}


def test_resid_sd_is_read_off_a_previous_estimate():
    import numpy as np
    from agentdescent.audit.ppi import Stratum, ppi_mean_stratified

    rng = np.random.default_rng(0)
    strata = []
    for name, w, noise in (("a", 0.6, 0.0), ("b", 0.4, 0.5)):
        y = (rng.random(80) < 0.5).astype(float)
        f = np.where((y == 0) & (rng.random(80) < noise), 1.0, y)
        strata.append(Stratum(name, w, f, y, (rng.random(200) < 0.5).astype(float)))
    got = resid_sd_from(ppi_mean_stratified(strata, seed=0))
    assert set(got) == {"a", "b"}
    assert got["b"] > got["a"], "the noisy stratum must show the larger residual sd"
    assert resid_sd_from(None) == {}


def test_the_whole_loop_closes_from_one_run_to_the_next_plan():
    """Counts and residuals from a run become the next run's rates."""
    store = AuditStore()
    for _ in range(40):
        store.append(_rec("boundary"))
    for _ in range(60):
        store.append(_rec("accepted"))
    for _ in range(400):
        store.observe_unlabelled("v1", "boundary", 0.5)
    for _ in range(3000):
        store.observe_unlabelled("v1", "accepted", 1.0)

    w = observed_weights(store, "v1")
    p = plan(AuditPolicy(enabled=True, min_per_stratum=10, min_dominant=20),
             weights=w, expected_units=4000,
             resid_sd={"boundary": 0.45, "accepted": 0.05})

    assert p.rates["boundary"] > p.rates["accepted"], (
        "the boundary layer is both smaller and less reliable, so it must be "
        "sampled at a higher rate")
    assert all(0.0 <= r <= 1.0 for r in p.rates.values())
    assert p.expected_units == 4000
