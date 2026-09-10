"""Where the improvement labels go, which is not where the calibration ones do.

`test_a_layer_with_a_hundred_agreeing_labels_is_not_unsampled` is the bug this
file exists to keep fixed. Counting only the *disagreements* as Good-Turing
draws made a layer where the verifier had never once been wrong look untouched,
so it scored 1.0 and drew the whole budget.
"""

import math

import pytest

from agentdescent.audit.coverage import (MIN_UNSEEN, Coverage, coverage_of,
                                         exhausted, plan_coverage, rarefaction,
                                         unseen_mass, unseen_mass_overall)


class _Rec:
    """Only the three fields coverage reads."""

    def __init__(self, f, y, key="k", mode=None):
        self.verifier_score, self.oracle_score = f, y
        self.key, self.mode = key, mode


def _cov(records, keys=()):
    return coverage_of(records, lambda r: r.key, lambda r: r.mode, keys=keys)


# -- the estimator -----------------------------------------------------------

def test_good_turing_is_the_singleton_rate():
    assert unseen_mass(["a", "b", "c"]) == pytest.approx(1.0)
    assert unseen_mass(["a", "a", "b", "b"]) == pytest.approx(0.0)
    assert unseen_mass(["a", "a", "b"]) == pytest.approx(1 / 3)


def test_an_agreeing_label_is_a_draw_on_a_species_already_seen():
    """`None` counts in the denominator and never in the numerator.

    Which is what makes the estimate answer the question actually being asked:
    "does the next label show an error mode nobody has seen", not "is the next
    *error* a new kind of error".
    """
    assert unseen_mass(["a", None, None, None]) == pytest.approx(0.25)
    assert unseen_mass([None] * 50) == pytest.approx(0.0)


def test_an_empty_sample_is_nan_not_zero():
    """Zero would say "nothing left to find" where there is no evidence at all."""
    assert unseen_mass([]) != unseen_mass([])


def test_it_tracks_the_true_discovery_rate_on_a_known_population():
    """The validation, synthetically: a population whose true P(new) is known.

    Seven species with the Phase 0 audit's frequencies. Drawn samples of 20, the
    estimate should land near the true fraction of the remainder that is unseen.
    """
    import random

    population = (["a"] * 12 + ["b"] * 9 + ["c"] * 5 + ["d"] * 2
                  + ["e", "f", "g"])
    rng = random.Random(0)
    est, truth = [], []
    for _ in range(400):
        drawn = rng.sample(population, 20)
        rest = list(population)
        for x in drawn:
            rest.remove(x)
        est.append(unseen_mass(drawn))
        truth.append(sum(1 for x in rest if x not in set(drawn)) / len(rest))
    assert sum(est) / len(est) == pytest.approx(sum(truth) / len(truth), abs=0.03)


# -- the bug -----------------------------------------------------------------

def test_a_layer_with_a_hundred_agreeing_labels_is_not_unsampled():
    """The first version scored it 1.0 and sent it the entire budget.

    A hundred labels that found nothing is strong evidence there is little to
    find. An *absence* of labels is no evidence at all. Only the second should
    attract budget, and the denominator is what tells them apart.
    """
    quiet = _cov([_Rec(1.0, 1.0, key="quiet") for _ in range(100)])["quiet"]
    unknown = _cov([], keys=["unknown"])["unknown"]

    assert quiet.labels == 100 and quiet.modes == 0
    assert quiet.unseen == 0.0 and quiet.exhausted
    assert unknown.labels == 0 and unknown.unseen == 1.0
    assert not unknown.exhausted


def test_a_key_never_seen_draws_the_budget_and_a_quiet_one_does_not():
    coverage = _cov([_Rec(1.0, 1.0, key="quiet") for _ in range(100)],
                    keys=["unknown"])
    plan = plan_coverage(weights={"quiet": 0.5, "unknown": 0.5},
                         expected_units=1000, coverage=coverage, target_n=60,
                         min_per_key=5)
    assert plan.target_n["unknown"] == 60 and plan.target_n["quiet"] == 5


# -- allocation --------------------------------------------------------------

def _mixed(key, n_agree, modes):
    rows = [_Rec(1.0, 1.0, key=key) for _ in range(n_agree)]
    rows += [_Rec(1.0, 0.0, key=key, mode=m) for m in modes]
    return rows


def test_the_budget_follows_what_is_still_undiscovered_not_what_is_common():
    """A key wrong constantly in one way, against one wrong rarely in many.

    Neyman would send the labels to the first -- that is where the residual is
    largest. Coverage sends them to the second, because that is where a label
    can still show something new.
    """
    same_bug = _mixed("same", 20, ["formatting"] * 30)
    varied = _mixed("varied", 45, ["a", "b", "c", "d", "e"])
    coverage = _cov(same_bug + varied)

    assert coverage["same"].modes == 1 and coverage["varied"].modes == 5
    assert coverage["same"].unseen < coverage["varied"].unseen

    plan = plan_coverage(weights={"same": 0.5, "varied": 0.5},
                         expected_units=1000, coverage=coverage, target_n=60)
    assert plan.target_n["varied"] > plan.target_n["same"]


def test_floors_are_applied_after_the_allocation():
    coverage = _cov(_mixed("tiny", 40, ["x"] * 10) + _mixed("big", 5, list("abcdef")))
    plan = plan_coverage(weights={"tiny": 0.5, "big": 0.5}, expected_units=1000,
                         coverage=coverage, target_n=20, min_per_key=8)
    assert plan.target_n["tiny"] >= 8, "the floor binds after Neyman-style share"


def test_the_rate_is_the_target_over_the_units_that_key_will_see():
    coverage = _cov(_mixed("a", 10, list("xyz")))
    plan = plan_coverage(weights={"a": 1.0}, expected_units=500,
                         coverage=coverage, target_n=50, min_per_key=1)
    assert plan.rate_for("a") == pytest.approx(plan.target_n["a"] / 500)
    assert plan.rate_for("never-seen") == 0.0, (
        "an unplanned key gets zero, not a quiet default nobody recorded")


def test_a_rate_above_one_is_capped_and_said_out_loud():
    coverage = _cov([], keys=["a"])
    plan = plan_coverage(weights={"a": 1.0}, expected_units=10,
                         coverage=coverage, target_n=100)
    assert plan.rate_for("a") == 1.0
    assert any("come up short" in w for w in plan.warnings)


def test_weights_that_do_not_add_up_are_normalised_and_flagged():
    coverage = _cov(_mixed("a", 5, ["x"]) + _mixed("b", 5, ["y"]))
    plan = plan_coverage(weights={"a": 0.3, "b": 0.3}, expected_units=100,
                         coverage=coverage, target_n=10)
    assert any("do not add up" in w or "not 1" in w for w in plan.warnings)
    assert sum(plan.weights.values()) == pytest.approx(1.0)


def test_no_keys_is_a_plan_that_says_so():
    plan = plan_coverage(weights={}, expected_units=100, coverage={})
    assert plan.rates == {} and "no keys" in plan.warnings[0]


# -- the stopping rule -------------------------------------------------------

def test_when_everything_is_exhausted_the_budget_belongs_elsewhere():
    """The calibration pool never saturates -- its interval keeps narrowing.

    The improvement pool does, and saying so is more useful than quietly
    spending on it forever.
    """
    coverage = _cov(_mixed("a", 100, ["x"] * 40) + _mixed("b", 100, ["y"] * 40))
    plan = plan_coverage(weights={"a": 0.5, "b": 0.5}, expected_units=1000,
                         coverage=coverage, target_n=60)
    assert plan.done
    assert any("flat part of the curve" in w for w in plan.warnings)
    assert exhausted(coverage) == ["a", "b"]


def test_a_pool_still_finding_things_is_not_done():
    coverage = _cov(_mixed("a", 10, list("abcdefghij")))
    plan = plan_coverage(weights={"a": 1.0}, expected_units=1000,
                         coverage=coverage, target_n=60)
    assert not plan.done and plan.unseen_overall > MIN_UNSEEN


def test_every_key_exhausted_falls_back_to_weights_rather_than_dividing_by_zero():
    coverage = {"a": Coverage("a", 100, 1, 0, 0.0),
                "b": Coverage("b", 100, 1, 0, 0.0)}
    plan = plan_coverage(weights={"a": 0.8, "b": 0.2}, expected_units=1000,
                         coverage=coverage, target_n=50, min_per_key=1)
    assert plan.target_n["a"] > plan.target_n["b"]
    assert any("every key is exhausted" in w for w in plan.warnings)


def test_the_overall_estimate_is_pooled_not_averaged():
    """A key with two labels and one with two hundred are not two equal
    opinions about how much is left to find."""
    coverage = {"tiny": Coverage("tiny", 2, 2, 2, 1.0),
                "huge": Coverage("huge", 200, 4, 0, 0.0)}
    pooled = unseen_mass_overall(coverage)
    averaged = (1.0 + 0.0) / 2
    assert pooled == pytest.approx(2 / 202)
    assert pooled < averaged / 10


def test_the_overall_estimate_with_no_labels_is_nan():
    assert math.isnan(unseen_mass_overall({"a": Coverage("a", 0, 0, 0, 1.0)}))


# -- the curve ---------------------------------------------------------------

def test_rarefaction_flattens():
    modes = ["a"] * 12 + ["b"] * 9 + ["c"] * 5 + ["d"] * 2 + ["e", "f", "g"]
    curve = dict(rarefaction(modes, [5, 10, 20, 30], reps=200))
    assert curve[5] < curve[10] < curve[20] < curve[30]
    assert curve[30] - curve[20] < curve[10] - curve[5], (
        "the whole argument for coverage sampling in one inequality")


def test_rarefaction_skips_a_size_larger_than_the_sample():
    assert rarefaction(["a", "b"], [1, 2, 50]) == [(1, 1.0), (2, 2.0)]


def test_rarefaction_is_reproducible():
    modes = list("aabbccddeeff")
    assert rarefaction(modes, [4], seed=7) == rarefaction(modes, [4], seed=7)


# -- unresolved records ------------------------------------------------------

def test_a_pending_record_is_not_a_draw():
    rows = [_Rec(1.0, None, key="a"), _Rec(1.0, 1.0, key="a")]
    assert _cov(rows)["a"].labels == 1


def test_an_unclassified_error_is_not_evidence_that_a_key_is_exhausted():
    """`mode` returning None for a disagreement it cannot name still counts as a
    draw, and still is not a mode -- so it neither invents variety nor hides it."""
    rows = [_Rec(1.0, 0.0, key="a", mode=None) for _ in range(10)]
    got = _cov(rows)["a"]
    assert got.labels == 10 and got.modes == 0 and got.unseen == 0.0
