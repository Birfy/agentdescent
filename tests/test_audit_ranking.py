"""Whether the verifier can order things, which is the only thing the gate does.

`test_a_verifier_can_be_badly_biased_and_order_perfectly` and
`test_a_verifier_can_be_nearly_unbiased_and_order_backwards` are the pair that
justifies the module: neither `delta_hat` nor `resid_sd` separates those two
cases, and the acceptance gate only ever asks the question they differ on.
"""

import math
import statistics
import uuid

import pytest

from agentdescent.audit import AuditRecord, Purpose
from agentdescent.audit.diagnose import residual_stats
from agentdescent.audit.ranking import (Flip, RankReport, kendall_tau_b,
                                        rank_agreement)


def _rec(f, y, *, sig="a", prob=1.0):
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id=uuid.uuid4().hex,
        artifact_signature=sig, output="o", verifier_version="v1",
        verifier_score=f, inclusion_prob=prob, purpose=Purpose.CALIBRATION,
        oracle_score=y, resolved_at=1.0)


def _artifact(sig, f_rate, y_rate, n=20, prob=1.0):
    """`n` units whose verifier and oracle means are exactly the rates given."""
    rows = []
    for i in range(n):
        rows.append(_rec(1.0 if i < round(f_rate * n) else 0.0,
                         1.0 if i < round(y_rate * n) else 0.0,
                         sig=sig, prob=prob))
    return rows


# -- the pair that justifies the module --------------------------------------

def test_a_verifier_can_be_badly_biased_and_order_perfectly():
    """Every score inflated by the same amount. `delta_hat` is large; the gate
    is untouched, because a constant cancels out of every comparison."""
    recs = _artifact("lo", 0.5, 0.2) + _artifact("hi", 0.9, 0.6)
    report = rank_agreement(recs)

    assert residual_stats(recs)["delta"] == pytest.approx(0.3, abs=0.01), (
        "premise: a large mean error")
    assert report.agreement == 1.0 and not report.flips
    assert report.picks_the_same_winner


def test_a_verifier_can_be_nearly_unbiased_and_order_backwards():
    """The opposite corner, and the one no other statistic here reports."""
    recs = _artifact("a", 0.50, 0.60) + _artifact("b", 0.55, 0.45)
    report = rank_agreement(recs)

    assert abs(residual_stats(recs)["delta"]) < 0.03, (
        "premise: the bias is almost nothing")
    assert report.agreement == 0.0 and len(report.flips) == 1
    assert not report.picks_the_same_winner
    assert report.best_by_verifier == "b" and report.best_by_oracle == "a"


# -- the tau caveat, which is the trap ---------------------------------------

def test_a_one_directional_verifier_cannot_have_a_discordant_pair():
    """So its Kendall tau looks excellent and means nothing.

    Two units are discordant only when the verifier prefers one and the truth
    prefers the other. When every error runs the same way, no such pair exists
    -- which is a fact about the *direction* of the bias, not about ordering.
    """
    recs = ([_rec(1.0, 1.0) for _ in range(40)]
            + [_rec(1.0, 0.0) for _ in range(10)]
            + [_rec(0.0, 0.0) for _ in range(30)])
    report = rank_agreement(recs)
    assert report.discordant == 0 and report.tau_b > 0.6
    assert report.one_directional
    assert "says nothing about ordering" in report.to_markdown()


def test_a_verifier_that_says_yes_to_everything_scores_the_same():
    """The reductio, kept as a test because the number is genuinely tempting."""
    real = rank_agreement([_rec(1.0, 1.0) for _ in range(40)]
                          + [_rec(1.0, 0.0) for _ in range(10)]
                          + [_rec(0.0, 0.0) for _ in range(30)])
    useless = rank_agreement([_rec(1.0, 1.0) for _ in range(40)]
                             + [_rec(1.0, 0.0) for _ in range(40)])
    assert useless.discordant == real.discordant == 0
    assert useless.one_directional


def test_errors_in_both_directions_can_be_discordant():
    recs = [_rec(1.0, 0.0), _rec(0.0, 1.0)]
    report = rank_agreement(recs)
    assert report.discordant == 1 and not report.one_directional


def test_a_verifier_that_never_errs_is_not_one_directional():
    report = rank_agreement([_rec(1.0, 1.0), _rec(0.0, 0.0)])
    assert not report.one_directional, (
        "there are no errors to have a direction"
    )


# -- tau itself --------------------------------------------------------------

def test_tau_b_is_one_on_a_perfect_ordering():
    tau, conc, disc = kendall_tau_b([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert tau == pytest.approx(1.0) and conc == 3 and disc == 0


def test_tau_b_is_minus_one_on_a_reversal():
    tau, _, disc = kendall_tau_b([1.0, 2.0, 3.0], [30.0, 20.0, 10.0])
    assert tau == pytest.approx(-1.0) and disc == 3


def test_tau_b_corrects_for_ties_rather_than_being_dragged_down_by_them():
    """Uncorrected, binary data reads far lower than the agreement it describes."""
    x = [1.0, 1.0, 0.0, 0.0]
    y = [1.0, 1.0, 0.0, 0.0]
    tau, conc, disc = kendall_tau_b(x, y)
    assert disc == 0
    assert tau == pytest.approx(1.0), (
        "perfect agreement, four ties, and tau-b still says 1.0")
    plain = (conc - disc) / (len(x) * (len(x) - 1) / 2)
    assert plain < 0.7, "which the uncorrected form does not"


def test_everything_tied_is_nan_rather_than_a_confident_zero():
    tau, _, _ = kendall_tau_b([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert tau != tau


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        kendall_tau_b([1.0, 2.0], [1.0])


# -- the gap filter ----------------------------------------------------------

def test_a_reversal_on_a_gap_the_gate_would_refuse_costs_nothing():
    """Which is why agreement is reported against the size of the gap.

    Two reversals: one on a 2-point apparent gain the gate would never act on,
    one on a 20-point gain it certainly would.
    """
    recs = (_artifact("base", 0.50, 0.55)
            + _artifact("tiny", 0.55, 0.50)      # gap +0.05, reversed
            + _artifact("big", 0.75, 0.45))      # gap +0.25, reversed
    report = rank_agreement(recs)
    assert len(report.flips) >= 2

    agree_all, flip_all = report.above(0.0)
    agree_big, flip_big = report.above(0.20)
    assert flip_all > flip_big, "the small reversal drops out"
    assert flip_big >= 1, "and the one that would have committed does not"


def test_the_filter_reads_the_gap_the_gate_can_see():
    """The verifier's gap, not the truth's -- the gate has only the first."""
    recs = _artifact("a", 0.50, 0.90, n=50) + _artifact("b", 0.52, 0.10, n=50)
    report = rank_agreement(recs)
    assert report.above(0.10) == (0, 0), (
        "a 2-point verifier gap is below the filter however large the truth's "
        "gap turned out to be")
    assert report.above(0.0)[1] == 1


# -- weighting and shape -----------------------------------------------------

def test_artifact_means_are_inclusion_weighted():
    """The audited units were deliberately not representative."""
    dense = _artifact("a", 1.0, 1.0, n=10, prob=1.0)
    rare = _artifact("a", 0.0, 0.0, n=10, prob=0.1)
    report = rank_agreement(dense + rare)
    assert report.by_artifact["a"]["verifier"] == pytest.approx(1 / 11, abs=1e-9)


def test_explicit_pairs_are_the_comparisons_the_run_actually_made():
    recs = _artifact("a", 0.5, 0.6) + _artifact("b", 0.6, 0.5) + _artifact("c", 0.7, 0.7)
    every = rank_agreement(recs)
    named = rank_agreement(recs, pairs=[("a", "b")])
    assert every.n_pairs == 3 and named.n_pairs == 1
    assert len(named.flips) == 1


def test_a_pair_naming_an_artifact_that_was_never_audited_is_skipped():
    recs = _artifact("a", 0.5, 0.5)
    assert rank_agreement(recs, pairs=[("a", "ghost")]).n_pairs == 0


def test_a_thinly_audited_artifact_can_be_dropped():
    """One unit gives a rate of 0 or 1 and would reverse orderings on nothing."""
    recs = _artifact("solid", 0.5, 0.5, n=20) + [_rec(1.0, 0.0, sig="thin")]
    assert rank_agreement(recs).n_artifacts == 2
    assert rank_agreement(recs, min_units=5).n_artifacts == 1


def test_a_tie_is_not_an_agreement_and_not_a_flip():
    recs = _artifact("a", 0.5, 0.5) + _artifact("b", 0.5, 0.8)
    report = rank_agreement(recs)
    assert report.ties == 1 and report.agree == 0 and not report.flips
    assert report.agreement != report.agreement, "nothing decided, so nan"


def test_no_resolved_records_is_nan_rather_than_perfect_agreement():
    report = rank_agreement([])
    assert report.n_units == 0 and report.tau_b != report.tau_b
    assert report.best_by_verifier is None


# -- reporting ---------------------------------------------------------------

def test_the_markdown_names_the_winner_when_the_two_disagree():
    recs = _artifact("a", 0.50, 0.60) + _artifact("b", 0.55, 0.45)
    text = rank_agreement(recs).to_markdown()
    assert "**not** the truth's" in text
    assert "Reversed" in text


def test_the_markdown_refuses_to_offer_an_interval():
    """Five artifacts from one run are a lineage; a binomial interval on ten
    pairs would be wrong about the dependence on top of being useless."""
    text = rank_agreement(_artifact("a", 0.5, 0.5) + _artifact("b", 0.6, 0.6)).to_markdown()
    assert "not a sample to put an interval around" in text


def test_a_flip_prints_both_gaps():
    flip = Flip("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", 0.12, -0.06, 32, 49)
    assert "+0.120" in str(flip) and "-0.060" in str(flip)
