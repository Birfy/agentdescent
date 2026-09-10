"""The card that decides whether a new verifier replaces the old one.

`test_the_phase_0_echo_rule_is_refused` is the reason the card exists and the
reason its top row is not the one the plan asked for. One obviously-correct hard
rule cut the measured bias by 74%, and the plan's headline metric would have
shipped it. It made the verifier worse.

The synthetic fixtures below are 11/11 rather than the measured 12/11, so the
arithmetic in the test is checkable by hand. `scripts/audit_diagnose.py` has the
real numbers and re-derives them offline.
"""

import math
import statistics
import uuid

import pytest

from agentdescent.audit import AuditRecord, Purpose
from agentdescent.audit.calibrator import Rectification
from agentdescent.audit.scorecard import (FLIP_ALARM, Cost, Goal, Metric,
                                          rescan, scorecard)


def _rec(f, y, *, sig="a1", prob=1.0, task=None, out="o"):
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id=task or uuid.uuid4().hex,
        artifact_signature=sig, output=out, verifier_version="v1",
        verifier_score=f, inclusion_prob=prob, purpose=Purpose.CALIBRATION,
        oracle_score=y, resolved_at=1.0)


def _rect(**kw):
    base = dict(verifier_version="v2", delta_hat=0.05, delta_se=0.03, theta=0.5,
                theta_ci=(0.44, 0.56), se=0.03, n=177, n_unlab=900,
                gain_factor=1.4, is_stale=False, stale_reason=None,
                resid_sd=0.38)
    base.update(kw)
    return Rectification(**base)


def _before():
    """The real audit: 177 pairs, 31 of them over-credited, nothing under."""
    return [_rec(1.0, 1.0) for _ in range(146)] + [_rec(1.0, 0.0) for _ in range(31)]


def _after():
    """The same set once the echo rule has been applied.

    Eleven over-credits corrected, eleven correct judgements broken -- the
    measured rule does 12 and 11; 11/11 keeps the sums checkable by hand.
    """
    return ([_rec(1.0, 1.0) for _ in range(135)]
            + [_rec(1.0, 0.0) for _ in range(20)]
            + [_rec(0.0, 0.0) for _ in range(11)]
            + [_rec(0.0, 1.0) for _ in range(11)])


# -- the headline ------------------------------------------------------------

def test_the_phase_0_echo_rule_is_refused():
    """The finding, at the level that decides whether a verifier ships.

    Every mean-based number improves. `delta_hat` falls by more than two
    thirds, which the plan calls "the only real target". The card refuses it
    anyway, and says why in both of the rows that can.
    """
    card = scorecard(_rect(delta_hat=0.0508), _after(),
                     previous=_rect(verifier_version="v1", delta_hat=0.1751),
                     previous_records=_before())

    assert card.get("delta_hat").value < 0.4 * card.get("delta_hat").previous, (
        "premise: the metric the plan leads with improves dramatically")
    assert not card.ship
    assert any("residual grew" in b for b in card.blockers)
    assert any("marked down" in b for b in card.blockers)
    assert card.get("sigma").regressed
    assert card.get("disagreement").verdict == "unchanged"


def test_delta_hat_never_returns_a_verdict():
    """It is reported and never scored, because it is the one metric a change
    can improve by breaking things."""
    card = scorecard(_rect(delta_hat=0.0508), _after(),
                     previous=_rect(delta_hat=0.9), previous_records=_before())
    row = card.get("delta_hat")
    assert row.goal is Goal.WATCH
    assert row.verdict == "--" and not row.regressed
    assert "cancel" in row.note


def test_a_fix_that_only_corrects_errors_ships():
    fixed = [_rec(1.0, 1.0) for _ in range(146)] + [_rec(0.0, 0.0) for _ in range(31)]
    card = scorecard(_rect(delta_hat=0.0), fixed,
                     previous=_rect(verifier_version="v1", delta_hat=0.1751),
                     previous_records=_before())
    assert card.ship and card.blockers == []
    assert card.get("sigma").verdict == "better"
    assert card.get("false-negative rate").value == 0.0


def test_the_false_negative_bound_is_named_as_a_policy_not_a_measurement():
    records = ([_rec(1.0, 1.0) for _ in range(90)]
               + [_rec(0.0, 1.0) for _ in range(10)])
    strict = scorecard(_rect(), records, max_false_negative=0.05)
    loose = scorecard(_rect(), records, max_false_negative=0.20)
    assert not strict.ship and loose.ship
    assert "policy" in strict.get("false-negative rate").note


def test_no_false_negative_rate_without_a_right_answer_to_reject():
    got = scorecard(_rect(), [_rec(0.0, 0.0) for _ in range(20)])
    value = got.get("false-negative rate").value
    assert value != value, "nan, not a confident zero"
    assert got.ship, "and a nan must not block"


def test_a_stricter_and_more_accurate_verifier_is_a_trade_not_a_win():
    """Both can be true at once, and the card refuses to call it free."""
    before = [_rec(1.0, 1.0) for _ in range(80)] + [_rec(1.0, 0.0) for _ in range(20)]
    after = ([_rec(1.0, 1.0) for _ in range(78)]
             + [_rec(0.0, 1.0) for _ in range(2)]
             + [_rec(0.0, 0.0) for _ in range(18)]
             + [_rec(1.0, 0.0) for _ in range(2)])
    card = scorecard(_rect(), after, previous=_rect(), previous_records=before,
                     max_false_negative=0.10)
    assert card.get("sigma").verdict == "better"
    assert card.get("false-negative rate").regressed
    assert card.ship
    assert any("not a free improvement" in n for n in card.notes)


# -- the other rows ----------------------------------------------------------

def test_a_falling_gain_factor_is_reported_and_does_not_block():
    """It says the audit got less efficient, not that this change is wrong --
    a different question, answered by a different verifier rather than a veto."""
    card = scorecard(_rect(gain_factor=1.02), _before(),
                     previous=_rect(gain_factor=1.9), previous_records=_before())
    assert card.get("gain_factor").regressed
    assert card.ship
    assert "no usable signal" in card.get("gain_factor").note


def test_a_verifier_that_costs_what_the_oracle_costs_is_refused():
    card = scorecard(_rect(), _before(), cost=Cost(0.9, 1.0))
    assert not card.ship
    assert any("skip the proxy" in b for b in card.blockers)
    assert scorecard(_rect(), _before(), cost=Cost(0.05, 1.0)).ship


def test_cost_without_an_oracle_measurement_is_a_row_not_a_gate():
    card = scorecard(_rect(), _before(), cost=Cost(2.0))
    assert card.ship
    assert card.get("seconds per decision").value == 2.0


def test_a_stale_rectification_labels_every_row_drawn_from_it():
    card = scorecard(_rect(is_stale=True, stale_reason="the judge was rewritten"),
                     _before())
    assert any("stale" in n for n in card.notes)
    assert "the judge was rewritten" in card.to_markdown()


def test_a_baseline_is_not_a_passing_grade():
    card = scorecard(_rect(), _before())
    assert card.ship and card.previous_version is None
    assert any("baseline" in n for n in card.notes)
    assert all(m.previous is None for m in card.metrics)


def test_a_row_that_can_block_is_not_a_row_that_did():
    """A baseline card marked every gate as though it had fired."""
    baseline = scorecard(_rect(), _before()).to_markdown()
    assert "(can block)" in baseline and "**(blocking)**" not in baseline

    refused = scorecard(_rect(), _after(), previous=_rect(),
                        previous_records=_before()).to_markdown()
    assert "**(blocking)**" in refused


def test_a_cost_that_blocks_marks_its_own_row():
    card = scorecard(_rect(), _before(), cost=Cost(0.9, 1.0))
    assert card.get("seconds per decision").triggered
    assert not scorecard(_rect(), _before(),
                         cost=Cost(0.05, 1.0)).get(
                             "seconds per decision").triggered


def test_the_markdown_puts_the_refusal_where_it_will_be_read():
    card = scorecard(_rect(), _after(), previous=_rect(), previous_records=_before())
    text = card.to_markdown()
    assert "## Do not ship" in text
    assert text.index("| sigma |") < text.index("| delta_hat |"), (
        "the row that decides comes before the row that lies")


# -- the ordering row --------------------------------------------------------

def test_the_card_carries_the_only_property_the_gate_uses():
    """A verifier can be badly biased and order perfectly.

    `delta_hat` and `sigma` both say "generous"; neither says "picks the wrong
    winner". The row exists because those are different facts.
    """
    def artifact(sig, f_rate, y_rate, n=20):
        return [_rec(1.0 if i < round(f_rate * n) else 0.0,
                     1.0 if i < round(y_rate * n) else 0.0, sig=sig)
                for i in range(n)]

    biased_but_ordered = artifact("lo", 0.5, 0.2) + artifact("hi", 0.9, 0.6)
    unbiased_but_reversed = artifact("a", 0.50, 0.60) + artifact("b", 0.55, 0.45)

    good = scorecard(_rect(), biased_but_ordered)
    bad = scorecard(_rect(), unbiased_but_reversed)

    assert good.get("ordering agreement").value == 1.0
    assert bad.get("ordering agreement").value == 0.0
    assert any("not the one ground truth" in n for n in bad.notes)
    assert not any("not the one ground truth" in n for n in good.notes)


def test_the_ordering_row_does_not_block():
    """A handful of pairs from one lineage is not something to gate on, and the
    row says so where someone deciding will read it."""
    def artifact(sig, f_rate, y_rate, n=20):
        return [_rec(1.0 if i < round(f_rate * n) else 0.0,
                     1.0 if i < round(y_rate * n) else 0.0, sig=sig)
                for i in range(n)]

    # the fixture also carries a few false negatives; raise that bound so the
    # only thing this test can be reading is the ordering row
    card = scorecard(_rect(), artifact("a", 0.50, 0.60) + artifact("b", 0.55, 0.45),
                     max_false_negative=0.5)
    row = card.get("ordering agreement")
    assert row.value == 0.0, "premise: every pair is ordered backwards"
    assert not row.blocking and not row.triggered
    assert "Not blocking" in row.note
    assert card.ship, "it is a flag, not a gate"


def test_a_single_artifact_has_no_ordering_to_report():
    card = scorecard(_rect(), _before())
    assert card.get("ordering agreement") is None
    assert card.rank is None


# -- the rescan --------------------------------------------------------------

def test_a_uniform_rescale_has_no_spread():
    recs = [_rec(1.0, 1.0), _rec(0.0, 0.0), _rec(1.0, 0.0)]
    got = rescan(recs, lambda r, ctx: r.verifier_score - 0.1)
    assert got.mean_shift == pytest.approx(-0.1)
    assert got.sigma_shift == pytest.approx(0.0)
    assert got.agreement == 0.0


def test_a_verifier_that_changed_its_mind_both_ways_looks_unchanged_on_average():
    """Why `sigma_shift` is on the report next to `mean_shift`.

    Half the outputs go up by 0.5 and half go down by 0.5. Every average says
    nothing happened; the run's history was rescored from end to end.
    """
    recs = [_rec(0.5, 1.0, task=f"t{i}") for i in range(20)]
    up = {r.record_id for r in recs[:10]}

    got = rescan(recs, lambda r, ctx: r.verifier_score
                 + (0.5 if r.record_id in up else -0.5))
    assert got.mean_shift == pytest.approx(0.0, abs=1e-12)
    assert got.sigma_shift == pytest.approx(0.5, rel=0.1)
    assert got.agreement == 0.0


def test_the_rescan_weights_by_inclusion_probability():
    """The audited units were deliberately not representative.

    One stratum sampled at 100% and another at 10%: an unweighted mean of the
    change is the first stratum's answer wearing the whole run's name.
    """
    recs = ([_rec(1.0, 1.0, prob=1.0) for _ in range(10)]
            + [_rec(1.0, 1.0, prob=0.1) for _ in range(10)])
    dense = {r.record_id for r in recs[:10]}
    got = rescan(recs, lambda r, ctx: 0.0 if r.record_id in dense else 1.0)

    unweighted = statistics.fmean(
        [(0.0 if r.record_id in dense else 1.0) - 1.0 for r in recs])
    assert unweighted == pytest.approx(-0.5)
    assert got.mean_shift == pytest.approx(-1 / 11, abs=1e-9), (
        "the rare stratum stands for ten times as many units, so a change "
        "confined to the dense one moves the population far less")


def test_a_reversed_pair_is_what_a_flipped_accept_looks_like():
    recs = ([_rec(0.4, 1.0, sig="base") for _ in range(5)]
            + [_rec(0.6, 1.0, sig="cand") for _ in range(5)])
    # the new verifier prefers the shorter signature's outputs
    got = rescan(recs, lambda r, ctx: 0.9 if r.artifact_signature == "base" else 0.1,
                 pairs=[("base", "cand")])
    assert got.n_pairs == 1 and len(got.flipped) == 1
    base, cand, was, now = got.flipped[0]
    assert (base, cand) == ("base", "cand")
    assert was > 0 > now
    assert got.flip_rate == 1.0 and got.alarming


def test_an_unchanged_ordering_is_not_a_flip():
    recs = ([_rec(0.4, 1.0, sig="base") for _ in range(5)]
            + [_rec(0.6, 1.0, sig="cand") for _ in range(5)])
    got = rescan(recs, lambda r, ctx: r.verifier_score * 2.0,
                 pairs=[("base", "cand")])
    assert got.flipped == [] and got.flip_rate == 0.0 and not got.alarming


def test_the_alarm_is_a_rate_not_a_count():
    """Twenty artifacts each a hair better than the last; one pair reverses."""
    recs = []
    for i in range(20):
        recs += [_rec(0.4 + 0.01 * i, 1.0, sig=f"a{i}") for _ in range(3)]
    pairs = [(f"a{i}", f"a{i+1}") for i in range(19)]
    new_scores = {"a0": 0.9, "a1": 0.1}
    got = rescan(recs, lambda r, ctx: new_scores.get(r.artifact_signature,
                                                     r.verifier_score),
                 pairs=pairs)
    assert got.n_pairs == 19 and len(got.flipped) == 1
    assert 0 < got.flip_rate <= FLIP_ALARM and not got.alarming


def test_a_pair_that_was_already_tied_has_no_ordering_to_reverse():
    recs = ([_rec(0.5, 1.0, sig="base") for _ in range(3)]
            + [_rec(0.5, 1.0, sig="cand") for _ in range(3)])
    got = rescan(recs, lambda r, ctx: 0.9 if r.artifact_signature == "base"
                 else 0.1, pairs=[("base", "cand")])
    assert got.n_pairs == 1 and got.flipped == []
    assert got.by_artifact["base"]["shift"] == pytest.approx(0.4)


def test_pairs_naming_signatures_the_sample_never_saw_are_skipped():
    recs = [_rec(0.5, 1.0, sig="a")]
    got = rescan(recs, lambda r, ctx: 0.5, pairs=[("a", "ghost"), ("a", "a")])
    assert got.n_pairs == 1, "only the pair both of whose sides were audited"


def test_the_rescan_measures_accuracy_against_the_labels_it_has():
    recs = [_rec(1.0, 0.0) for _ in range(10)] + [_rec(1.0, 1.0) for _ in range(10)]
    got = rescan(recs, lambda r, ctx: r.oracle_score)
    assert got.sigma_before > 0.5
    assert got.sigma_after == pytest.approx(0.0)


def test_an_empty_rescan_reports_nothing_rather_than_dividing():
    got = rescan([], lambda r, ctx: 1.0)
    assert got.n == 0 and got.agreement != got.agreement
    assert got.flip_rate != got.flip_rate


def test_the_rescan_says_what_it_did_not_do():
    """The plan calls this a Ledger replay and says it is nearly free.

    The Ledger stores artifact states; the outputs a verifier scores were never
    kept, so re-deciding a past merge needs the rollouts back. What is free is
    re-scoring the outputs the audit kept -- a probability sample, with an `n`.
    """
    from agentdescent.audit.scorecard import RescanReport

    doc = RescanReport.__doc__
    assert "Not a re-decide" in doc and "probability sample" in doc


def test_an_alarming_rescan_blocks_the_card():
    recs = ([_rec(0.4, 1.0, sig="base") for _ in range(5)]
            + [_rec(0.6, 1.0, sig="cand") for _ in range(5)])
    report = rescan(recs, lambda r, ctx: 0.9 if r.artifact_signature == "base"
                    else 0.1, pairs=[("base", "cand")])
    card = scorecard(_rect(), _before(), rescan_report=report)
    assert not card.ship
    assert any("reverse order" in b for b in card.blockers)
    assert "Rescan" in card.to_markdown()


# -- the row type ------------------------------------------------------------

def test_a_row_with_nothing_to_compare_to_has_no_opinion():
    row = Metric("x", 1.0, Goal.LOWER)
    assert row.verdict == "--" and not row.regressed
    assert math.isnan(row.change)


def test_higher_and_lower_read_the_same_change_opposite_ways():
    up = Metric("x", 2.0, Goal.HIGHER, previous=1.0)
    down = Metric("x", 2.0, Goal.LOWER, previous=1.0)
    assert up.verdict == "better" and down.verdict == "worse"
    assert down.regressed and not up.regressed
