"""A fix has to be measured on the answers it was not aimed at.

The test that matters is `test_a_fix_that_only_trades_directions_does_not_help`.
It reproduces, synthetically, what a real audit produced: two obviously-correct
hard rules that removed eleven verifier errors, created eleven new ones, cut the
mean bias by 71%, and left the residual *larger* than before.

Every number that summarises a fix by its mean said that was a success. Only the
residual said otherwise, which is why `FixReport.helps` reads the residual.
"""

import math
import statistics
import string
import uuid

import pytest

from agentdescent.audit import AuditRecord, Purpose
from agentdescent.audit.diagnose import (Direction, DisagreementReport, Kind,
                                         classify_disagreements, evaluate_fix,
                                         reference_classifier, residual_stats)


def _rec(f, y, *, task=None, output="out"):
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id=task or uuid.uuid4().hex,
        artifact_signature="s", output=output, verifier_version="v1",
        verifier_score=f, inclusion_prob=1.0, purpose=Purpose.CALIBRATION,
        oracle_score=y, resolved_at=1.0)


def _population(n_agree=146, n_over=31, n_under=0):
    """The shape of the real audit: 177 pairs, 31 of them over-credited."""
    recs = [_rec(1.0, 1.0, task=f"ok{i}") for i in range(n_agree)]
    recs += [_rec(1.0, 0.0, task=f"over{i}") for i in range(n_over)]
    recs += [_rec(0.0, 1.0, task=f"under{i}") for i in range(n_under)]
    return recs


# -- the statistic -----------------------------------------------------------

def test_residual_stats_reports_the_pair_that_matters():
    recs = _population(n_agree=146, n_over=31)
    got = residual_stats(recs)
    assert got["n"] == 177
    assert got["delta"] == pytest.approx(31 / 177, abs=1e-9)
    assert got["disagree"] == pytest.approx(31 / 177, abs=1e-9)
    assert got["sigma"] > 0.37


def test_unresolved_records_are_not_counted_as_agreement():
    recs = [_rec(1.0, 1.0), AuditRecord(
        record_id="p", task_id="t", artifact_signature="s", output="o",
        verifier_version="v1", verifier_score=1.0, inclusion_prob=1.0,
        purpose=Purpose.CALIBRATION)]
    assert residual_stats(recs)["n"] == 1


def test_no_resolved_records_is_nan_rather_than_zero():
    got = residual_stats([])
    assert got["n"] == 0 and got["delta"] != got["delta"]


# -- direction ---------------------------------------------------------------

def test_direction_is_recorded_because_a_fix_can_trade_one_for_the_other():
    report = classify_disagreements(_population(n_over=6, n_under=4))
    assert report.by_direction[Direction.OVER] == 6
    assert report.by_direction[Direction.UNDER] == 4
    assert report.n_disagree == 10


def test_without_a_classifier_everything_is_unclassified_not_guessed():
    """A classifier that cannot see the reference cannot tell formatting from
    judgement, and a guess would put a number nobody measured on the report that
    decides what to fix."""
    report = classify_disagreements(_population(n_over=5))
    assert report.by_kind == {Kind.UNCLASSIFIED: 5}
    # still worth running: the counts and the direction split are the two
    # numbers that catch a fix trading OVER for UNDER
    assert report.by_direction[Direction.OVER] == 5


# -- classifying -------------------------------------------------------------

_ART = {"a", "an", "the"}


def _norm(t):
    t = t.lower()
    t = "".join(c for c in t if c not in set(string.punctuation))
    return " ".join(w for w in t.split() if w not in _ART)


def test_formatting_is_what_normalisation_alone_would_fix():
    recs = [_rec(1.0, 0.0, task="t1", output="The Answer!"),
            _rec(1.0, 0.0, task="t2", output="something else")]
    context = {"t1": "answer", "t2": "answer"}
    report = classify_disagreements(
        recs, reference_classifier(_norm, lambda ctx: ctx), context)
    kinds = {d.record.task_id: d.kind for d in report.items}
    assert kinds["t1"] is Kind.FORMATTING
    assert kinds["t2"] is Kind.UNCLASSIFIED


def test_the_two_domain_judgements_are_the_callers():
    """Whether a shorter answer is the oracle being pedantic or the answer being
    incomplete is a fact about the domain, not something a library can decide."""
    recs = [_rec(1.0, 0.0, task="echo", output="what is the capital"),
            _rec(1.0, 0.0, task="short", output="Childers")]
    context = {"echo": ("what is the capital", "Paris"),
               "short": ("who was he", "Robert Erskine Childers DSC")}

    classify = reference_classifier(
        _norm, lambda ctx: ctx[1],
        spec_gap_when=lambda out, ref, ctx: _norm(out) in _norm(ctx[0]),
        ambiguous_when=lambda out, ref, ctx: _norm(out) in _norm(ref))
    report = classify_disagreements(recs, classify, context)
    kinds = {d.record.task_id: d.kind for d in report.items}
    assert kinds["echo"] is Kind.SPEC_GAP
    assert kinds["short"] is Kind.AMBIGUOUS


# -- the floor ---------------------------------------------------------------

def test_the_ambiguous_bucket_is_a_floor_not_an_opportunity():
    """`floor_sigma` is what remains once everything fixable is fixed.

    A plan that targets a residual below it is not a plan to improve the
    verifier; it is a plan to redefine what counts as correct.
    """
    recs = _population(n_agree=140, n_over=20)
    context = {r.task_id: r.task_id for r in recs}
    # half the over-credits are the oracle being pedantic
    classify = reference_classifier(
        _norm, lambda ctx: "x",
        ambiguous_when=lambda out, ref, ctx: str(ctx).endswith(tuple("02468")))
    report = classify_disagreements(recs, classify, context)

    assert report.by_kind.get(Kind.AMBIGUOUS, 0) > 0
    assert 0.0 < report.floor_sigma < report.sigma, (
        "the floor must be below today's residual and above zero")
    assert report.sigma_without[Kind.AMBIGUOUS] < report.sigma


def test_a_verifier_whose_errors_are_all_ambiguous_has_no_headroom():
    recs = _population(n_agree=100, n_over=10)
    classify = reference_classifier(_norm, lambda ctx: "x",
                                    ambiguous_when=lambda o, r, c: True)
    report = classify_disagreements(recs, classify,
                                    {r.task_id: 1 for r in recs})
    assert report.floor_sigma == pytest.approx(report.sigma), (
        "nothing here is fixable, so the floor is where we already are")


def test_the_markdown_names_the_floor():
    report = classify_disagreements(_population(n_over=5))
    assert "Floor" in report.to_markdown()


# -- evaluating a fix --------------------------------------------------------

def test_a_fix_that_only_trades_directions_does_not_help():
    """The finding this module exists for, reproduced.

    On a real 177-pair audit, two hard rules fixed 11 over-credits and broke 11
    correct judgements. Every mean-based number improved -- `delta` fell 71% --
    and the residual, which is what the acceptance gate's variance is built from,
    went up.
    """
    recs = _population(n_agree=146, n_over=31)
    over = [r for r in recs if r.residual > 0][:11]
    right = [r for r in recs if r.residual == 0.0][:11]
    targets = {r.record_id for r in over} | {r.record_id for r in right}

    def rule(record, ctx):
        return 0.0 if record.record_id in targets else record.verifier_score

    got = evaluate_fix(recs, rule)

    assert got.fixed == 11 and got.broken == 11
    assert abs(got.delta_after) < abs(got.delta_before) * 0.4, (
        "premise: the mean improves dramatically")
    assert got.sigma_after > got.sigma_before, (
        "and the residual -- the thing that matters -- gets worse")
    assert got.disagree_after == pytest.approx(got.disagree_before)
    assert not got.helps
    assert "does not help" in got.to_markdown()


def test_a_fix_that_only_corrects_errors_helps():
    recs = _population(n_agree=146, n_over=31)
    targets = {r.record_id for r in recs if r.residual > 0}

    def rule(record, ctx):
        return 0.0 if record.record_id in targets else record.verifier_score

    got = evaluate_fix(recs, rule)
    assert got.fixed == 31 and got.broken == 0
    assert got.sigma_after < got.sigma_before and got.helps
    assert got.false_negative_rate == 0.0


def test_the_false_negative_rate_is_over_the_answers_it_was_not_aimed_at():
    recs = _population(n_agree=100, n_over=10)
    right = [r for r in recs if r.residual == 0.0][:25]
    targets = {r.record_id for r in right}

    def rule(record, ctx):
        return 0.0 if record.record_id in targets else record.verifier_score

    got = evaluate_fix(recs, rule)
    assert got.broken == 25 and got.fixed == 0
    assert got.false_negative_rate == pytest.approx(25 / 100)
    assert not got.helps


def test_a_fix_evaluated_only_on_its_targets_would_have_looked_like_a_win():
    """Why `evaluate_fix` takes the whole set and not the disagreements.

    Restricted to the pairs it was aimed at, the trading fix removes eleven
    errors and breaks nothing: the disagreement rate and the bias both fall by
    35% and it reads as a clean win. The same fix on the whole set is the test
    above -- eleven fresh errors, and a larger residual.

    `helps` survives even this misuse, and the reason is worth keeping. On the
    restricted view every residual is `+1`, so `sigma_before` is **zero** -- a
    constant has no spread -- and any change at all raises it. Reading the
    residual rather than a mean makes the metric refuse a comparison that was
    never valid, instead of endorsing it.
    """
    recs = _population(n_agree=146, n_over=31)
    over = [r for r in recs if r.residual > 0][:11]
    right = [r for r in recs if r.residual == 0.0][:11]
    targets = {r.record_id for r in over} | {r.record_id for r in right}

    def rule(record, ctx):
        return 0.0 if record.record_id in targets else record.verifier_score

    disagreements_only = [r for r in recs if r.residual != 0.0]
    misleading = evaluate_fix(disagreements_only, rule)
    honest = evaluate_fix(recs, rule)

    assert misleading.broken == 0, "premise: aimed only at its targets, it breaks nothing"
    assert honest.broken == 11

    # the two numbers a person reaches for, and both of them lie here
    assert misleading.disagree_after / misleading.disagree_before == pytest.approx(
        20 / 31, rel=0.01)
    assert misleading.delta_after / misleading.delta_before == pytest.approx(
        20 / 31, rel=0.01)

    # while on the whole set the residual is worse and the verdict is honest
    assert honest.sigma_after > honest.sigma_before
    assert not honest.helps
    # and `helps` refuses the restricted view too, because a constant residual
    # has no spread to improve on
    assert not misleading.helps


def test_a_no_op_fix_changes_nothing():
    recs = _population(n_over=7)
    got = evaluate_fix(recs, lambda record, ctx: record.verifier_score)
    assert got.fixed == got.broken == 0
    assert got.sigma_after == got.sigma_before
    assert got.unchanged == got.n_pairs


def test_an_empty_set_reports_nothing_rather_than_dividing():
    got = evaluate_fix([], lambda record, ctx: 1.0)
    assert got.n_pairs == 0 and got.sigma_after != got.sigma_after
