"""Searching for a verifier fix instead of guessing one.

`test_the_rule_that_lowers_the_bias_and_raises_the_residual_ranks_last` is the
point. The search is ranked on the residual, so the rule that cuts the mean
error by three quarters while making the verifier worse sorts to the bottom on
its own -- nobody has to remember not to ship it.

`test_a_bundle_that_passes_on_someone_elses_work_is_flagged` is the other half:
a search is a bundle generator, and a bundle launders whatever is in it.
"""

import string
import uuid

import pytest

from agentdescent.audit import AuditRecord, Purpose
from agentdescent.audit.propose import (MAX_COMBINATIONS, Candidate, Rule,
                                        length_rules, search)


def _rec(f, y, *, task=None, output="answer"):
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id=task or uuid.uuid4().hex,
        artifact_signature="a", output=output, verifier_version="v1",
        verifier_score=f, inclusion_prob=1.0, purpose=Purpose.IMPROVEMENT,
        oracle_score=y, resolved_at=1.0)


def _by_id(names):
    """A rule that rejects exactly the records whose task_id is in `names`."""
    return Rule(names if isinstance(names, str) else "-".join(sorted(names)),
                lambda rec, ctx: rec.task_id in set(
                    [names] if isinstance(names, str) else names))


def _population():
    """Ten over-credits a rule could fix, four right answers it could break."""
    recs = [_rec(1.0, 1.0, task=f"ok{i}") for i in range(40)]
    recs += [_rec(1.0, 0.0, task=f"bad{i}") for i in range(10)]
    return recs


# -- the ranking -------------------------------------------------------------

def test_the_rule_that_lowers_the_bias_and_raises_the_residual_ranks_last():
    """A trading rule against a clean one, and the search sorts them itself."""
    recs = _population()
    clean = _by_id([f"bad{i}" for i in range(6)])          # 6 fixed, 0 broken
    clean = Rule("clean", clean.predicate)
    trading = Rule("trading", _by_id(
        [f"bad{i}" for i in range(6, 10)] + [f"ok{i}" for i in range(4)]
    ).predicate)                                            # 4 fixed, 4 broken

    report = search(recs, [clean, trading], max_size=1)
    assert [c.rules for c in report.candidates] == [("clean",), ("trading",)]
    assert report.best.rules == ("clean",) and report.best.helps
    assert report.candidates[-1].rules == ("trading",)


def test_a_rule_that_helps_nothing_still_gets_scored():
    recs = _population()
    useless = Rule("useless", lambda rec, ctx: False)
    report = search(recs, [useless], max_size=1)
    assert report.candidates[0].report.fixed == 0
    assert not report.candidates[0].helps
    assert report.clean == []


def test_combinations_are_built_up_to_max_size():
    rules = [Rule(f"r{i}", lambda rec, ctx: False) for i in range(4)]
    assert search(_population(), rules, max_size=1).n_combinations == 4
    assert search(_population(), rules, max_size=2).n_combinations == 10
    assert search(_population(), rules, max_size=3).n_combinations == 14


def test_a_shorter_combination_wins_a_tie():
    """Two rules that do the same thing are not better than one of them."""
    recs = _population()
    one = _by_id([f"bad{i}" for i in range(6)])
    twin = Rule("twin", one.predicate)
    report = search(recs, [Rule("one", one.predicate), twin], max_size=2)
    assert report.best.rules == ("one",), "the pair does no more and costs more"


# -- laundering --------------------------------------------------------------

def test_a_bundle_that_passes_on_someone_elses_work_is_flagged():
    """The finding, as a check.

    A rule that trades one error for one does not help alone. Paired with a rule
    that works, the pair's residual improves -- so the pair "helps" while still
    carrying a rule that breaks correct judgements.
    """
    recs = _population()
    clean = Rule("clean", _by_id([f"bad{i}" for i in range(8)]).predicate)
    trading = Rule("trading", _by_id(["bad8", "bad9", "ok0", "ok1"]).predicate)

    report = search(recs, [clean, trading], max_size=2)
    pair = next(c for c in report.candidates if len(c.rules) == 2)

    assert not report.candidates[-1].helps, "premise: trading does not help alone"
    assert pair.helps and pair.launders
    assert pair.passengers == ("trading",)
    assert pair not in report.clean
    assert "launders" in report.to_markdown()
    assert "Passing on someone else's work" in report.to_markdown()


def test_a_combination_of_rules_that_all_help_is_not_flagged():
    recs = _population()
    a = Rule("a", _by_id([f"bad{i}" for i in range(5)]).predicate)
    b = Rule("b", _by_id([f"bad{i}" for i in range(5, 10)]).predicate)
    report = search(recs, [a, b], max_size=2)
    assert report.best.rules == ("a", "b")
    assert not report.best.launders and report.best.passengers == ()
    assert report.best in report.clean


# -- the floor ---------------------------------------------------------------

def test_beating_the_floor_is_a_warning_not_a_result():
    """Either a disagreement classified AMBIGUOUS is fixable after all, or the
    combination is fitting the sample. Both are reasons to go and look."""
    recs = _population()
    everything = Rule("everything",
                      _by_id([f"bad{i}" for i in range(10)]).predicate)
    report = search(recs, [everything], max_size=1, floor=0.3)
    assert report.best.sigma < 0.3
    assert report.below_the_floor == [report.best]
    assert "Below the floor" in report.to_markdown()


def test_without_a_floor_nothing_is_below_it():
    report = search(_population(), [Rule("r", lambda rec, ctx: True)])
    assert report.below_the_floor == []
    assert "floor" not in report.to_markdown()


# -- the discipline ----------------------------------------------------------

def test_a_rule_may_only_reject():
    """A rule that could raise a score would be free to buy a lower residual
    with a higher false-negative rate in one move, and the report would show
    only the first."""
    recs = [_rec(0.0, 1.0, task="missed") for _ in range(10)]
    recs += [_rec(1.0, 1.0, task=f"ok{i}") for i in range(20)]
    greedy = Rule("greedy", lambda rec, ctx: True)
    report = search(recs, [greedy], max_size=1)
    assert report.best.report.fixed == 0, (
        "it cannot repair an under-credit, only create more")


def test_too_many_combinations_is_truncated_and_said_out_loud():
    rules = [Rule(f"r{i}", lambda rec, ctx: False) for i in range(30)]
    report = search(_population(), rules, max_size=2, max_combinations=50)
    assert report.n_combinations == 50
    assert any("fitting the noise" in w for w in report.warnings)
    assert "fitting the noise" in report.to_markdown()


def test_nothing_to_search_is_not_an_error():
    assert search([], [Rule("r", lambda rec, ctx: True)]).best is None
    assert search(_population(), []).best is None
    assert "nothing to search" in search([], []).warnings


def test_unresolved_records_are_not_searched_over():
    pending = AuditRecord(
        record_id="p", task_id="t", artifact_signature="a", output="o",
        verifier_version="v1", verifier_score=1.0, inclusion_prob=1.0,
        purpose=Purpose.IMPROVEMENT)
    report = search(_population() + [pending],
                    [Rule("r", lambda rec, ctx: False)])
    assert report.best.report.n_pairs == 50


# -- the library -------------------------------------------------------------

_PUNCT = set(string.punctuation)


def _norm(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    return " ".join(w for w in text.split() if w not in {"a", "an", "the"})


def _fire(rule, output, context):
    return rule(_rec(1.0, 0.0, output=output), context)


def test_every_library_rule_needs_only_a_reference_and_a_normaliser():
    rules = {r.name: r for r in
             length_rules(_norm, lambda c: c["gold"],
                          question_of=lambda c: c["question"])}
    context = {"question": "Which county surrounds Fairfax City?",
               "gold": "Fairfax County"}

    assert _fire(rules["echoes-the-question"], "Fairfax City", context)
    assert not _fire(rules["echoes-the-question"], "Fairfax County", context)
    assert _fire(rules["far-shorter(0.6)"], "Fair", context)
    assert not _fire(rules["far-shorter(0.6)"], "Fairfax Cnty", context)
    assert _fire(rules["far-longer(1.6)"], "the county of Fairfax in Virginia", context)
    assert _fire(rules["empty"], "  !!  ", context)
    assert _fire(rules["shares-no-token-with-reference"], "Arlington", context)
    assert not _fire(rules["shares-no-token-with-reference"], "Fairfax", context)


def test_a_rule_with_no_joined_context_never_fires():
    """An unclassifiable record must not be rejected for being unclassifiable."""
    for rule in length_rules(_norm, lambda c: c["gold"],
                             question_of=lambda c: c["question"]):
        assert not _fire(rule, "anything", None) or rule.name == "empty"


def test_the_question_rule_is_opt_in():
    names = {r.name for r in length_rules(_norm, lambda c: c["gold"])}
    assert "echoes-the-question" not in names
    assert "far-shorter(0.6)" in names


def test_the_ratios_are_dials():
    names = {r.name for r in length_rules(_norm, lambda c: c["gold"],
                                          short=(0.5,), long=(2.0,))}
    assert "far-shorter(0.5)" in names and "far-longer(2.0)" in names
    assert "far-shorter(0.6)" not in names
