"""Credit attribution must add up, and must not over-count interactions."""

import pytest

from agentdescent.credit import (
    Edit,
    leave_one_out_credit,
    shapley_credit,
)


def _additive(weights):
    def evaluate(edits):
        return sum(weights.get(edit.edit_id, 0.0) for edit in edits)
    return evaluate


def _edits(*ids):
    return [Edit(edit_id=x) for x in ids]


def test_additive_edits_get_exact_leave_one_out_credit():
    edits = _edits("a", "b", "c")
    report = leave_one_out_credit(edits, _additive({"a": 1.0, "b": 2.0, "c": 3.0}))
    assert report.baseline == 0.0
    assert report.full == 6.0
    assert report.gain == 6.0
    assert report.interactions == pytest.approx(0.0)
    assert [(row.edit_id, row.credit) for row in report.ranking] == [
        ("c", 3.0), ("b", 2.0), ("a", 1.0)]
    assert report.by_id()["b"].share == pytest.approx(2.0 / 6.0)


def test_leave_one_out_over_counts_when_edits_only_work_together():
    def evaluate(edits):
        ids = {edit.edit_id for edit in edits}
        return 2.0 if {"a", "b"} <= ids else 0.0

    report = leave_one_out_credit(_edits("a", "b"), evaluate)
    assert report.gain == 2.0
    assert [row.credit for row in report.ranking] == [2.0, 2.0]
    assert report.interactions == pytest.approx(2.0)
    assert "interaction" in report.detail


def test_shapley_splits_a_synergy_and_stays_efficient():
    def evaluate(edits):
        ids = {edit.edit_id for edit in edits}
        return 2.0 if {"a", "b"} <= ids else 0.0

    report = shapley_credit(_edits("a", "b"), evaluate)
    assert report.method == "shapley"
    assert report.gain == 2.0
    assert sum(row.credit for row in report.credits) == pytest.approx(2.0)
    assert [row.credit for row in report.ranking] == [1.0, 1.0]
    assert report.samples == 2


def test_shapley_gives_zero_to_a_redundant_edit():
    def evaluate(edits):
        return 1.0 if edits else 0.0

    report = shapley_credit(_edits("a", "b"), evaluate)
    assert [row.credit for row in report.ranking] == [0.5, 0.5]
    assert sum(row.credit for row in report.credits) == pytest.approx(1.0)


def test_shapley_sampling_is_deterministic_and_efficient():
    edits = _edits(*"abcdefghij")
    calls = []

    def evaluate(subset):
        calls.append(tuple(e.edit_id for e in subset))
        return float(len(subset))

    one = shapley_credit(edits, evaluate, permutations=16, seed=3, exact_limit=2)
    two = shapley_credit(edits, evaluate, permutations=16, seed=3, exact_limit=2)
    assert one == two
    assert one.samples == 16
    assert sum(row.credit for row in one.credits) == pytest.approx(10.0)
    # A shared subset cache is the point: 16 permutations of 10 edits would be
    # 160 marginal pairs before caching; only distinct subsets are measured.
    assert len(calls) < 512


def test_empty_edit_sets_and_duplicate_ids_are_refused():
    with pytest.raises(ValueError):
        leave_one_out_credit([], _additive({}))
    with pytest.raises(ValueError):
        shapley_credit([], _additive({}))
    with pytest.raises(ValueError):
        shapley_credit(_edits("a", "a"), _additive({"a": 1.0}))


def test_non_finite_scores_are_refused_not_silently_sorted():
    with pytest.raises(ValueError):
        leave_one_out_credit(_edits("a"), lambda edits: float("nan"))
    with pytest.raises(ValueError):
        shapley_credit(_edits("a"), lambda edits: float("inf"))


def test_explicit_baseline_and_full_avoid_re_evaluating_the_endpoints():
    seen = []

    def evaluate(edits):
        seen.append(tuple(e.edit_id for e in edits))
        return 10.0 if edits else 8.0

    report = leave_one_out_credit(_edits("a"), evaluate, baseline=8.0, full=10.0)
    assert report.gain == 2.0
    assert report.by_id()["a"].credit == 2.0
    # Only the "without a" subset needs a call; endpoints were supplied.
    assert seen == [()]


def test_top_negative_and_share_edges():
    report = shapley_credit(_edits("a"), lambda edits: 1.0)
    assert report.top(0) == ()
    assert report.top(5) == report.credits
    with pytest.raises(ValueError):
        report.top(-1)
    empty_gain = leave_one_out_credit(_edits("a"), lambda edits: 1.0,
                                      baseline=1.0, full=1.0)
    assert empty_gain.by_id()["a"].share == 0.0
