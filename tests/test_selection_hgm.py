"""CladeThompson: HGM's clade-metaproductivity Thompson sampling (#276).

Every test runs on a hand-built candidate tree, as the other selection tests
do; lineage under ``evolve()`` comes from the population archive (#275).
"""

from collections import Counter

import pytest

from agentdescent.selection import (
    Candidate, CladeThompson, SelectionContext, clade_counts,
)


def _c(version, score=None, per_task=None, parent=None):
    return Candidate(artifact_id="a", version=version, score=score,
                     per_task=per_task or {}, parent=parent)


def _ctx(*cands, round=0, budget=1.0):
    return SelectionContext(head=cands[0], candidates=cands, round=round,
                            budget_remaining=budget)


def _mismatch_tree():
    # 1: low-scoring parent whose children all succeed.
    # 5: high-scoring node whose children all fail.
    return (
        _c(1, 0.2),
        _c(2, 0.9, parent=1), _c(3, 0.9, parent=1), _c(4, 0.9, parent=1),
        _c(5, 0.7),
        _c(6, 0.05, parent=5), _c(7, 0.05, parent=5), _c(8, 0.05, parent=5),
    )


def test_metaproductive_parent_beats_high_scoring_node_with_failing_clade():
    tree = _mismatch_tree()
    picks = Counter()
    for r in range(300):
        picks.update(c.version for c in CladeThompson(seed=3).select(
            _ctx(*tree, round=r), 1))
    assert picks[1] > 2 * picks[5], picks


def test_single_candidate_returns_head():
    head = _c(1, 0.5)
    assert CladeThompson().select(_ctx(head), 3) == [head] * 3


def test_no_eligible_node_returns_head():
    tree = (_c(1, 0.0), _c(2, None, parent=1), _c(3, 0.0, parent=1))
    assert CladeThompson().select(_ctx(*tree), 2) == [tree[0]] * 2


def test_same_seed_same_choices_and_rng_owned_stream():
    import random

    tree = _mismatch_tree()
    a = [c.version for r in range(20)
         for c in CladeThompson(seed=7).select(_ctx(*tree, round=r), 2)]
    b = [c.version for r in range(20)
         for c in CladeThompson(seed=7).select(_ctx(*tree, round=r), 2)]
    assert a == b
    one = CladeThompson(rng=random.Random(11)).select(_ctx(*tree), 4)
    two = CladeThompson(rng=random.Random(11)).select(_ctx(*tree), 4)
    assert [c.version for c in one] == [c.version for c in two]


def test_cycles_and_dangling_parents_neither_loop_nor_raise():
    tree = (_c(1, 0.5, parent=3), _c(2, 0.6, parent=1), _c(3, 0.4, parent=2),
            _c(4, 0.8, parent=99))
    counts = clade_counts(tree)
    # Each node in the 3-cycle reaches every member exactly once.
    total_s = 0.5 + 0.6 + 0.4
    for s, f in counts[:3]:
        assert s == pytest.approx(total_s)
        assert f == pytest.approx(3 - total_s)
    assert counts[3] == pytest.approx((0.8, 0.2))
    assert len(CladeThompson().select(_ctx(*tree), 5)) == 5


def test_clade_counts_match_hand_computation_and_do_not_accumulate():
    tree = (
        _c(1, 0.4),
        _c(2, per_task={"t1": 1.0, "t2": 0.0, "t3": 1.0}, parent=1),
        _c(3, 0.25, parent=2),
        _c(4, None, parent=1),          # unscored: contributes nothing
        _c(5, 1.5, parent=1),           # clipped to 1.0
    )
    expected = [
        (0.4 + 2.0 + 0.25 + 1.0, 0.6 + 1.0 + 0.75 + 0.0),
        (2.0 + 0.25, 1.0 + 0.75),
        (0.25, 0.75),
        (0.0, 0.0),
        (1.0, 0.0),
    ]
    assert clade_counts(tree) == pytest.approx(expected)
    policy = CladeThompson(seed=1)
    for r in range(5):
        policy.select(_ctx(*tree, round=r), 3)
    # Upstream main grows a node's measures on every expand(); the paper's
    # formula -- and this port -- does not.
    assert clade_counts(tree) == pytest.approx(expected)


def test_cool_down_concentrates_on_highest_cmp_as_budget_runs_out():
    tree = (_c(1, 0.55), _c(2, 0.6, parent=1), _c(3, 0.6, parent=1),
            _c(4, 0.45))
    # Large tau concentrates each Beta on its mean, (1 + S) / (2 + S + F):
    # node 1, whose clade carries the most evidence, not its higher-scoring
    # children with one observation each.
    counts = clade_counts(tree)
    means = [(1 + s) / (2 + s + f) for s, f in counts]
    top = max(range(len(tree)), key=lambda i: means[i])
    assert top == 0 and sorted(means)[-1] > sorted(means)[-2]

    def share(budget):
        picks = Counter()
        policy = CladeThompson(cool_down_beta=2.0, max_tau=1e4, seed=5)
        for r in range(200):
            picks.update(c.version for c in policy.select(
                _ctx(*tree, round=r, budget=budget), 1))
        return picks[tree[top].version] / 200

    assert share(1e-6) >= 0.95
    assert share(1e-6) > share(1.0)


def test_tau_schedule():
    assert CladeThompson().tau(0.01) == 1.0
    p = CladeThompson(cool_down_beta=1.0, max_tau=50)
    assert p.tau(1.0) == 1.0
    assert p.tau(0.5) == pytest.approx(2.0)
    assert p.tau(0.0) == 50
    assert p.tau(1e-9) == 50
    with pytest.raises(ValueError):
        CladeThompson(cool_down_beta=-1)
