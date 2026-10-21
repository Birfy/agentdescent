"""Evolved decision code that beats the shipped implementation on quality-per-cost.

The reward includes cost, so the search finds a rule that is as good *and
cheaper* than the hand-written seed; the report measures the evolved rule
against the seed on both axes. The fixtures pin the surfaces that make this
honest: source cost is a deterministic count, the cost-adjusted reward prefers
cheaper-at-equal-quality, the grid problem scores against the seed's own
ordering, and the end-to-end run shows a cheaper rule with equal quality
winning a positive cost-adjusted gain.
"""

import pytest

from agentdescent import (
    MetaOutcome,
    cost_adjusted_reward,
    evolve_fast_rule,
    priority_grid_problem,
    source_ops,
)


def _seed_source():
    return """def priority(rank, visits, total, prior, depth, n_nodes):
    c = 1.0
    return rank + c * (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)
"""


def _slim_source():
    # the same values with the c=1.0 multiply folded away: equal ordering,
    # strictly fewer arithmetic nodes
    return """def priority(rank, visits, total, prior, depth, n_nodes):
    return rank + (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)
"""


def test_source_ops_is_a_deterministic_cost_count():
    assert source_ops(_seed_source()) > source_ops(_slim_source())
    assert source_ops(_slim_source()) > 0
    assert source_ops("not python {") == 10 ** 6      # unparseable = infinitely costly


def test_cost_adjusted_reward_prefers_cheaper_at_equal_quality():
    reward = cost_adjusted_reward(lam=0.1, baseline=10)
    cheap = MetaOutcome(curve=[1.0, 1.0], final=1.0, detail={"ops": 5})
    dear = MetaOutcome(curve=[1.0, 1.0], final=1.0, detail={"ops": 10})
    assert reward(cheap) > reward(dear)
    # at the baseline's own cost it scores quality - lam
    assert reward(dear) == pytest.approx(0.9)


def test_the_grid_problem_scores_against_the_seed_ordering():
    from agentdescent import priority_selection, PRIORITY_SEED
    spec = priority_selection()
    seed = spec.compile(PRIORITY_SEED)
    problem = priority_grid_problem()
    outcome = problem(seed, 0)
    assert outcome.final == pytest.approx(1.0)        # the seed agrees with itself
    assert outcome.detail["ops"] == source_ops(PRIORITY_SEED)
    assert isinstance(outcome.curve, list) and outcome.curve


def test_the_seed_ordering_is_reproduced_by_an_equivalent_cheaper_rule():
    from agentdescent import priority_selection
    spec = priority_selection()
    slim = spec.compile(_slim_source())
    problem = priority_grid_problem()
    # same math -> same ordering -> equal quality, but fewer ops
    assert problem(slim, 0).final == pytest.approx(1.0)
    assert problem(slim, 0).detail["ops"] < source_ops(_seed_source())


def test_kendall_tau_is_a_true_rank_correlation():
    from agentdescent.evolved_programs import _kendall_tau
    ref = [3, 1, 4, 0, 2]          # non-identity, non-reversed reference
    # identical and exact-reverse orderings are the endpoints
    assert _kendall_tau(ref, list(ref)) == pytest.approx(1.0)
    assert _kendall_tau(ref, list(reversed(ref))) == pytest.approx(-1.0)
    # swapping the two items that are adjacent in the reference inverts exactly
    # one pair, so tau drops by 2 / C(5,2)
    swapped = list(ref)
    i, j = swapped.index(3), swapped.index(1)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    assert _kendall_tau(ref, swapped) == pytest.approx(1.0 - 2.0 / 10.0)
    # agree with a rank-based reference implementation on every permutation
    def reference_tau(a, b):
        ra = {v: p for p, v in enumerate(a)}
        rb = {v: p for p, v in enumerate(b)}
        agree = sum(1 for x in range(5) for y in range(x + 1, 5)
                    if (ra[x] < ra[y]) == (rb[x] < rb[y]))
        return 2 * agree / 10.0 - 1.0
    import itertools
    for perm in itertools.permutations(range(5)):
        assert _kendall_tau(ref, list(perm)) == pytest.approx(reference_tau(ref, list(perm)))


def test_evolve_fast_rule_beats_the_seed_on_quality_per_cost():
    from agentdescent import priority_selection

    spec = priority_selection()

    def slim_propose(rendered, task, output, reward):
        return _slim_source()          # propose the cheaper equivalent

    train = {"g": priority_grid_problem()}
    val = {"v": priority_grid_problem()}
    report, result = evolve_fast_rule(
        train, val, propose=slim_propose, seeds=(0, 1, 2, 3),
        val_seeds=(4, 5, 6, 7), rounds=1, n_workers=1, repo_path=None)
    # the evolved rule is the slim one: fewer ops, equal quality
    assert report.evolved_ops < report.seed_ops
    assert report.evolved_quality == pytest.approx(report.seed_quality)
    assert report.gain > 0
    assert "1.0 / n_nodes" in report.value


def test_evolve_fast_rule_needs_a_proposer(tmp_path):
    from agentdescent import PrioritySelection, priority_selection
    with pytest.raises(ValueError):
        evolve_fast_rule({"g": priority_grid_problem()},
                         {"v": priority_grid_problem()})


def test_evolve_fast_rule_runs_with_default_seeds(tmp_path):
    # the default seeds must provide enough outer tasks (>= 4), or the
    # documented call with no seeds would raise
    from agentdescent import priority_selection

    spec = priority_selection()

    def slim_propose(rendered, task, output, reward):
        return _slim_source()

    report, result = evolve_fast_rule(
        {"g": priority_grid_problem()}, {"v": priority_grid_problem()},
        propose=slim_propose, rounds=1, n_workers=1, repo_path=None)
    assert report.evolved_ops < report.seed_ops
    assert report.gain > 0


def test_priority_grid_problem_rejects_a_bad_reference():
    with pytest.raises(ValueError):
        priority_grid_problem(reference=[0, 1, 2, 3])     # not 0..4
    with pytest.raises(ValueError):
        priority_grid_problem(reference=[1, 2, 3, 4, 5])  # 5 not a grid index
    with pytest.raises(ValueError):
        priority_grid_problem(reference=[0, 1, 2, 3, 3])  # duplicated


def test_evolve_fast_rule_accepts_sequence_problems(tmp_path):
    # val_problems as a plain list used to crash on .values()
    from agentdescent import priority_selection

    spec = priority_selection()

    def slim_propose(rendered, task, output, reward):
        return _slim_source()

    train = [priority_grid_problem()]
    val = [priority_grid_problem()]
    report, result = evolve_fast_rule(
        train, val, propose=slim_propose, seeds=(0, 1, 2, 3),
        val_seeds=(4, 5, 6, 7), rounds=1, n_workers=1, repo_path=None)
    assert report.evolved_ops < report.seed_ops
    assert report.gain > 0
