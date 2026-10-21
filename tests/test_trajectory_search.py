"""Trajectory tree search has to be right where the loop becomes a learned MCTS.

The tree is over rendered states, expanded by a branch callable, valued by a
critic, and searched by UCB. The fixtures pin the surfaces that make this
honest: the tree back-propagates and selects by UCB, a monotone landscape is
climbed toward the optimum with the budget concentrating on the promising
branch, the report shows the estimate next to a freshly verified value, and a
landscape with a trap shows the tree spending its budget past it.
"""

import pytest

from agentdescent import (
    MetaOutcome,
    ParamSlot,
    SearchReport,
    SearchTree,
    search_evolve,
)
from agentdescent.sampling import DifficultyWeighted


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def _landscape(value, seed):
    c = getattr(value, "c", 0.2)
    return MetaOutcome(curve=[min(1.0, c)], final=min(1.0, c), rollouts=1)


def _trap(value, seed):
    c = getattr(value, "c", 0.2)
    r = 0.1 if 0.4 <= c <= 0.6 else min(1.0, c)     # a trap band
    return MetaOutcome(curve=[r], final=r, rollouts=1)


def _branch(rendered, seed):
    import re
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    c = float(m.group(1)) if m else 0.2
    return [f"c: {min(1.0, c + 0.1):.2f}", f"c: {min(1.0, c + 0.2):.2f}"]


def test_the_tree_back_propagates_and_selects():
    tree = SearchTree("c: 0.2", ucb_c=1.4)
    tree.expand("c: 0.2", ["c: 0.3", "c: 0.4"])
    tree.update("c: 0.3", 1.0, path=["c: 0.2", "c: 0.3"])
    tree.update("c: 0.4", 0.5, path=["c: 0.2", "c: 0.4"])
    # the good branch is selected for expansion (higher mean + bonus)
    assert tree.select() == "c: 0.3"
    assert tree.best() == "c: 0.3"
    assert tree.n_nodes == 3


def test_search_evolve_climbs_a_monotone_landscape():
    best, report = search_evolve(
        _landscape, _sampler_slot(), "c: 0.2", _branch,
        seeds=(0,), budget=30, ucb_c=1.4)
    assert isinstance(report, SearchReport)
    assert report.budget == 30
    assert report.nodes >= 5                        # a real tree was built
    # the tree reached the top of the landscape
    assert report.verified_value == pytest.approx(1.0)
    assert "c: 1.00" in best


def test_ucb_concentrates_on_the_promising_branch():
    best, report = search_evolve(
        _landscape, _sampler_slot(), "c: 0.2", _branch,
        seeds=(0,), budget=20, ucb_c=1.4)
    assert report.depth >= 5                    # a deep promising path
    assert report.verified_value == pytest.approx(1.0)


def test_the_tree_spends_past_a_trap():
    best, report = search_evolve(
        _trap, _sampler_slot(), "c: 0.2", _branch,
        seeds=(0,), budget=40, ucb_c=1.4)
    # UCB pushes past the trap band (c in [0.4, 0.6]) to the high region
    assert report.verified_value == pytest.approx(1.0)


def test_search_report_estimates_are_inspectable():
    best, report = search_evolve(
        _landscape, _sampler_slot(), "c: 0.2", _branch,
        seeds=(0, 1), budget=10, ucb_c=1.4)
    assert 0.0 <= report.value_estimate <= 1.0
    assert 0.0 <= report.verified_value <= 1.0


def test_search_returns_the_best_evaluated_state():
    # root(0) -> a(1) -> b(0): back-prop makes root and a tie on mean, so the
    # old tree.best() returned root by insertion order and lost the 1.0 state
    def land(value, seed):
        c = getattr(value, "c", 0.2)
        v = {0.2: 0.0, 0.3: 1.0, 0.4: 0.0}.get(round(c, 1), 0.0)
        return MetaOutcome(curve=[v], final=v, rollouts=1)

    def chain(rendered, seed):
        import re
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.2
        return [f"c: {min(1.0, c + 0.1):.2f}"]

    best, report = search_evolve(
        land, _sampler_slot(), "c: 0.20", chain, seeds=(0,), budget=2)
    assert best == "c: 0.30"                 # the evaluated 1.0 state
    assert report.verified_value == pytest.approx(1.0)


def test_a_back_stepping_branch_does_not_crash():
    # a branch that can step back builds an ancestor cycle; the search must
    # finish without RecursionError or an infinite select
    def land(value, seed):
        c = getattr(value, "c", 0.2)
        return MetaOutcome(curve=[c], final=c, rollouts=1)

    def bounce(rendered, seed):
        import re
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.5
        return [f"c: {max(0.0, c - 0.1):.2f}", f"c: {min(1.0, c + 0.1):.2f}"]

    best, report = search_evolve(
        land, _sampler_slot(), "c: 0.50", bounce, seeds=(0,), budget=10)
    assert "c:" in best
    assert report.budget == 10


def test_a_dead_end_leaf_does_not_starve_the_search():
    # c:1.00 is a dead end (branch returns nothing) and scores 0; the search
    # must mark it visited and move on to the good siblings instead of spinning
    def land(value, seed):
        c = getattr(value, "c", 0.2)
        if round(c, 1) == 1.0:
            return MetaOutcome(curve=[0.0], final=0.0, rollouts=1)
        return MetaOutcome(curve=[min(1.0, c)], final=min(1.0, c), rollouts=1)

    def branch(rendered, seed):
        import re
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.2
        if c >= 0.8:
            return []
        return [f"c: {min(1.0, c + 0.1):.2f}", f"c: {min(1.0, c + 0.2):.2f}"]

    best, report = search_evolve(
        land, _sampler_slot(), "c: 0.20", branch, seeds=(0,), budget=10)
    assert best != "c: 1.00"                 # the dead end was not the answer
    assert report.verified_value > 0.5
