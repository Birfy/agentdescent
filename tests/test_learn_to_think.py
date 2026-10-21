"""Learn-to-think has to be right where the framework learns when to spend.

The route rule is gated source answering `direct` / `search` / `reflect`; a
problem is one workload instance scored by quality-per-cost; `learn_to_think`
evolves the route on that reward. The fixtures pin the surfaces that make this
honest: the gate refuses imports / bad signatures / invalid returns, the seed
routes sensibly, the problem carries its decision and cost, and an evolved rule
that searches the hard tasks beats the seed's budget-conditional rule on
quality-per-cost.
"""

import pytest

from agentdescent import (
    LearnToThinkReport,
    MetaOutcome,
    learn_to_think,
    quality_per_cost,
    think_policy_spec,
    think_problem,
)


SEARCH_ON_HARD = """class Policy:
    # search whenever the task is hard, whatever the budget
    def route(self, difficulty, budget_left, reward):
        if difficulty >= 0.7:
            return "search"
        if difficulty >= 0.4:
            return "reflect"
        return "direct"
"""

DIRECT_ONLY = """class Policy:
    def route(self, difficulty, budget_left, reward):
        return "direct"
"""

BETTER_ROUTE = """class Policy:
    # a low budget cannot fund a search (it truncates): answer directly; with
    # budget, search the hard tasks, reflect the medium ones, answer the easy
    # ones directly -- the seed's blind spot is the medium range, where it
    # under-spends and answers directly
    def route(self, difficulty, budget_left, reward):
        if budget_left < 0.5:
            return "direct"
        if difficulty >= 0.6:
            return "search"
        if difficulty >= 0.1:
            return "reflect"
        return "direct"
"""


def test_the_seed_routes_sensibly():
    from agentdescent.learn_to_think import ROUTE_SEED
    spec = think_policy_spec()
    policy = spec.compile(ROUTE_SEED)
    assert policy.route(0.2, 0.9, 0.4) == "direct"
    assert policy.route(0.9, 0.9, 0.1) == "search"


def test_the_gate_refuses_unsafe_rules():
    spec = think_policy_spec()
    with pytest.raises(ValueError):
        spec.compile("import os\nclass Policy:\n    def route(self,d,b,r):\n        return 'direct'\n")
    with pytest.raises(ValueError):
        spec.compile("class Policy:\n    def route(self,d,b):\n        return 'direct'\n")
    with pytest.raises(ValueError):
        spec.compile("class Policy:\n    def route(self,d,b,r):\n        return 'maybe'\n")


def test_the_problem_carries_its_decision_and_cost():
    spec = think_policy_spec()
    policy = spec.compile(spec.render(spec.initial()))
    problem = think_problem()
    outcome = problem(policy, 9)                 # difficulty 0.9, budget 0.2
    assert isinstance(outcome, MetaOutcome)
    assert outcome.detail["route"] == "reflect"  # seed: hard, low budget -> reflect
    assert outcome.detail["difficulty"] == 0.9
    assert "cost" in outcome.detail
    assert quality_per_cost(outcome) == pytest.approx(outcome.final)


def test_searching_overdraws_the_budget_and_the_optimum_route_depends_on_it():
    spec = think_policy_spec()
    direct = spec.compile(DIRECT_ONLY)
    search = spec.compile(SEARCH_ON_HARD)
    problem = think_problem()
    # low budget (seed 6: d=0.6, budget 0.4): the search overdrew and is
    # truncated, so the cheap direct answer scores higher
    low_direct = problem(direct, 6)
    low_search = problem(search, 6)
    assert low_direct.detail["budget_left"] == 0.4
    assert low_direct.final > low_search.final
    # high budget (seed 7: d=0.7, budget 1.0): searching the hard task pays
    high_direct = problem(direct, 7)
    high_search = problem(search, 7)
    assert high_search.detail["budget_left"] == 1.0
    assert high_search.final > high_direct.final


def test_learn_to_think_evolves_a_better_route():
    problems = {"t": think_problem()}
    val = {"v": think_problem()}
    report, result = learn_to_think(
        problems, val, propose=lambda r, t, o, w: BETTER_ROUTE,
        # the seed's medium-difficulty blind spots are the held-out tail
        seeds=(4, 5, 7, 8, 1, 2, 3), val_seeds=(8, 9, 0, 1),
        rounds=1, n_workers=1, repo_path=None)
    assert isinstance(report, LearnToThinkReport)
    # the evolved rule searches the hard task with budget, answers easy ones
    assert report.routes_on["hard"] == "search"
    assert report.routes_on["easy"] == "direct"
    assert report.evolved_score > report.seed_score
    assert report.gain > 0
    assert result.meta["gain"] == pytest.approx(report.gain)


def test_learn_to_think_accepts_sequence_problems():
    problems = [think_problem()]
    val = [think_problem()]
    report, result = learn_to_think(
        problems, val, propose=lambda r, t, o, w: BETTER_ROUTE,
        seeds=(4, 5, 7, 8, 1, 2, 3), val_seeds=(8, 9, 0, 1),
        rounds=1, n_workers=1, repo_path=None)
    assert isinstance(report, LearnToThinkReport)
    assert report.evolved_score > report.seed_score
    assert report.gain > 0


def test_learn_to_think_needs_a_proposer():
    with pytest.raises(ValueError):
        learn_to_think({"t": think_problem()}, {"v": think_problem()})
