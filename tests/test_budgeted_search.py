"""Search should stop when another expansion stops buying value."""

import random

import pytest

from agentdescent.budgeted_search import BudgetedSearch
from agentdescent.planning import PlanResult
from agentdescent.search_fabric import SearchFabric, SearchResult
from agentdescent.stopping import MarginalGainStop


class _Fork:
    def actions(self, state):
        return ("bad", "good") if state == "root" else ()

    def step(self, state, action):
        from agentdescent.planning import PlanTransition
        return PlanTransition(action, reward=0.0, terminal=False)


def _fabric():
    from agentdescent.process_reward import ProcessRewardModel, StepExample
    prm = ProcessRewardModel()
    prm.fit([
        StepExample((1.0, 0.0), 1.0),
        StepExample((0.9, 0.1), 1.0),
        StepExample((0.0, 1.0), 0.0),
        StepExample((0.1, 0.9), 0.0),
    ], min_n=4)

    def features(state):
        return (1.0, 0.0) if state == "good" else (0.0, 1.0)

    return SearchFabric(_Fork(), prm, features, budget=5, seed=0)


def _fake_plan(values, fabric):
    def plan(_state):
        budget = fabric.planner.budget
        value = values[budget]
        result = PlanResult(action="good", value=value, visits=budget,
                            nodes=2, root_visits=budget)
        return SearchResult(plan=result, value=value, prm_score=value)
    return plan


def test_budgeted_search_stops_on_the_exact_plateau_round():
    fabric = _fabric()
    fabric.plan = _fake_plan({5: 0.5, 20: 0.6, 60: 0.6}, fabric)
    fabric.planner.budget = 999
    before_rng = fabric.planner.rng.getstate()
    search = BudgetedSearch(fabric, patience=1, min_gain_per_expansion=0.01,
                            seed=4)
    report = search.run("root", budgets=(5, 20, 60))
    assert [row.budget for row in report.rounds] == [5, 20]
    assert report.rounds[1].gain == pytest.approx(0.1)
    assert report.rounds[1].gain_per_expansion == pytest.approx(0.1 / 15)
    assert report.rounds[1].cumulative_expansions == 25
    assert report.rounds[1].stopped
    assert report.stopped_reason == "patience"
    assert fabric.planner.budget == 999
    assert fabric.planner.rng.getstate() == before_rng
    data = report.to_dict()
    assert data["rounds"][1]["cumulative_expansions"] == 25
    assert data["stopped_reason"] == "patience"


def test_marginal_gain_stop_path_and_all_budgets():
    fabric = _fabric()
    fabric.plan = _fake_plan({5: 0.5, 20: 0.5, 60: 0.5}, fabric)
    search = BudgetedSearch(
        fabric, patience=5, min_gain_per_expansion=0.0,
        stop=MarginalGainStop(min_gain_per_call=1e-4, patience=1), seed=0)
    report = search.run("root", budgets=(5, 20, 60))
    assert len(report.rounds) == 2
    assert report.stopped_reason == "stopping"
    assert report.rounds[-1].stopped


def test_invalid_inputs_are_refused():
    with pytest.raises(TypeError):
        BudgetedSearch(object())
    with pytest.raises(ValueError):
        BudgetedSearch(_fabric(), patience=0)
    with pytest.raises(ValueError):
        BudgetedSearch(_fabric(), min_gain_per_expansion=-1)
    with pytest.raises(TypeError):
        BudgetedSearch(_fabric(), stop=object())
    with pytest.raises(ValueError):
        BudgetedSearch(_fabric(), seed=True)
    with pytest.raises(ValueError):
        BudgetedSearch(_fabric()).run("root", budgets=())
    with pytest.raises(ValueError):
        BudgetedSearch(_fabric()).run("root", budgets=(0,))
    with pytest.raises(ValueError):
        BudgetedSearch(_fabric()).run("root", budgets=(1.5,))
    with pytest.raises(ValueError):
        BudgetedSearch(_fabric()).run("root", budgets=(True,))
