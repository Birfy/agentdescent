"""A process reward model must be able to drive the planner and reranker."""

import pytest

from agentdescent.planning import PlanTransition
from agentdescent.process_reward import ProcessRewardModel, StepExample
from agentdescent.search_fabric import SearchFabric
from agentdescent.verifier_ensemble import VerifierEnsemble, VerifierSpec


class _Fork:
    def actions(self, state):
        return ("bad", "good") if state == "root" else ()

    def step(self, state, action):
        return PlanTransition(action, reward=0.0, terminal=False)


def _prm():
    model = ProcessRewardModel()
    model.fit([
        StepExample((1.0, 0.0), 1.0),
        StepExample((0.9, 0.1), 1.0),
        StepExample((0.0, 1.0), 0.0),
        StepExample((0.1, 0.9), 0.0),
    ], min_n=4)
    return model


def _features(state):
    return (1.0, 0.0) if state == "good" else (0.0, 1.0)


def test_prm_value_guides_uct_and_reranks_candidates():
    fabric = SearchFabric(_Fork(), _prm(), _features,
                          budget=60, seed=0)
    result = fabric.plan("root")
    assert result.action == "good"
    assert result.prm_score > 0.9
    assert result.value > 0.9
    assert fabric.rank(["bad", "good"]) == (1, 0)
    assert fabric.select(["bad", "good"]) == 1


def test_verifier_blends_into_the_value():
    verifier = VerifierEnsemble([
        VerifierSpec("is-good", lambda state: state == "good", cost=1),
    ])
    fabric = SearchFabric(_Fork(), _prm(), _features,
                          budget=60, seed=0, verifier=verifier,
                          verifier_weight=0.5)
    result = fabric.plan("root")
    assert result.action == "good"
    assert result.verifier_score is not None
    assert result.verifier_passed is True
    assert "verifier" in result.detail


class _CountingFork(_Fork):
    def __init__(self):
        self.steps = 0

    def step(self, state, action):
        self.steps += 1
        return super().step(state, action)


def test_plan_does_not_step_the_simulator_twice():
    simulator = _CountingFork()
    fabric = SearchFabric(simulator, _prm(), _features, budget=20, seed=0)
    result = fabric.plan("root")
    # one step per expanded node; the reported target state comes from UCT.
    assert simulator.steps == result.plan.nodes - 1
    assert result.plan.state == "good"


def test_incomplete_verifier_panel_cannot_report_a_pass():
    verifier = VerifierEnsemble([
        VerifierSpec("cheap", lambda state: state == "good", cost=1),
        VerifierSpec("strict", lambda state: False, cost=5),
    ], strategy="cascade")
    fabric = SearchFabric(_Fork(), _prm(), _features,
                          budget=20, seed=0, verifier=verifier,
                          verifier_weight=0.5, verifier_cost=2)
    result = fabric.plan("root")
    assert result.verifier_complete is False
    assert result.verifier_passed is False
    assert "incomplete" in result.detail


def test_invalid_configuration_is_refused():
    with pytest.raises(TypeError):
        SearchFabric(_Fork(), object(), _features)
    with pytest.raises(TypeError):
        SearchFabric(_Fork(), _prm(), None)
    with pytest.raises(ValueError):
        SearchFabric(_Fork(), _prm(), _features, verifier_weight=2)
    with pytest.raises(ValueError):
        SearchFabric(_Fork(), _prm(), _features, verifier_cost=-1)
    with pytest.raises(RuntimeError):
        SearchFabric(_Fork(), ProcessRewardModel(), _features).plan("root")
