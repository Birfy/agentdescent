"""Budgeted PRM/verifier search should be a first-class loop step."""

from types import SimpleNamespace

import pytest

from agentdescent import budgeted_loop
from agentdescent.budgeted_loop import BudgetedStep, BudgetedStepConfig
from agentdescent.budgeted_search import BudgetedSearch, BudgetedSearchReport
from agentdescent.planning import PlanTransition
from agentdescent.process_reward import ProcessRewardModel, StepExample
from agentdescent.search_fabric import SearchFabric
from agentdescent.verifier_ensemble import VerifierEnsemble, VerifierSpec


class _Fork:
    def actions(self, state):
        return ("good", "bad") if state == "root" else ()

    def step(self, state, action):
        return PlanTransition(action, reward=0.0, terminal=False)


class _Flat:
    def actions(self, state):
        return ("left", "right") if state == "root" else ()

    def step(self, state, action):
        return PlanTransition("leaf", reward=0.0, terminal=False)


def _prm():
    model = ProcessRewardModel()
    model.fit([
        StepExample((1.0, 0.0), 1.0),
        StepExample((0.9, 0.1), 1.0),
        StepExample((0.0, 1.0), 0.0),
        StepExample((0.1, 0.9), 0.0),
    ], min_n=4)
    return model


def _fork_features(state):
    return (1.0, 0.0) if state == "good" else (0.0, 1.0)


def _flat_features(state):
    return (1.0, 0.0)


def _fabric(simulator=None, features=_fork_features, **kwargs):
    return SearchFabric(simulator or _Fork(), _prm(), features, budget=5,
                        seed=0, **kwargs)


def _context(budget_scale=1.0, seed=3):
    return SimpleNamespace(budget_scale=budget_scale, seed=seed)


def test_step_returns_the_budgeted_search_best_as_an_observation():
    fabric = _fabric()
    config = BudgetedStepConfig(
        budgets=(4, 8, 16), min_gain_per_expansion=0.0, patience=10)
    step = BudgetedStep(fabric, lambda context: "root", config=config)

    observation = step(_context(seed=3))
    report = BudgetedSearch(
        fabric, min_gain_per_expansion=config.min_gain_per_expansion,
        patience=config.patience).run("root", budgets=(4, 8, 16))

    assert observation.reward == pytest.approx(report.best.value)
    assert observation.cost == report.rounds[-1].cumulative_expansions
    assert observation.cost > 0
    assert observation.risk == 0.0
    assert observation.artifact_id == f"search:3:{report.best.action}"
    for key in ("stopped_reason", "best_value", "rounds", "prm_score",
                "verifier_score", "verifier_complete", "detail"):
        assert key in observation.evidence
    assert observation.evidence["best_value"] == pytest.approx(report.best.value)
    assert observation.evidence["rounds"][0]["budget"] == 4
    assert set(observation.evidence["rounds"][0]) >= {
        "budget", "value", "gain", "gain_per_expansion", "stopped"}


def test_budget_scale_multiplies_each_configured_budget():
    fabric = _fabric()
    config = BudgetedStepConfig(
        budgets=(4, 8), min_gain_per_expansion=0.0, patience=10)
    step = BudgetedStep(fabric, lambda context: "root", config=config)

    base = step(_context(budget_scale=1.0, seed=1))
    scaled = step(_context(budget_scale=2.0, seed=1))

    assert scaled.cost > base.cost
    assert [row["budget"] for row in base.evidence["rounds"]] == [4, 8]
    assert [row["budget"] for row in scaled.evidence["rounds"]] == [8, 16]


def test_scale_budget_false_ignores_the_controller_scale():
    fabric = _fabric()
    config = BudgetedStepConfig(
        budgets=(4, 8), min_gain_per_expansion=0.0, patience=10,
        scale_budget=False)
    step = BudgetedStep(fabric, lambda context: "root", config=config)

    base = step(_context(budget_scale=1.0, seed=1))
    scaled = step(_context(budget_scale=100.0, seed=1))

    assert scaled.cost == base.cost
    assert [row["budget"] for row in scaled.evidence["rounds"]] == [4, 8]


def test_flat_landscape_stops_before_spending_every_budget():
    fabric = _fabric(simulator=_Flat(), features=_flat_features)
    config = BudgetedStepConfig(
        budgets=(4, 8, 16, 32), min_gain_per_expansion=1e-4, patience=2)
    step = BudgetedStep(fabric, lambda context: "root", config=config)

    observation = step(_context(seed=9))
    rounds = observation.evidence["rounds"]

    assert observation.evidence["stopped_reason"] == "patience"
    assert len(rounds) == 3
    assert rounds[-1]["stopped"] is True
    assert observation.cost < sum(config.budgets)


def test_incomplete_verifier_panel_is_not_treated_as_a_pass():
    ensemble = VerifierEnsemble([
        VerifierSpec("one", lambda candidate: 1.0, cost=1.0),
        VerifierSpec("two", lambda candidate: 1.0, cost=1.0),
    ], strategy="weighted", min_votes=2)
    fabric = _fabric(verifier=ensemble, verifier_cost=1.0)
    step = BudgetedStep(
        fabric, lambda context: "root",
        config=BudgetedStepConfig(budgets=(4,), patience=2))

    observation = step(_context(seed=5))

    assert observation.evidence["verifier_complete"] is False
    assert observation.reward == pytest.approx(observation.evidence["prm_score"])


@pytest.mark.parametrize("kwargs", [
    {"budgets": ()},
    {"budgets": (0,)},
    {"budgets": (1.5,)},
    {"budgets": (True,)},
    {"patience": 0},
    {"patience": True},
    {"min_gain_per_expansion": float("nan")},
    {"min_gain_per_expansion": -1.0},
    {"artifact_prefix": ""},
    {"scale_budget": 1},
])
def test_invalid_configs_are_refused(kwargs):
    with pytest.raises(ValueError):
        BudgetedStepConfig(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_risk_is_refused(value):
    step = BudgetedStep(
        _fabric(), lambda context: "root", risk=lambda best: value,
        config=BudgetedStepConfig(budgets=(4,), patience=2))

    with pytest.raises(ValueError):
        step(_context(seed=0))


def test_risk_and_evidence_callbacks_are_composed():
    step = BudgetedStep(
        _fabric(), lambda context: "root", risk=lambda best: 0.25,
        evidence_of=lambda best: {"action": str(best.action)},
        config=BudgetedStepConfig(budgets=(4,), patience=2))

    observation = step(_context(seed=2))

    assert observation.risk == 0.25
    assert observation.evidence["action"] == "good"
    assert observation.evidence["detail"] == observation.evidence["plan_detail"]


def test_no_best_result_becomes_an_error_observation(monkeypatch):
    class _EmptySearch:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, state, *, budgets):
            return BudgetedSearchReport(
                rounds=(), best=None, stopped_reason="patience")

    monkeypatch.setattr(budgeted_loop, "BudgetedSearch", _EmptySearch)
    step = BudgetedStep(
        _fabric(), lambda context: "root",
        config=BudgetedStepConfig(budgets=(4, 8), patience=2))

    observation = step(_context(seed=0))

    assert observation.reward == 0.0
    assert observation.cost == 12
    assert observation.error == "budgeted search produced no result"
    assert observation.evidence["budgets"] == [4, 8]


def test_wrong_collaborator_types_are_refused():
    fabric = _fabric()
    with pytest.raises(TypeError):
        BudgetedStep(object(), lambda context: "root")
    with pytest.raises(TypeError):
        BudgetedStep(fabric, 5)
    with pytest.raises(TypeError):
        BudgetedStep(fabric, lambda context: "root", config=object())
    with pytest.raises(TypeError):
        BudgetedStep(fabric, lambda context: "root", risk=5)
    with pytest.raises(TypeError):
        BudgetedStep(fabric, lambda context: "root", evidence_of=5)
