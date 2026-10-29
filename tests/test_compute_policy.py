"""Compute allocation must choose a strategy by difficulty, cost and evidence."""

import pytest

from agentdescent.compute_policy import (
    ComputeAction,
    ComputeContext,
    ComputeController,
    ComputeOption,
)
from agentdescent.model_router import ModelRouter, ModelSpec


def _router(*, exploration=0.0, seed=0):
    return ModelRouter([
        ModelSpec("cheap", cost=1, quality_prior=(5, 5)),
        ModelSpec("strong", cost=10, quality_prior=(5, 5)),
    ], cost_weight=0.0, exploration=exploration, seed=seed)


def _options():
    return [
        ComputeOption("direct", ComputeAction.DIRECT, calls=1,
                      quality_prior=(1, 3), max_difficulty=0.5),
        ComputeOption("prm", ComputeAction.PRM_RERANK, calls=2,
                      quality_prior=(4, 2), min_difficulty=0.3,
                      max_difficulty=0.7),
        ComputeOption("mcts", ComputeAction.MCTS, calls=8,
                      quality_prior=(6, 2), min_difficulty=0.6),
    ]


def test_difficulty_band_selects_the_strategy():
    controller = ComputeController(_options(), _router(), exploration=0.0)
    easy = controller.choose(ComputeContext(difficulty=0.1))
    assert easy.action == ComputeAction.DIRECT
    hard = controller.choose(ComputeContext(difficulty=0.9))
    assert hard.action == ComputeAction.MCTS
    assert hard.calls == 8


def test_max_cost_filters_expensive_strategies():
    controller = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, calls=1),
        ComputeOption("mcts", ComputeAction.MCTS, calls=8,
                      quality_prior=(6, 2)),
    ], _router(), exploration=0.0)
    decision = controller.choose(ComputeContext(
        difficulty=0.9, max_cost=2.0))
    assert decision.action == ComputeAction.DIRECT
    assert decision.expected_cost <= 2.0


def test_route_budget_is_per_call_not_per_option():
    router = ModelRouter([
        ModelSpec("cheap", cost=1, quality_prior=(1, 9)),
        ModelSpec("mid", cost=2, quality_prior=(9, 1)),
    ], cost_weight=0.0, exploration=0.0)
    controller = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, quality_prior=(1, 9)),
        ComputeOption("mcts", ComputeAction.MCTS, calls=4,
                      quality_prior=(9, 1)),
    ], router, exploration=0.0)
    decision = controller.choose(ComputeContext(max_cost=4.0))
    assert decision.option == "mcts"
    assert decision.model == "cheap"
    assert decision.expected_cost == 4.0


def test_calls_are_reserved_at_choose_time():
    controller = ComputeController([
        ComputeOption("mcts", ComputeAction.MCTS, calls=8),
    ], _router(), exploration=0.0, max_calls=8)
    first = controller.choose()
    assert controller.reserved_calls == 8
    with pytest.raises(RuntimeError):
        controller.choose()
    controller.observe(first, reward=0.5)
    assert controller.reserved_calls == 0
    assert controller.total_calls == 8
    with pytest.raises(RuntimeError):
        controller.choose()


def test_router_cost_stays_per_call():
    router = ModelRouter([ModelSpec("cheap", cost=1)],
                         cost_weight=0.0, exploration=0.0)
    controller = ComputeController([
        ComputeOption("mcts", ComputeAction.MCTS, calls=8),
    ], router, exploration=0.0)
    decision = controller.choose()
    controller.observe(decision, reward=0.5)
    assert router.report().models[0].mean_cost == pytest.approx(1.0)
    with pytest.raises(ValueError):
        controller.observe(decision, reward=0.5)


def test_observe_updates_the_option_and_model_posteriors():
    router = _router()
    controller = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, calls=1,
                      quality_prior=(1, 1)),
    ], router, exploration=0.0)
    decision = controller.choose()
    assert decision.model in ("cheap", "strong")
    before = controller.report().options[0].posterior_mean
    controller.observe(decision, reward=1.0)
    after = controller.report().options[0].posterior_mean
    assert after > before
    report = controller.report()
    assert report.pulls == 1
    assert report.total_calls == 1
    assert report.options[0].pulls == 1
    assert report.router.pulls == 1
    assert report.router.models


def test_budget_stops_before_overspending_calls():
    controller = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, calls=1),
    ], _router(), exploration=0.0, max_calls=2)
    for _ in range(2):
        controller.observe(controller.choose(), reward=0.5)
    with pytest.raises(RuntimeError):
        controller.choose()


def test_same_seed_is_reproducible():
    one = ComputeController(_options(), _router(seed=7), seed=7)
    two = ComputeController(_options(), _router(seed=7), seed=7)
    context = ComputeContext(difficulty=0.8, tokens=100)
    first = one.choose(context)
    second = two.choose(context)
    assert first == second
    one.observe(first, reward=0.7)
    two.observe(second, reward=0.7)
    assert one.choose(context) == two.choose(context)


def test_required_tags_skip_options_without_a_matching_model():
    router = ModelRouter([
        ModelSpec("text", cost=1, tags=("text",)),
        ModelSpec("vision", cost=2, tags=("vision",)),
    ], cost_weight=0.0, exploration=0.0)
    controller = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, calls=1),
    ], router, exploration=0.0)
    with pytest.raises(ValueError):
        controller.choose(ComputeContext(required_tags=("audio",)))
    decision = controller.choose(ComputeContext(required_tags=("vision",)))
    assert decision.model == "vision"


def test_invalid_options_and_contexts_are_refused():
    with pytest.raises(ValueError):
        ComputeOption("x", "magic")
    with pytest.raises(ValueError):
        ComputeOption("x", ComputeAction.DIRECT, calls=0)
    with pytest.raises(ValueError):
        ComputeOption("x", ComputeAction.DIRECT, quality_prior=(0, 1))
    with pytest.raises(ValueError):
        ComputeOption("x", ComputeAction.DIRECT, min_difficulty=0.8,
                      max_difficulty=0.2)
    with pytest.raises(ValueError):
        ComputeContext(tokens=-1)
    with pytest.raises(ValueError):
        ComputeContext(difficulty=2)
    with pytest.raises(ValueError):
        ComputeContext(required_tags="vision")
    with pytest.raises(TypeError):
        ComputeController(_options(), object())
    with pytest.raises(ValueError):
        ComputeController([], _router())
