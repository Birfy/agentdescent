"""Routing must trade quality against cost without starving either side."""

import pytest

from agentdescent.model_router import (
    ModelRouter,
    ModelSpec,
    RouteContext,
)


def _router(**kwargs):
    return ModelRouter([
        ModelSpec("fast", cost=1, max_context=256_000, tags=("text",),
                  quality_prior=(2, 2)),
        ModelSpec("strong", cost=20, tags=("text", "vision"), quality_prior=(6, 2)),
    ], **kwargs)


def test_cost_weight_prefers_the_cheap_model_when_quality_is_close():
    router = _router(cost_weight=1.0, exploration=0.0)
    picks = [router.choose(RouteContext(tokens=1000)).model for _ in range(40)]
    assert picks.count("fast") > picks.count("strong")


def test_strong_prior_wins_when_cost_weight_is_low():
    router = _router(cost_weight=0.0, exploration=0.0)
    picks = [router.choose(RouteContext(tokens=1000)).model for _ in range(40)]
    assert picks.count("strong") > picks.count("fast")


def test_context_filters_tokens_tags_and_budget():
    router = _router()
    assert router.choose(RouteContext(tokens=200_000)).model == "fast"
    with pytest.raises(ValueError):
        router.choose(RouteContext(tokens=300_000))
    assert router.choose(RouteContext(required_tags=("vision",))).model == "strong"
    assert router.choose(RouteContext(max_cost=1)).model == "fast"
    with pytest.raises(ValueError):
        router.choose(RouteContext(required_tags=("audio",)))


def test_exploration_is_flagged_and_can_be_disabled():
    router = _router(exploration=1.0)
    decision = router.choose(RouteContext())
    assert decision.exploration and decision.reason == "exploration"
    one = ModelRouter([ModelSpec("a", 1), ModelSpec("b", 2)],
                      exploration=0.0, seed=3)
    two = ModelRouter([ModelSpec("a", 1), ModelSpec("b", 2)],
                      exploration=0.0, seed=3)
    assert one.choose().model == two.choose().model


def test_observe_updates_posterior_and_report():
    router = _router(exploration=0.0)
    router.observe("fast", reward=1.0, cost=2)
    router.observe("fast", reward=0.0, cost=4)
    report = router.report()
    fast = next(row for row in report.models if row.model == "fast")
    assert fast.pulls == 2
    assert fast.mean_reward == 0.5
    assert fast.mean_cost == 3
    assert report.pulls == 2


def test_same_seed_is_reproducible():
    one = _router(seed=7)
    two = _router(seed=7)
    assert [one.choose().model for _ in range(10)] == [two.choose().model for _ in range(10)]


def test_invalid_models_contexts_and_observations_are_refused():
    with pytest.raises(ValueError):
        ModelRouter([])
    with pytest.raises(ValueError):
        ModelRouter([ModelSpec("a", 1), ModelSpec("a", 1)])
    with pytest.raises(ValueError):
        ModelSpec("a", -1)
    with pytest.raises(ValueError):
        ModelSpec("a", 1, quality_prior=(0, 1))
    with pytest.raises(ValueError):
        RouteContext(tokens=-1)
    with pytest.raises(ValueError):
        RouteContext(difficulty=2)
    router = _router()
    with pytest.raises(KeyError):
        router.observe("missing", reward=1)
    with pytest.raises(ValueError):
        router.observe("fast", reward=2)
    with pytest.raises(ValueError):
        router.observe("fast", reward=0.5, cost=-1)


def test_difficulty_buys_more_quality():
    def picks(difficulty):
        router = ModelRouter([
            ModelSpec("fast", cost=1, max_context=256_000),
            ModelSpec("strong", cost=20, quality_prior=(6, 2)),
        ], cost_weight=1.0, exploration=0.0, seed=5)
        return sum(1 for _ in range(80)
                   if router.choose(RouteContext(difficulty=difficulty)).model == "strong")

    assert picks(1.0) > picks(0.0)


def test_expected_quality_is_the_posterior_mean_not_the_sample():
    router = _router(exploration=0.0)
    router.observe("fast", reward=1.0)
    decision = router.choose(RouteContext(difficulty=0.0))
    stats = router._stats[decision.model]
    assert decision.expected_quality == pytest.approx(
        stats["alpha"] / (stats["alpha"] + stats["beta"]))
    assert 0.0 <= decision.sampled_quality <= 1.0
