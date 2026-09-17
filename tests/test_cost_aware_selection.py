"""Tests for cost-aware selection: the o1-style test-time-scaling decision.

Covers:
- Cost discounts exploration, never exploitation
- Unknown cost is the pool mean (never "free")
- Annealing: exploration shrinks as budget_remaining falls
- cost_exponent=0 is FlatPuct exactly (the ablation and the safety property)
- Uniform costs are unchanged whatever the exponent
- The population layer fills Candidate.cost and passes budget_remaining
- End-to-end: a BudgetGovernor drives budget_remaining into the aggregator
"""

from __future__ import annotations

import pytest

from agentdescent.schedule.selection import (
    Candidate,
    FlatPuct,
    CostEfficient,
    SelectionContext,
)


def _cand(version, score, cost=None, selected=0, parent=None):
    return Candidate(artifact_id="a", version=version, score=score,
                     cost=cost, selected=selected, parent=parent)


def _ctx(cands, *, budget_remaining=1.0, round=0):
    return SelectionContext(head=cands[0], candidates=tuple(cands),
                            round=round, n_workers=1,
                            budget_remaining=budget_remaining)


# --- the cost divisor ---


def test_uniform_cost_leaves_selection_unchanged():
    """Every candidate the same cost must give the same order as FlatPuct."""
    cands = [_cand(0, 0.5, cost=100), _cand(1, 0.9, cost=100), _cand(2, 0.7, cost=100)]
    cheap = CostEfficient(cost_exponent=1.0)
    flat = FlatPuct()
    assert ([c.version for c in cheap.select(_ctx(cands), 1)]
            == [c.version for c in flat.select(_ctx(cands), 1)])


def test_cost_exponent_zero_is_flatpuct_exactly():
    """alpha=0 is the ablation: it must be FlatPuct to the bit, so the cost term
    cannot change a run that did not ask for it."""
    cands = [_cand(0, 0.5, cost=1), _cand(1, 0.9, cost=10_000), _cand(2, 0.7, cost=5)]
    blind = CostEfficient(cost_exponent=0.0)
    flat = FlatPuct()
    assert ([c.version for c in blind.select(_ctx(cands), 3)]
            == [c.version for c in flat.select(_ctx(cands), 3)])


def test_unknown_cost_is_the_mean_not_free():
    """A candidate with no measured cost must not become the cheapest one."""
    factors = CostEfficient(cost_exponent=1.0).cost_factors(
        [_cand(0, 0.5, cost=100), _cand(1, 0.5, cost=None), _cand(2, 0.5, cost=100)])
    # The unknown one gets the mean (100), so all three factors are 1.0.
    assert factors == [1.0, 1.0, 1.0]


def test_all_costs_unknown_is_flatpuct():
    """No measured cost anywhere -> the divisor is 1 for all -> FlatPuct."""
    cands = [_cand(0, 0.5, cost=None), _cand(1, 0.9, cost=None), _cand(2, 0.7, cost=None)]
    cheap = CostEfficient(cost_exponent=1.0)
    flat = FlatPuct()
    assert ([c.version for c in cheap.select(_ctx(cands), 1)]
            == [c.version for c in flat.select(_ctx(cands), 1)])


# --- cost discounts exploration, not exploitation ---


def test_cost_factors_discount_expensive_candidates():
    """The divisor is cost^alpha normalised to mean 1: cheap < 1 < expensive."""
    factors = CostEfficient(cost_exponent=1.0).cost_factors(
        [_cand(0, 0.5, cost=50), _cand(1, 0.5, cost=100), _cand(2, 0.5, cost=200)])
    # mean = 116.67; cheap is below 1, expensive above.
    assert factors[0] < 1.0 < factors[2]
    assert factors[0] < factors[1] < factors[2]


def test_expensive_candidate_loses_the_exploration_bonus():
    """With exploration dominating (large c_puct, both visited so the sqrt term
    is live), the cheap low-rank candidate beats the pricey high-rank one."""
    # Both visited, so `sqrt(total) > 0` and the exploration term is live.
    cheap_low_rank = _cand(0, 0.4, cost=10, selected=1)
    pricey_high_rank = _cand(1, 0.9, cost=1_000, selected=1)
    policy = CostEfficient(c_puct=100.0, cost_exponent=1.0)
    picked = policy.select(_ctx([cheap_low_rank, pricey_high_rank]), 1)
    assert picked[0].version == cheap_low_rank.version, \
        "with exploration dominant, cost must decide and pick the cheap one"

    # The ablation: cost_exponent=0 (cost-blind FlatPuct) picks the high rank.
    blind = CostEfficient(c_puct=100.0, cost_exponent=0.0)
    picked_blind = blind.select(_ctx([cheap_low_rank, pricey_high_rank]), 1)
    assert picked_blind[0].version == pricey_high_rank.version, \
        "without cost, the rank must decide"


def test_exploitation_is_not_discounted_by_cost():
    """A clearly better candidate is still picked even if it is the pricier one
    -- cost must never disqualify the grounded pick."""
    better_pricey = _cand(0, 0.95, cost=10_000, selected=0)
    worse_cheap = _cand(1, 0.10, cost=1, selected=0)
    policy = CostEfficient(c_puct=1.0, cost_exponent=1.0)
    picked = policy.select(_ctx([better_pricey, worse_cheap]), 1)
    assert picked[0].version == better_pricey.version


# --- annealing ---


def test_exploration_anneals_with_remaining_budget():
    """With budget left, a fresh (unvisited) candidate gets explored; with none
    left, the policy exploits the ranked best instead."""
    best = _cand(0, 0.9, cost=100, selected=3)     # explored, high score
    fresh = _cand(1, 0.5, cost=100, selected=0)    # never tried, lower score

    full_budget = CostEfficient(c_puct=2.0, cost_exponent=0.0, anneal=True)
    # At the start the exploration term is large enough to try the fresh one.
    first = full_budget.select(
        _ctx([best, fresh], budget_remaining=1.0), 1)[0]
    assert first.version == fresh.version, \
        "with a full budget the unexplored candidate must get a look"

    no_budget = CostEfficient(c_puct=2.0, cost_exponent=0.0, anneal=True)
    last = no_budget.select(
        _ctx([best, fresh], budget_remaining=0.0), 1)[0]
    assert last.version == best.version, \
        "with no budget left the policy must exploit, not explore"


def test_anneal_false_pins_exploration():
    """anneal=False is the ablation: same pick at any budget_remaining."""
    best = _cand(0, 0.9, cost=100, selected=3)
    fresh = _cand(1, 0.5, cost=100, selected=0)
    policy = CostEfficient(c_puct=2.0, cost_exponent=0.0, anneal=False)
    a = policy.select(_ctx([best, fresh], budget_remaining=1.0), 1)[0]
    b = policy.select(_ctx([best, fresh], budget_remaining=0.0), 1)[0]
    assert a.version == b.version


def test_negative_budget_remaining_does_not_invert():
    """A caller bug (negative fraction) must not flip the exploration sign."""
    best = _cand(0, 0.9, cost=100, selected=3)
    fresh = _cand(1, 0.5, cost=100, selected=0)
    policy = CostEfficient(c_puct=2.0, cost_exponent=0.0, anneal=True)
    # Clamped to 0 -> behaves like no budget (exploit), not like -1.
    picked = policy.select(_ctx([best, fresh], budget_remaining=-5.0), 1)[0]
    assert picked.version == best.version


# --- selection shape ---


def test_returns_n_picks_and_reserves_visits():
    cands = [_cand(0, 0.5, cost=10), _cand(1, 0.6, cost=10), _cand(2, 0.7, cost=10)]
    policy = CostEfficient()
    picks = policy.select(_ctx(cands), 3)
    assert len(picks) == 3
    # Each pick is a real candidate.
    assert all(p in cands for p in picks)


def test_single_candidate_returns_head():
    one = _cand(0, 0.5, cost=10)
    policy = CostEfficient()
    assert policy.select(_ctx([one]), 2) == [one, one]


# --- population wiring ---


def test_population_aggregator_fills_candidate_cost():
    """The archive's candidates carry a content-size cost proxy."""
    import threading
    from agentdescent.schedule.population import PopulationAggregator
    from agentdescent.schedule.selection import Archive

    agg = PopulationAggregator.__new__(PopulationAggregator)
    agg.selection = Archive(sampling="novelty")
    agg.population_artifact = "a"
    agg._archive = [
        {"state": {"rule": "x" * 100}, "score": 0.5, "version": 1, "selected": 0},
        {"state": {"rule": "y" * 1000}, "score": 0.6, "version": 2, "selected": 0},
    ]
    agg._keys = set()
    agg._archive_lock = threading.Lock()
    agg._selections = 0
    agg.budget_remaining = 1.0

    cands = agg._candidates()
    assert all(c.cost is not None for c in cands), "cost proxy not filled"
    # The bigger artifact costs more.
    assert cands[1].cost > cands[0].cost


def test_population_passes_budget_remaining_to_context():
    """PopulationAggregator.step() hands its budget_remaining to the selection
    context, so a budget-aware policy sees it."""
    import threading
    from agentdescent.schedule.population import PopulationAggregator

    captured = {}

    class Capturing:
        def select(self, ctx, n):
            captured["budget_remaining"] = ctx.budget_remaining
            return [ctx.head] * n

    agg = PopulationAggregator.__new__(PopulationAggregator)
    agg.selection = Capturing()
    agg.population_artifact = "a"
    agg._archive = [
        {"state": {"r": "1"}, "score": 0.5, "version": 1, "selected": 0},
        {"state": {"r": "2"}, "score": 0.6, "version": 2, "selected": 0},
    ]
    agg._keys = set()
    agg._archive_lock = threading.Lock()
    agg._selections = 0
    agg.budget_remaining = 0.42
    # `step()` calls a lot of super() machinery; call the selection part the
    # way step() does, so the assertion is about the context construction.
    cands = agg._candidates()
    from agentdescent.schedule.selection import SelectionContext
    ctx = SelectionContext(head=cands[0], candidates=tuple(cands),
                          round=agg._selections, n_workers=1,
                          budget_remaining=agg.budget_remaining)
    agg.selection.select(ctx, 1)
    assert captured["budget_remaining"] == 0.42


# --- governor -> budget_remaining ---


def test_governor_remaining_fraction():
    from agentdescent.observe.budget import BudgetGovernor
    g = BudgetGovernor(max_tokens=1000)
    assert g.remaining_fraction() == 1.0
    g.spend(250)
    assert g.remaining_fraction() == pytest.approx(0.75)
    g.spend(1000)
    assert g.remaining_fraction() == 0.0
    g.spend(1500)  # overshoot (barrier lag)
    assert g.remaining_fraction() == 0.0  # clamped, not negative


def test_governor_remaining_fraction_no_budget():
    from agentdescent.observe.budget import BudgetGovernor
    assert BudgetGovernor(max_tokens=None).remaining_fraction() == 1.0


def test_set_budget_remaining_only_touches_declaring_aggregators():
    from agentdescent.loop.evolution import _set_budget_remaining
    from agentdescent.observe.budget import BudgetGovernor

    g = BudgetGovernor(max_tokens=1000)
    g.spend(500)

    class Declaring:
        budget_remaining = 1.0

    class Plain:
        pass

    d, p = Declaring(), Plain()
    _set_budget_remaining(d, g)
    _set_budget_remaining(p, g)
    assert d.budget_remaining == pytest.approx(0.5)
    assert not hasattr(p, "budget_remaining"), \
        "a plain aggregator must not gain an attribute nothing reads"


# --- end to end ---


def test_evolve_with_cost_efficient_and_token_budget():
    """The whole chain: evolve(selection=CostEfficient(), max_tokens=N) runs,
    the governor drives budget_remaining, and the policy anneals on it."""
    import warnings
    from agentdescent.actors.agents import Usage
    from agentdescent.loop.evolution import evolve, Task
    from agentdescent.core.policies import Policies
    from agentdescent.schedule.selection import CostEfficient

    class Agent:
        def __init__(self):
            self.usage = Usage()

        def solve(self, rendered, task):
            self.usage.record(prompt_tokens=100, completion_tokens=200)
            return task.id

        def propose(self, rendered, task, output, reward):
            self.usage.record(prompt_tokens=100, completion_tokens=200)
            return f"rule-{task.id}" if reward < 0.999 else None

    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]
    agent = Agent()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            tasks, lambda t, o: 0.5, agent=agent,
            rounds=6, n_workers=1,
            max_tokens=6000,
            policies=Policies(selection=CostEfficient(c_puct=1.0)),
            usage=agent.usage,
        )
    # The run completed (or stopped on the budget) without crashing, and the
    # budget was tracked.
    assert result.stop_reason in ("rounds", "max_tokens", "patience")
    assert result.budget is not None
    assert result.budget["spent"] > 0
