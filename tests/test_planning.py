"""The planner must find the action a simulator rewards, within budget."""

import pytest

from agentdescent.planning import (
    MCTSPlanner,
    PlanTransition,
)


class _Fork:
    def actions(self, state):
        return {"root": ("good", "bad"), "mid": ("win",)}.get(state, ())

    def step(self, state, action):
        if state == "root" and action == "good":
            return PlanTransition("good", reward=10, terminal=True)
        if state == "root" and action == "bad":
            return PlanTransition("bad", reward=0, terminal=True)
        if state == "root" and action == "mid":
            return PlanTransition("mid", reward=0)
        if state == "mid" and action == "win":
            return PlanTransition("end", reward=5, terminal=True)
        raise AssertionError((state, action))


def test_finds_the_best_immediate_action():
    planner = MCTSPlanner(_Fork(), lambda _state: 0.0, budget=40, seed=0)
    result = planner.plan("root")
    assert result.action == "good"
    assert result.visits > 0
    assert result.nodes > 1
    assert result.root_visits == 40


def test_finds_a_multistep_path():
    class Fork(_Fork):
        def actions(self, state):
            return {"root": ("bad", "mid"), "mid": ("win",)}.get(state, ())

    planner = MCTSPlanner(Fork(), lambda _state: 0.0, budget=120, seed=2)
    assert planner.plan("root").action == "mid"


def test_value_function_guides_nonterminal_leaves():
    class Fork(_Fork):
        def actions(self, state):
            return {"root": ("a", "b")}.get(state, ())

        def step(self, state, action):
            return PlanTransition(action, reward=0, terminal=False)

    planner = MCTSPlanner(Fork(), lambda state: 10 if state == "a" else 0,
                          budget=50)
    assert planner.plan("root").action == "a"


def test_terminal_return_does_not_decay_with_visits():
    planner = MCTSPlanner(_Fork(), lambda _state: 0.0, budget=100, seed=0)
    result = planner.plan("root")
    assert result.action == "good"
    assert result.value == pytest.approx(10.0)
    assert result.visits > 1


def test_multistep_path_wins_even_when_lexicographic_order_prefers_the_other_arm():
    class Fork(_Fork):
        def actions(self, state):
            return {"root": ("bad", "aaa"), "aaa": ("win",)}.get(state, ())

        def step(self, state, action):
            if state == "root" and action == "aaa":
                return PlanTransition("aaa", reward=0)
            if state == "aaa" and action == "win":
                return PlanTransition("end", reward=5, terminal=True)
            return super().step(state, action)

    planner = MCTSPlanner(Fork(), lambda _state: 0.0, budget=120, seed=2)
    assert planner.plan("root").action == "aaa"


def test_immediate_reward_on_a_nonterminal_edge_is_backed_up():
    class Fork:
        def actions(self, state):
            return ("rewarded", "plain") if state == "root" else ()

        def step(self, state, action):
            return PlanTransition(action, reward=3 if action == "rewarded" else 0)

    planner = MCTSPlanner(Fork(), lambda _state: 0.0, budget=40, seed=1)
    result = planner.plan("root")
    assert result.action == "rewarded"
    assert result.value == pytest.approx(3.0)


def test_cost_weight_penalises_the_expensive_terminal_edge():
    class Fork:
        def actions(self, state):
            return ("expensive", "cheap")

        def step(self, state, action):
            cost = 5 if action == "expensive" else 0
            return PlanTransition(action, reward=10, terminal=True, cost=cost)

    planner = MCTSPlanner(Fork(), lambda _state: 0.0, budget=60,
                          cost_weight=1.0, seed=0)
    result = planner.plan("root")
    assert result.action == "cheap"
    assert result.value == pytest.approx(10.0)


def test_same_seed_is_reproducible():
    one = MCTSPlanner(_Fork(), lambda _s: 0.0, budget=30, seed=7)
    two = MCTSPlanner(_Fork(), lambda _s: 0.0, budget=30, seed=7)
    assert one.plan("root") == two.plan("root")


def test_invalid_simulator_values_and_empty_state_are_refused():
    with pytest.raises(TypeError):
        MCTSPlanner(object(), lambda _s: 0)
    with pytest.raises(TypeError):
        MCTSPlanner(_Fork(), None)
    with pytest.raises(ValueError):
        MCTSPlanner(_Fork(), lambda _s: 0, budget=0)
    class NonTerminal:
        def actions(self, state):
            return ("stay",)

        def step(self, state, action):
            return PlanTransition(state, reward=0, terminal=False)

    with pytest.raises(ValueError):
        MCTSPlanner(NonTerminal(), lambda _s: float("nan"), budget=10).plan("root")
    with pytest.raises(ValueError):
        MCTSPlanner(_Fork(), lambda _s: 0, budget=10).plan("terminal")
    with pytest.raises(ValueError):
        PlanTransition("x", reward=float("nan"))
    with pytest.raises(ValueError):
        PlanTransition("x", cost=-1)
