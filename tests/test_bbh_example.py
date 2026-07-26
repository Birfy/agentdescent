"""Offline tests for the BBH skill-evolution example's pure helpers.

These exercise dataset shaping and scoring without any network or LLM calls.
"""

from examples.skill_evolution import (
    build_tasks,
    estimate_calls,
    extract_choice,
    is_multiple_choice,
    make_reward,
)


def test_multiple_choice_detection():
    assert is_multiple_choice("(B)")
    assert is_multiple_choice(" (D) ")
    assert not is_multiple_choice("barn damp delmarva")
    assert not is_multiple_choice("(AB)")


def test_extract_choice():
    assert extract_choice("The answer is (C).") == "C"
    assert extract_choice("... so it must be (A) then (D)") == "D"  # last wins
    assert extract_choice("D") == "D"
    assert extract_choice("no option here") is None


def test_mc_reward_is_exact_match():
    reward = make_reward(mc=True)
    from agentdescent.evolution import Task
    task = Task(id="t", prompt="q", meta={"target": "(B)", "mc": True})
    assert reward(task, "I think the answer is (B).") == 1.0
    assert reward(task, "The answer is (C).") == 0.0


def test_freeform_reward_is_graded():
    reward = make_reward(mc=False)
    from agentdescent.evolution import Task
    task = Task(id="t", prompt="q", meta={"target": "barn damp delmarva", "mc": False})
    assert reward(task, "barn damp delmarva") == 1.0
    partial = reward(task, "barn damp")
    assert 0.0 < partial < 1.0
    # format noise (a preamble) lowers the score.
    assert reward(task, "The sorted list is: barn damp delmarva") < 1.0


def test_build_tasks_shapes_examples():
    examples = [{"input": f"Sort: w{i}", "target": "(A)"} for i in range(20)]
    tasks = build_tasks(examples, limit=8, seed=1)
    assert len(tasks) == 8
    assert all(t.meta["mc"] for t in tasks)
    assert all(t.prompt and t.meta["target"] for t in tasks)
    # deterministic given the seed
    again = build_tasks(examples, limit=8, seed=1)
    assert [t.id for t in tasks] == [t.id for t in again]


def test_estimate_calls_scales():
    assert estimate_calls(4, 2, 10) > estimate_calls(2, 2, 10)
    assert estimate_calls(4, 4, 10) > estimate_calls(4, 2, 10)
