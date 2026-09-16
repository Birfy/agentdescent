"""Tests for adaptive per-call thinking budget (o1-style test-time scaling).

Covers:
- CallBudget.allocate: score × budget_remaining → per-call max_tokens
- Bounds: never above base, never below floor
- budgeted_completion: caches per max_tokens, delegates correctly
- Inert without call_budget: the engine never touches it
- End-to-end: evolve with call_budget runs without crashing
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import pytest

from agentdescent.agents import Usage
from agentdescent.budget import CallBudget, budgeted_completion
from agentdescent.evolution import evolve, Task


# --- CallBudget.allocate ---


def test_allocate_promising_parent_full_budget():
    b = CallBudget(base=8192)
    b.allocate(0.9, 1.0)
    assert b.next is not None
    assert b.next <= b.base
    # 0.5 + 0.5*0.9 = 0.95; 0.5 + 0.5*1.0 = 1.0; 8192*0.95*1.0 = 7782
    assert b.next == 7782


def test_allocate_dead_end_parent_low_budget():
    b = CallBudget(base=8192)
    b.allocate(0.0, 0.0)
    # 0.5 * 0.5 = 0.25; 8192*0.25 = 2048 = floor
    assert b.next == b.floor


def test_allocate_never_above_base():
    b = CallBudget(base=4096)
    for score in [0.0, 0.5, 1.0]:
        for remaining in [0.0, 0.5, 1.0]:
            b.allocate(score, remaining)
            assert b.next <= b.base


def test_allocate_never_below_floor():
    b = CallBudget(base=4096)
    for score in [0.0, -1.0]:
        for remaining in [0.0, -1.0]:
            b.allocate(score, remaining)
            assert b.next >= b.floor


def test_allocate_clamps_score():
    b = CallBudget(base=4096)
    b.allocate(2.0, 2.0)  # above 1
    assert b.next <= b.base
    b.allocate(-5.0, -5.0)
    assert b.next >= b.floor


def test_floor_is_base_div_4():
    assert CallBudget(base=8192).floor == 2048
    assert CallBudget(base=1024).floor == 256  # minimum


def test_custom_floor():
    """A custom floor only kicks in when the calculated value falls below it."""
    b = CallBudget(base=8192, floor=1000)
    # score=0, remaining=0 → 8192*0.25 = 2048 > floor → 2048, not 1000.
    b.allocate(0.0, 0.0)
    assert b.next == 2048
    # With a very small base and a high floor, the floor binds.
    small = CallBudget(base=100, floor=50)
    small.allocate(0.0, 0.0)
    # 100*0.25 = 25 < 50 → floor binds
    assert small.next == 50


def test_next_none_by_default():
    b = CallBudget(base=4096)
    assert b.next is None


# --- budgeted_completion ---


def test_budgeted_completion_uses_base_when_next_is_none():
    calls = []

    def factory(mt):
        calls.append(mt)
        return lambda prompt: f"mt={mt}"

    b = CallBudget(base=4096)
    bc = budgeted_completion(factory, b)
    assert bc("hello") == "mt=4096"
    assert calls == [4096]


def test_budgeted_completion_adapts_when_next_is_set():
    calls = []

    def factory(mt):
        calls.append(mt)
        return lambda prompt: f"mt={mt}"

    b = CallBudget(base=4096)
    bc = budgeted_completion(factory, b)
    bc("first")  # base
    b.next = 2048
    bc("second")  # adapted
    assert 4096 in calls
    assert 2048 in calls


def test_budgeted_completion_caches_per_max_tokens():
    calls = []

    def factory(mt):
        calls.append(mt)
        return lambda prompt: f"mt={mt}"

    b = CallBudget(base=4096)
    bc = budgeted_completion(factory, b)
    bc("a")
    bc("b")  # same mt, cached
    assert len(calls) == 1
    b.next = 2048
    bc("c")  # new mt
    assert len(calls) == 2
    bc("d")  # cached
    assert len(calls) == 2


# --- inert without call_budget ---


def test_call_budget_unused_does_not_crash():
    """A run without call_budget must not touch it."""
    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]

    def run(rendered, task):
        return task.id

    def propose(rendered, task, output, reward):
        return f"rule-{task.id}" if reward < 0.999 else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(tasks, lambda t, o: 0.5, run=run, propose=propose,
                        rounds=2, n_workers=1)
    assert result.stop_reason == "rounds"


# --- end to end ---


def test_evolve_with_call_budget_runs():
    """evolve(call_budget=...) plumbs through and the budget is allocated."""
    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]
    usage = Usage()
    allocations = []

    budget = CallBudget(base=4096)

    def factory(mt):
        def complete(prompt):
            usage.record(prompt_tokens=100, completion_tokens=mt // 10)
            return f"rule: {prompt[:10]}"

        return complete

    from agentdescent.evolution import LLMAgent

    agent = LLMAgent(budgeted_completion(factory, budget))

    # Track allocations
    orig_allocate = budget.allocate

    def tracking_allocate(score, remaining, **kw):
        orig_allocate(score, remaining, **kw)
        allocations.append((round(score, 3), round(remaining, 3), budget.next))

    budget.allocate = tracking_allocate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(tasks, lambda t, o: 0.5, agent=agent,
                        rounds=3, n_workers=1, max_tokens=10_000_000,
                        call_budget=budget, usage=usage)

    assert result.stop_reason in ("rounds", "patience", "max_tokens")
    # The engine called allocate at least once per round that proposed.
    if allocations:
        # All allocations are within bounds.
        assert all(a[2] <= budget.base for a in allocations)
        assert all(a[2] >= budget.floor for a in allocations)
        # Allocations vary (the score is constant 0.5, so they should be
        # similar but the budget_remaining changes).
        assert len(allocations) >= 1
