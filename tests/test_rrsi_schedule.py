"""Deterministic fixtures for the RRSI cosine-annealed edit budget."""

from __future__ import annotations

import math

import pytest

from examples.rrsi.schedule import budget_table, edit_budget


def test_endpoints_match_paper_formula():
    """t=0 is b_max; t=T lands on b_min after the float guard."""
    T, b_min, b_max = 10, 1, 5
    assert edit_budget(0, T, b_min, b_max) == b_max
    assert edit_budget(T, T, b_min, b_max) == b_min


def test_budget_is_nonincreasing_when_annealing_down():
    table = budget_table(12, 1, 8)
    assert table[0] == 8
    assert table[-1] >= 1
    assert all(table[i] >= table[i + 1] for i in range(len(table) - 1))


def test_budget_table_length_and_values_match_edit_budget():
    T, b_min, b_max = 7, 2, 6
    table = budget_table(T, b_min, b_max)
    assert len(table) == T
    assert table == [edit_budget(t, T, b_min, b_max) for t in range(T)]


def test_float_guard_keeps_t_equals_T_at_b_min():
    """Without round(..., 9), cos(pi) noise can ceil above b_min."""
    for T in (1, 2, 3, 8, 16, 64):
        for b_min, b_max in ((1, 1), (1, 4), (3, 9)):
            assert edit_budget(T, T, b_min, b_max) == b_min


def test_degenerate_horizon_returns_b_max():
    assert edit_budget(0, 0, 1, 4) == 4
    assert edit_budget(99, -1, 2, 5) == 5


def test_t_is_clamped_into_closed_interval():
    T, b_min, b_max = 5, 1, 5
    assert edit_budget(-3, T, b_min, b_max) == edit_budget(0, T, b_min, b_max)
    assert edit_budget(100, T, b_min, b_max) == edit_budget(T, T, b_min, b_max)


def test_constant_budget_when_b_min_equals_b_max():
    table = budget_table(5, 3, 3)
    assert table == [3, 3, 3, 3, 3]


def test_matches_direct_formula_on_interior_points():
    T, b_min, b_max = 8, 1, 5
    for t in range(T + 1):
        raw = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))
        assert edit_budget(t, T, b_min, b_max) == int(math.ceil(round(raw, 9)))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"t": 0, "T": 4, "b_min": 0, "b_max": 3}, ">= 1"),
        ({"t": 0, "T": 4, "b_min": 2, "b_max": 1}, "<="),
        ({"t": 0.5, "T": 4, "b_min": 1, "b_max": 3}, "t must be an int"),
        ({"t": 0, "T": 4.0, "b_min": 1, "b_max": 3}, "T must be an int"),
        ({"t": 0, "T": 4, "b_min": True, "b_max": 3}, "b_min must be an int"),
        ({"t": 0, "T": -1, "b_min": 1, "b_max": 3}, None),  # T<=0 is allowed for edit_budget
    ],
)
def test_rejects_invalid_bounds(kwargs, match):
    if match is None:
        assert edit_budget(**kwargs) == kwargs["b_max"]
        return
    with pytest.raises(ValueError, match=match):
        edit_budget(**kwargs)


def test_budget_table_rejects_negative_T():
    with pytest.raises(ValueError, match="T must be >= 0"):
        budget_table(-1, 1, 3)
    assert budget_table(0, 1, 3) == []


@pytest.mark.parametrize(
    "b_min,b_max,match",
    [
        (0, 0, ">= 1"),
        (3, 1, "<="),
        (True, 3, "b_min must be an int"),
        (1, 2.0, "b_max must be an int"),
    ],
)
def test_budget_table_validates_bounds_even_for_zero_rounds(b_min, b_max, match):
    # T == 0 yields an empty table, but invalid bounds must still be refused
    # rather than silently returning [].
    with pytest.raises(ValueError, match=match):
        budget_table(0, b_min, b_max)
