"""The matrix report's refusals, which are the only reason to have one.

Rendering a markdown table is not worth a test. Refusing to render a speedup
that is not one is, and each test here pins a way the table could otherwise
publish a number that means something other than what it says.
"""
from __future__ import annotations

import pytest

from bench import matrix_report
from bench.matrix_run import ROWS, SCHEDULING_ONLY, SEMANTICS


ARMS = ("serial", "parallel", "async")


def _cell(row="gepa", arm="serial", seed=0, wall=100.0, test=0.5, **extra):
    return dict(row=row, dataset="HotpotQA", arm=arm, seed=seed, width=8,
                budget=16, wall_seconds=wall, test=test, **extra)


def _payload(cells, **head):
    return dict(budget_rollouts=16, width=8, async_ratio=3, model="m",
                provider="p", cells=list(cells), **head)


def test_every_row_declares_what_each_arm_changed():
    """A row in ROWS without a SEMANTICS entry is a blank column, and a blank
    column is a claim that nothing changed made by omission."""
    for row in ROWS:
        assert row["name"] in SEMANTICS, row["name"]
        for arm in ARMS:
            assert SEMANTICS[row["name"]].get(arm), (row["name"], arm)


def test_a_row_missing_from_semantics_is_refused_not_rendered_blank():
    saved = SEMANTICS.pop("gepa")
    try:
        with pytest.raises(SystemExit, match="SEMANTICS"):
            matrix_report.render(_payload([_cell()]))
    finally:
        SEMANTICS["gepa"] = saved


def test_the_evoskill_async_arm_is_recorded_as_an_algorithm_change():
    """The audit's one finding. If this ever reverts to the clean string, either
    the port stopped swapping its aggregator or somebody papered over it."""
    assert SEMANTICS["evoskill"]["async"] != SCHEDULING_ONLY
    assert "admission rule changed" in SEMANTICS["evoskill"]["async"]
    assert SEMANTICS["evoskill"]["parallel"] == SCHEDULING_ONLY


def test_speedup_is_withheld_when_the_arms_did_not_spend_the_same_budget():
    """The pin is the comparison. Checked against what the cells measured, not
    trusted because a flag was passed."""
    cells = [_cell(arm="serial", wall=200.0, engine_rollouts=16),
             _cell(arm="parallel", wall=50.0, engine_rollouts=64)]
    text = matrix_report.render(_payload(cells))
    assert "withheld" in text
    assert "unequal budget" in text
    assert "4.00×" not in text


def test_the_barrier_overshoot_is_not_treated_as_an_unequal_budget():
    """The synchronous path checks at the round barrier, so an N-worker arm may
    legitimately overshoot by up to N-1 -- refusing that would refuse every
    honest sync cell."""
    cells = [_cell(arm="serial", wall=200.0, engine_rollouts=16),
             _cell(arm="parallel", wall=100.0, engine_rollouts=23)]
    text = matrix_report.render(_payload(cells))
    assert "unequal budget" not in text
    assert "2.00×" in text


def test_speedup_is_net_of_time_lost_in_failed_calls():
    """A retry storm in one arm is endpoint weather, not the other arm's
    scheduler. 300-100 against 100 is 2x, not 3x."""
    cells = [_cell(arm="serial", wall=300.0, failure_seconds=100.0),
             _cell(arm="parallel", wall=100.0)]
    assert "2.00×" in matrix_report.render(_payload(cells))


def test_a_single_seed_row_is_daggered_and_called_out():
    text = matrix_report.render(_payload([_cell(), _cell(arm="parallel")]))
    assert "gepa †" in text
    assert "below the 3" in text


def test_three_seeds_report_a_range_and_lose_the_dagger():
    cells = [_cell(arm=arm, seed=s, wall=100.0 + 10 * s, test=0.5 + 0.1 * s)
             for arm in ("serial", "parallel") for s in range(3)]
    text = matrix_report.render(_payload(cells))
    assert "gepa †" not in text
    assert "[100–120]" in text


def test_the_async_stale_rate_never_appears_without_its_denominator():
    """A bare discard count cannot be read at all -- the mistake this codebase
    already made once on the async path."""
    no_data = matrix_report.render(_payload([_cell(), _cell(arm="async")]))
    assert "not recorded" in no_data

    cells = [_cell(), _cell(arm="async", stale_considered=40, stale_discarded=10)]
    assert "10/40 (25%)" in matrix_report.render(_payload(cells))


def test_failed_cells_do_not_become_measurements():
    """A cell with an error has no wall-clock worth reporting; it must not be
    averaged into the arm that failed."""
    cells = [_cell(arm="serial", wall=200.0),
             _cell(arm="parallel", wall=100.0),
             dict(row="gepa", arm="parallel", seed=1, error="boom")]
    text = matrix_report.render(_payload(cells))
    assert "2.00×" in text


def test_an_unmeasured_row_renders_empty_rather_than_inventing_a_dash_speedup():
    text = matrix_report.render(_payload([_cell()]))
    for line in text.splitlines():
        if line.startswith("| ace "):
            assert line.count("—") == 6
            break
    else:                                     # pragma: no cover - guard
        pytest.fail("the ace row was not rendered")
