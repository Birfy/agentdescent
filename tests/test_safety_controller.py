"""Deployment safety must be explicit: kill, canary, rollback."""

from __future__ import annotations

import pytest

from agentdescent import CanaryResult, KillSwitch, canary, rollback_plan


def test_the_kill_switch_is_file_based_and_reports_its_reason(tmp_path):
    switch = KillSwitch(str(tmp_path / "STOP"))
    assert not switch.engaged() and switch.reason() == ""
    switch.engage("cost spike")
    assert switch.engaged() and switch.reason() == "cost spike"
    switch.release()
    assert not switch.engaged()


def test_a_significant_gain_promotes():
    inc = {str(i): 0.2 for i in range(20)}
    cand = {str(i): 0.8 for i in range(20)}
    result = canary(inc, cand)
    assert isinstance(result, CanaryResult)
    assert result.verdict == "promote" and result.mean_gain == pytest.approx(0.6)


def test_a_significant_loss_rolls_back():
    inc = {str(i): 0.8 for i in range(20)}
    cand = {str(i): 0.2 for i in range(20)}
    result = canary(inc, cand)
    assert result.verdict == "rollback" and result.mean_gain == pytest.approx(-0.6)


def test_a_mixed_or_small_effect_holds():
    inc = {str(i): 0.5 for i in range(10)}
    cand = dict(inc)
    cand["0"] = 0.55
    assert canary(inc, cand).verdict == "hold"
    assert canary({}, {}).verdict == "hold"


def test_min_effect_is_required_to_promote():
    inc = {str(i): 0.0 for i in range(10)}
    cand = {str(i): 0.1 for i in range(10)}
    assert canary(inc, cand).verdict == "promote"
    assert canary(inc, cand, min_effect=0.2).verdict == "hold"


def test_rollback_plan_is_a_pure_diff():
    plan = rollback_plan({"a": "now", "b": "new", "c": "gone"},
                         {"a": "old", "b": "new"})
    assert plan["write"] == {"a": "old"}
    assert plan["delete"] == ("c",)
    assert plan["noop"] is False
    delete_only = rollback_plan({"a": "x", "extra": "y"}, {"a": "x"})
    assert delete_only["noop"] is False
    assert rollback_plan({"a": "x"}, {"a": "x"})["noop"] is True


def test_missing_candidate_tasks_block_promotion():
    result = canary({"a": 0.9, "b": 0.9, "c": 0.9}, {"a": 1.0})
    assert result.verdict == "hold" and result.missing == ("b", "c")
    assert "missing" in result.reason


def test_a_significant_loss_smaller_than_min_effect_rolls_back():
    inc = {str(i): 0.8 for i in range(30)}
    cand = {str(i): 0.79 for i in range(30)}
    result = canary(inc, cand, min_effect=0.05, max_regression=0.0)
    assert result.verdict == "rollback" and result.mean_gain < 0


def test_a_directory_at_the_switch_path_still_stops(tmp_path):
    switch = KillSwitch(str(tmp_path / "STOP"))
    (tmp_path / "STOP").mkdir()
    assert switch.engaged()
    switch.release()
    assert not switch.engaged()


def test_invalid_alpha_or_scores_are_refused():
    with pytest.raises(ValueError, match="alpha"):
        canary({"a": 1}, {"a": 1}, alpha=1.5)
    for value in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_effect"):
            canary({"a": 1}, {"a": 1}, min_effect=value)
    with pytest.raises(ValueError, match="finite"):
        canary({"a": float("nan")}, {"a": 1.0})
