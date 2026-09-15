"""Tests for the budget governor: graceful degradation before the token wall.

Covers:
- Hard stop: the wall fires at 100%
- Fusion tournament degrades at the soft floor (75%)
- Self-verify degrades at the hard floor (90%)
- Projection: affords_next_round predicts correctly
- Inert without a budget
- Governor.summary() reports the right fields
"""

from __future__ import annotations

import pytest

from agentdescent.budget import BudgetGovernor, SOFT_FLOOR, HARD_FLOOR


# --- basic thresholds ---


def test_governor_inert_without_budget():
    g = BudgetGovernor(max_tokens=None)
    assert not g.active
    assert not g.over_budget()
    assert g.allow_fusion_tournament()
    assert g.allow_self_verify()
    assert g.affords_next_round()
    s = g.summary()
    assert s["max_tokens"] is None
    assert s["remaining"] is None
    assert s["fusion_degraded"] is False
    assert s["self_verify_degraded"] is False


def test_hard_stop_at_budget():
    g = BudgetGovernor(max_tokens=1000)
    g.spend(999)
    assert not g.over_budget()
    g.spend(1000)
    assert g.over_budget()
    g.spend(1200)
    assert g.over_budget()


def test_fusion_degrades_at_soft_floor():
    g = BudgetGovernor(max_tokens=1000)
    g.spend(740)
    assert g.allow_fusion_tournament()
    g.spend(750)
    assert not g.allow_fusion_tournament()  # at 75%
    g.spend(800)
    assert not g.allow_fusion_tournament()


def test_self_verify_degrades_at_hard_floor():
    g = BudgetGovernor(max_tokens=1000)
    g.spend(890)
    assert g.allow_self_verify()
    g.spend(900)
    assert not g.allow_self_verify()  # at 90%
    g.spend(950)
    assert not g.allow_self_verify()


def test_fusion_degrades_before_self_verify():
    g = BudgetGovernor(max_tokens=1000)
    g.spend(760)  # past soft floor, before hard floor
    assert not g.allow_fusion_tournament()
    assert g.allow_self_verify()


# --- projection ---


def test_affords_next_round_with_history():
    g = BudgetGovernor(max_tokens=3000)
    # Three rounds at 500 each = 1500 spent. Remaining 1500 / avg 500 = 3 rounds.
    g.spend(500)
    g.spend(1000)
    g.spend(1500)
    assert g.affords_next_round()
    # Now simulate a huge round that leaves only 100 tokens.
    g.spend(2900)
    assert not g.affords_next_round()


def test_affords_next_round_no_history():
    g = BudgetGovernor(max_tokens=1000)
    # No spend call yet — always afford (the run has not started).
    assert g.affords_next_round()


def test_affords_next_round_zero_spend():
    g = BudgetGovernor(max_tokens=1000)
    g.spend(0)  # a run with no token reporting
    assert g.affords_next_round()  # can't project, so afford


def test_affords_next_round_uneven_spend():
    g = BudgetGovernor(max_tokens=2000)
    g.spend(100)
    g.spend(500)  # delta 400
    g.spend(1500)  # delta 1000
    # avg delta = (100+400+1000)/3 = 500
    # remaining = 2000 - 1500 = 500
    # 500 >= 500 → exactly affords
    assert g.affords_next_round()
    # one more round at avg 500 → 2000 spent, 0 left
    g.spend(2000)
    assert not g.affords_next_round()


# --- summary ---


def test_summary_fields():
    g = BudgetGovernor(max_tokens=1000, soft_floor=0.5, hard_floor=0.8)
    g.spend(600)
    s = g.summary()
    assert s["max_tokens"] == 1000
    assert s["spent"] == 600
    assert s["remaining"] == 400
    assert s["fusion_degraded"] is True  # 600 > 500
    assert s["self_verify_degraded"] is False  # 600 < 800


def test_summary_all_none_without_budget():
    g = BudgetGovernor()
    s = g.summary()
    assert s["max_tokens"] is None
    assert s["spent"] == 0
    assert s["remaining"] is None


# --- custom floors ---


def test_custom_floors():
    g = BudgetGovernor(max_tokens=1000, soft_floor=0.5, hard_floor=0.8)
    g.spend(400)
    assert g.allow_fusion_tournament()  # 40% < 50%
    assert g.allow_self_verify()
    g.spend(510)
    assert not g.allow_fusion_tournament()  # 51% > 50%
    assert g.allow_self_verify()  # 51% < 80%
    g.spend(810)
    assert not g.allow_self_verify()  # 81% > 80%
