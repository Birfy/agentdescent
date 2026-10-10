import math
from dataclasses import replace

import pytest

from examples.rrsi.selection import (
    Candidate,
    Measurement,
    SelectionConfig,
    judge,
    select_round,
    update_best_score,
)


CFG = SelectionConfig(beta0=0.1, beta1=1.0, w_s=0.0, w_c=1.0, w_n=0.5)
INCUMBENT = Measurement(score=0.70, cost=100.0)


def candidate(name, score, cost=100.0, **kwargs):
    return Candidate(name, measurement=Measurement(score, cost), **kwargs)


def test_historical_floor_is_best_minus_delta_not_incumbent_plus_delta():
    cand = candidate("cheaper", 0.68, 50.0)
    decision = judge(cand, INCUMBENT, best_score=0.72, delta=0.05,
                     config=CFG, incumbent_counts={})
    assert decision.admissible
    assert decision.delta_score == pytest.approx(-0.02)


def test_large_gain_uses_linear_cost_budget():
    allowed = judge(candidate("allowed", 0.90, 115.0), INCUMBENT, 0.70, 0.05,
                    CFG, {})
    denied = judge(candidate("denied", 0.90, 140.0), INCUMBENT, 0.70, 0.05,
                   CFG, {})
    assert allowed.admissible
    assert not denied.admissible


def test_within_band_uses_cost_and_structural_novelty():
    fresh = Candidate("fresh", components=("skill",),
                      measurement=Measurement(0.70, 100.0))
    no_saving = judge(fresh, INCUMBENT, 0.70, 0.05, CFG, {"skill": 0})
    costly = judge(Candidate("costly", components=("skill",),
                             measurement=Measurement(0.70, 160.0)),
                   INCUMBENT, 0.70, 0.05, CFG, {"skill": 0})
    assert no_saving.admissible and no_saving.novelty == 1
    assert not costly.admissible


@pytest.mark.parametrize("score,cost", [
    (math.nan, 100), (math.inf, 100), (-0.01, 100), (1.01, 100),
    (0.8, -1), (0.8, math.nan), (0.8, math.inf),
])
def test_invalid_measurements_are_rejected(score, cost):
    d = judge(candidate("bad", score, cost), INCUMBENT, 0.70, 0.05, CFG, {})
    assert not d.admissible
    assert "invalid" in d.reason


def test_invalid_historical_score_and_overflowing_cost_ratio_are_rejected():
    assert not judge(candidate("x", .8), INCUMBENT, math.nan, .05, CFG, {}).admissible
    tiny = Measurement(0.5, 5e-324)
    d = judge(candidate("overflow", .9, 1.7e308), tiny, .5, .05, CFG, {})
    assert not d.admissible
    assert d.reason == "invalid measurement arithmetic"


@pytest.mark.parametrize("field", ["beta0", "beta1", "w_s", "w_c", "w_n"])
def test_negative_selection_weights_are_rejected(field):
    config = replace(CFG, **{field: -0.1})
    d = judge(candidate("candidate", 0.9), INCUMBENT, 0.7, 0.05,
              config, {})
    assert not d.admissible
    assert d.reason == "invalid selection configuration"


def test_critic_reject_remains_rejected_even_if_measurement_is_attached():
    cand = Candidate("critic", measurement=Measurement(.99, 1),
                     gate_failure="critic_reject")
    d = judge(cand, INCUMBENT, .70, .05, CFG, {})
    assert not d.admissible
    assert d.reason == "critic_reject"


def test_domain_veto_and_winner_ties_are_deterministic():
    candidates = [candidate("first", .8), candidate("second", .8), candidate("guarded", .9)]
    winner, decisions = select_round(
        candidates, INCUMBENT, .7, .05, CFG, {},
        guard_fn=lambda _inc, ev: ["safety metric dropped"] if ev.score > .85 else [],
    )
    assert winner is candidates[0]
    assert not decisions[2].admissible
    assert "safety metric dropped" in decisions[2].reason


def test_no_admissible_candidate_and_missing_tokens_match_upstream_zero_rule():
    winner, decisions = select_round([candidate("too costly", .70, 200)],
                                     INCUMBENT, .70, .05, CFG, {})
    assert winner is None and not decisions[0].admissible

    unknown_cost = candidate("unknown telemetry", .90, None)
    d = judge(unknown_cost, INCUMBENT, .70, .05, CFG, {})
    assert d.admissible
    assert d.delta_cost == 0.0


def test_missing_measurement_and_gate_failure_do_not_get_fake_metrics():
    for c in (Candidate("screened", measurement=None, gate_failure="smoke_fail"),
              Candidate("not-run")):
        d = judge(c, INCUMBENT, .7, .05, CFG, {})
        assert not d.admissible
        assert d.score is None and d.cost is None


def test_best_score_is_monotone_and_no_winner_preserves_it():
    winner = candidate("winner", .75)
    assert update_best_score(.70, winner) == .75
    assert update_best_score(.75, candidate("lower", .70)) == .75
    assert update_best_score(.75, None) == .75


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1])
def test_update_best_rejects_invalid_history(value):
    with pytest.raises(ValueError):
        update_best_score(value, None)
