"""The scaling planner must recover a curve it can see and refuse to invent one."""

import math

import pytest

from agentdescent.scaling import (
    BudgetPlan,
    PowerLaw,
    ScalingPoint,
    fit_power_law,
    marginal_return,
    plan_budget,
)


def _points(a=1.0, b=2.0, c=0.6, budgets=(1, 2, 4, 8, 16, 32)):
    return [ScalingPoint(x, a - b * x ** (-c)) for x in budgets]


def test_exact_curve_is_recovered():
    report = fit_power_law(_points())
    assert report.fit.a == pytest.approx(1.0, abs=1e-6)
    assert report.fit.b == pytest.approx(2.0, abs=1e-6)
    assert report.fit.c == pytest.approx(0.6, abs=1e-6)
    assert report.r2 == pytest.approx(1.0, abs=1e-9)
    assert report.verdict == "good"


def test_predict_and_marginal_are_monotone_and_positive():
    fit = PowerLaw(a=1.0, b=2.0, c=0.6)
    assert fit.predict(1) < fit.predict(10) < fit.predict(100)
    assert fit.predict(100) < fit.a
    assert marginal_return(fit, 1) > marginal_return(fit, 100) > 0


def test_target_plan_solves_the_inverse_curve():
    plan = plan_budget(_points(), target=0.9, max_budget=1000)
    assert plan.achievable
    assert plan.budget == pytest.approx(147.3612599456, rel=1e-6)
    assert plan.predicted == pytest.approx(0.9, abs=1e-6)
    assert isinstance(plan, BudgetPlan)


def test_impossible_targets_are_named_not_clamped():
    above = plan_budget(_points(), target=1.1, max_budget=10_000)
    assert not above.achievable and above.budget is None
    too_much = plan_budget(_points(), target=0.9999, max_budget=100)
    assert not too_much.achievable
    assert "max_budget" in too_much.detail


def test_min_budget_is_respected():
    plan = plan_budget(_points(), target=0.1, max_budget=1000, min_budget=10)
    assert plan.budget == pytest.approx(10)


def test_a_flat_or_noisy_curve_is_not_called_good():
    points = [ScalingPoint(1, 0.5), ScalingPoint(2, 0.9), ScalingPoint(4, 0.2),
              ScalingPoint(8, 0.9)]
    report = fit_power_law(points)
    assert report.verdict in ("weak", "poor")
    assert report.r2 < 0.9


def test_too_few_or_duplicate_budgets_are_refused():
    with pytest.raises(ValueError):
        fit_power_law([ScalingPoint(1, 0.1), ScalingPoint(2, 0.2)])
    with pytest.raises(ValueError):
        fit_power_law([ScalingPoint(1, 0.1), ScalingPoint(1, 0.2), ScalingPoint(2, 0.3)])
    with pytest.raises(ValueError):
        fit_power_law(_points(), c_grid=[])


def test_invalid_points_are_refused():
    for budget, score in [(0, 0.1), (-1, 0.1), (1, math.nan)]:
        with pytest.raises(ValueError):
            ScalingPoint(budget, score)


def test_invalid_law_and_plan_arguments_are_refused():
    with pytest.raises(ValueError):
        PowerLaw(a=1, b=0, c=1)
    with pytest.raises(ValueError):
        PowerLaw(a=1, b=1, c=0)
    with pytest.raises(ValueError):
        plan_budget(_points(), target=0.5, max_budget=0)
    with pytest.raises(ValueError):
        plan_budget(_points(), target=0.5, max_budget=10, min_budget=11)
    with pytest.raises(ValueError):
        plan_budget(_points(), target=math.nan, max_budget=10)


def test_plan_refuses_an_unreliable_fit_unless_asked():
    points = [ScalingPoint(1, 0.5), ScalingPoint(2, 0.9),
              ScalingPoint(4, 0.2), ScalingPoint(8, 0.9)]
    refused = plan_budget(points, target=0.95, max_budget=1e6)
    assert not refused.achievable
    assert refused.fit is not None and refused.fit.verdict == "poor"
    assert "require_reliable=False" in refused.detail
    allowed = plan_budget(points, target=0.95, max_budget=1e6,
                          require_reliable=False)
    assert allowed.achievable
    assert not allowed.reliable


def test_target_already_observed_is_not_extrapolated():
    points = _points()
    plan = plan_budget(points, target=0.5, max_budget=1000, min_budget=1)
    assert plan.achievable
    assert plan.budget == pytest.approx(16.0)
    assert "already reached" in plan.detail


def test_off_grid_exponent_is_refined_and_plans_can_reuse_the_grid():
    points = _points(c=0.62)
    report = fit_power_law(points)
    assert report.fit.c == pytest.approx(0.62, abs=0.02)
    assert report.fit.a == pytest.approx(1.0, abs=0.05)
    plan = plan_budget(points, target=0.9, max_budget=1000,
                       c_grid=[0.62])
    assert plan.achievable
    assert plan.fit.fit.c == pytest.approx(0.62, abs=0.02)
