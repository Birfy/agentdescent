"""How much more compute buys how much more improvement?

A self-improvement budget is a bet: another round, another worker, another
meta-session. The bet is worth making when the measured curve says the next
unit of compute still moves the score, and it is a mistake when the curve has
flattened. Scaling laws make that judgment explicit:

    score(budget) = a - b * budget ** (-c)

``a`` is the asymptote (the task's ceiling), ``b`` the initial gap, and ``c``
how quickly the gap closes. Fitting is deterministic: for each exponent in a
fixed grid, ordinary least squares on ``budget ** -c`` gives ``a`` and ``b``;
the exponent with the smallest residual wins, then a local ternary refinement
sharpens an off-grid exponent. No scipy, no random restarts.

    report = fit_power_law(points)
    print(report.fit.a, report.fit.c, report.r2)
    plan = plan_budget(points, target=0.9, max_budget=10_000)
    print(plan.budget, plan.achievable, plan.reliable)

Scores are assumed **higher-is-better**; negate a loss before fitting. Fits whose
OLS asymptote falls at or below an observed score are skipped for that exponent,
so a noisy plateau can end in "no saturating power law fits". The fit is only
descriptive: validate on a held-out budget before it justifies a large spend,
and ``plan_budget`` refuses to extrapolate from a ``weak``/``poor`` fit unless
the caller explicitly asks for it. A target already reached by an observation is
returned as "already reached" without extrapolating the law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ScalingPoint",
    "PowerLaw",
    "ScalingReport",
    "BudgetPlan",
    "fit_power_law",
    "marginal_return",
    "plan_budget",
]

_DEFAULT_GRID: Tuple[float, ...] = tuple(i / 20 for i in range(1, 61))


@dataclass(frozen=True)
class ScalingPoint:
    """One measured (compute budget, score) observation."""

    budget: float
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.budget)) or float(self.budget) <= 0:
            raise ValueError("ScalingPoint.budget must be finite and > 0")
        if not math.isfinite(float(self.score)):
            raise ValueError("ScalingPoint.score must be finite")


@dataclass(frozen=True)
class PowerLaw:
    """``score = a - b * budget ** (-c)``."""

    a: float
    b: float
    c: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise ValueError(f"PowerLaw.{name} must be finite")
        if self.b <= 0:
            raise ValueError("PowerLaw.b must be > 0 (the curve must rise)")
        if self.c <= 0:
            raise ValueError("PowerLaw.c must be > 0")

    def predict(self, budget: float) -> float:
        value = float(budget)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("budget must be finite and > 0")
        return self.a - self.b * value ** (-self.c)

    def marginal(self, budget: float) -> float:
        """Derivative ``d score / d budget`` (positive, decreasing)."""
        value = float(budget)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("budget must be finite and > 0")
        return self.b * self.c * value ** (-self.c - 1.0)


@dataclass(frozen=True)
class ScalingReport:
    """A fitted law plus the diagnostics needed to distrust it."""

    fit: PowerLaw
    n: int
    r2: float
    rmse: float
    verdict: str
    detail: str = ""

    @property
    def reliable(self) -> bool:
        return self.verdict in ("good", "acceptable")


@dataclass(frozen=True)
class BudgetPlan:
    """The compute needed to reach a target, or why it cannot be reached."""

    target: float
    budget: Optional[float]
    predicted: Optional[float]
    achievable: bool
    detail: str = ""
    fit: Optional["ScalingReport"] = None
    reliable: bool = False


def _points(points: Iterable[ScalingPoint]) -> List[ScalingPoint]:
    rows = list(points)
    if len(rows) < 3:
        raise ValueError("fit_power_law needs at least three points")
    if len({row.budget for row in rows}) != len(rows):
        raise ValueError("budget values must be unique")
    return sorted(rows, key=lambda row: row.budget)


def _fit_for_c(rows: Sequence[ScalingPoint], c: float) -> Optional[Tuple[float, float, float]]:
    xs = [row.budget ** (-c) for row in rows]
    ys = [row.score for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    # y = intercept + slope * x, and x = budget**-c, so a=intercept, b=-slope.
    a, b = intercept, -slope
    if b <= 0 or a <= max(ys):
        # A curve whose asymptote is below an observed score cannot be the
        # saturating law we are fitting; reject rather than report nonsense.
        return None
    sse = sum((y - (a - b * x)) ** 2 for x, y in zip(xs, ys))
    if not math.isfinite(sse):
        return None
    return a, b, sse


def fit_power_law(points: Iterable[ScalingPoint], *,
                  c_grid: Optional[Sequence[float]] = None,
                  refine: bool = True) -> ScalingReport:
    """Fit the saturating power law by deterministic grid + least squares."""
    rows = _points(points)
    grid = _DEFAULT_GRID if c_grid is None else tuple(float(c) for c in c_grid)
    if not grid:
        raise ValueError("c_grid must not be empty")
    best = None
    for c in grid:
        if not math.isfinite(c) or c <= 0:
            raise ValueError("c_grid values must be finite and > 0")
        candidate = _fit_for_c(rows, c)
        if candidate is None:
            continue
        a, b, sse = candidate
        if best is None or sse < best[0]:
            best = (sse, a, b, c)
    if best is None:
        raise ValueError("no saturating power law fits these points")
    if refine:
        step = (abs(grid[1] - grid[0]) if len(grid) > 1 else 0.05)
        step = max(0.005, min(0.2, step))
        lo = max(1e-6, best[3] - step)
        hi = best[3] + step

        def sse_at(value: float) -> float:
            candidate = _fit_for_c(rows, value)
            return float("inf") if candidate is None else candidate[2]

        for _ in range(60):
            third = (hi - lo) / 3.0
            mid1, mid2 = lo + third, hi - third
            if sse_at(mid1) < sse_at(mid2):
                hi = mid2
            else:
                lo = mid1
        for candidate_c in (lo, (lo + hi) / 2.0, hi):
            candidate = _fit_for_c(rows, candidate_c)
            if candidate is not None and candidate[2] < best[0]:
                best = (candidate[2], candidate[0], candidate[1], candidate_c)
    sse, a, b, c = best
    fit = PowerLaw(a=a, b=b, c=c)
    mean_y = sum(row.score for row in rows) / len(rows)
    total = sum((row.score - mean_y) ** 2 for row in rows)
    r2 = 1.0 - sse / total if total > 0 else 1.0
    rmse = math.sqrt(sse / len(rows))
    if r2 >= 0.98:
        verdict = "good"
    elif r2 >= 0.9:
        verdict = "acceptable"
    elif r2 >= 0.7:
        verdict = "weak"
    else:
        verdict = "poor"
    detail = (f"n={len(rows)}, a={a:.6g}, b={b:.6g}, c={c:.4g}, "
              f"r2={r2:.4f}, rmse={rmse:.6g}")
    return ScalingReport(fit=fit, n=len(rows), r2=r2, rmse=rmse,
                         verdict=verdict, detail=detail)


def marginal_return(fit: PowerLaw, budget: float) -> float:
    """Convenience wrapper around ``PowerLaw.marginal``."""
    return fit.marginal(budget)


def plan_budget(points: Iterable[ScalingPoint], *, target: float,
                max_budget: float, min_budget: float = 1.0,
                require_reliable: bool = True,
                c_grid: Optional[Sequence[float]] = None) -> BudgetPlan:
    """Find the smallest budget whose fitted score reaches ``target``.

    A target already met by an observed point returns that point's budget and
    does not extrapolate. Otherwise the fit is inverted; with the default
    ``require_reliable=True`` a ``weak``/``poor`` fit refuses to plan. Returns
    ``achievable=False`` when the target is above the fitted asymptote, the fit
    is unreliable, or the needed budget exceeds ``max_budget``. ``min_budget``
    is the smallest budget the caller will actually run; the result is never
    below it.
    """
    target = float(target)
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    if not math.isfinite(float(max_budget)) or max_budget <= 0:
        raise ValueError("max_budget must be finite and > 0")
    if not math.isfinite(float(min_budget)) or min_budget <= 0:
        raise ValueError("min_budget must be finite and > 0")
    if min_budget > max_budget:
        raise ValueError("min_budget must be <= max_budget")
    rows = _points(points)
    report = fit_power_law(rows, c_grid=c_grid)
    fit = report.fit
    reached = [row for row in rows if row.score >= target]
    if reached:
        budget = max(float(min_budget), reached[0].budget)
        return BudgetPlan(target=target, budget=budget,
                          predicted=fit.predict(budget), achievable=True,
                          detail=(f"target already reached at observed budget "
                                  f"{reached[0].budget:g}"),
                          fit=report, reliable=report.reliable)
    if require_reliable and not report.reliable:
        return BudgetPlan(target=target, budget=None, predicted=None,
                          achievable=False,
                          detail=(f"fit is {report.verdict} (r2={report.r2:.3f}); "
                                  f"pass require_reliable=False to extrapolate"),
                          fit=report, reliable=False)
    if target >= fit.a:
        return BudgetPlan(target=target, budget=None, predicted=None,
                          achievable=False,
                          detail=f"target >= fitted asymptote {fit.a:.6g}",
                          fit=report, reliable=report.reliable)
    need = (fit.b / (fit.a - target)) ** (1.0 / fit.c)
    need = max(float(min_budget), float(need))
    if need > max_budget:
        return BudgetPlan(target=target, budget=None,
                          predicted=fit.predict(max_budget), achievable=False,
                          detail=f"needs {need:.6g} > max_budget {max_budget:g}",
                          fit=report, reliable=report.reliable)
    predicted = fit.predict(need)
    return BudgetPlan(target=target, budget=need, predicted=predicted,
                      achievable=True,
                      detail=f"budget {need:.6g} -> {predicted:.6g}",
                      fit=report, reliable=report.reliable)
