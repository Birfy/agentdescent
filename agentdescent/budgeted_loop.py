"""Run budgeted PRM/verifier search as one step of the durable RSI loop.

``SearchFabric`` turns a PRM and an optional verifier panel into UCT values;
``BudgetedSearch`` spends increasing budgets until extra expansions stop
paying. This module composes both into the loop kernel's step contract:

    fabric = SearchFabric(simulator, prm, features, verifier=panel)
    step = BudgetedStep(fabric, lambda context: context.snapshot.payload["state"])
    observation = step(context)
    observation.reward, observation.cost, observation.evidence["stopped_reason"]

The step reads only ``context.budget_scale`` and ``context.seed``, so any
object with those attributes works. When ``scale_budget`` is enabled the
controller's scale multiplies each configured budget, letting the outer loop
buy more or less search per generation. The observation carries the best
value, the spend, an optional risk callback's verdict, the artifact id and an
auditable round-by-round evidence map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .budgeted_search import BudgetedSearch, BudgetedSearchReport
from .loop_kernel import LoopObservation
from .search_fabric import SearchFabric, SearchResult

__all__ = ["BudgetedStepConfig", "BudgetedStep"]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class BudgetedStepConfig:
    """The budgets and stopping rule one search step spends through."""

    budgets: Tuple[int, ...] = (8, 16, 32)
    min_gain_per_expansion: float = 1e-4
    patience: int = 2
    artifact_prefix: str = "search"
    scale_budget: bool = True

    def __post_init__(self) -> None:
        try:
            planned = tuple(self.budgets)
        except TypeError:
            raise ValueError(
                "budgets must be a non-empty sequence of positive ints") from None
        if not planned or any(
                isinstance(budget, bool) or not isinstance(budget, int)
                or budget < 1 for budget in planned):
            raise ValueError(
                "budgets must be a non-empty sequence of positive ints")
        gain = _finite(self.min_gain_per_expansion, "min_gain_per_expansion")
        if gain < 0:
            raise ValueError("min_gain_per_expansion must be >= 0")
        if isinstance(self.patience, bool) or not isinstance(
                self.patience, int) or self.patience < 1:
            raise ValueError("patience must be an int >= 1")
        if not isinstance(self.artifact_prefix, str) or not self.artifact_prefix.strip():
            raise ValueError("artifact_prefix must be a non-empty string")
        if not isinstance(self.scale_budget, bool):
            raise ValueError("scale_budget must be a bool")
        object.__setattr__(self, "budgets", tuple(sorted(set(planned))))
        object.__setattr__(self, "min_gain_per_expansion", gain)


class BudgetedStep:
    """A ``LoopContext -> LoopObservation`` step over budgeted search."""

    def __init__(self, fabric: SearchFabric, state: Callable[[Any], Any], *,
                 config: Optional[BudgetedStepConfig] = None,
                 risk: Optional[Callable[[SearchResult], float]] = None,
                 evidence_of: Optional[
                     Callable[[SearchResult], Mapping[str, Any]]] = None) -> None:
        if not isinstance(fabric, SearchFabric):
            raise TypeError("fabric must be a SearchFabric")
        if not callable(state):
            raise TypeError("state must be callable")
        if config is None:
            config = BudgetedStepConfig()
        elif not isinstance(config, BudgetedStepConfig):
            raise TypeError("config must be a BudgetedStepConfig or None")
        if risk is not None and not callable(risk):
            raise TypeError("risk must be callable or None")
        if evidence_of is not None and not callable(evidence_of):
            raise TypeError("evidence_of must be callable or None")
        self.fabric = fabric
        self.state = state
        self.config = config
        self.risk = risk
        self.evidence_of = evidence_of

    def _budgets(self, budget_scale: Any) -> Tuple[int, ...]:
        config = self.config
        if not config.scale_budget:
            return tuple(config.budgets)
        scale = _finite(budget_scale, "budget_scale")
        if scale < 0:
            raise ValueError("budget_scale must be >= 0")
        scaled = tuple(max(1, int(round(budget * scale)))
                       for budget in config.budgets)
        return tuple(sorted(set(scaled)))

    def _risk(self, best: SearchResult) -> float:
        if self.risk is None:
            return 0.0
        value = _finite(self.risk(best), "risk")
        if not 0.0 <= value <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        return value

    def _rounds(self, report: BudgetedSearchReport) -> Tuple[Dict[str, Any], ...]:
        return tuple(
            {
                "budget": row.budget,
                "value": _finite(row.result.value, "round value"),
                "gain": row.gain,
                "gain_per_expansion": row.gain_per_expansion,
                "stopped": bool(row.stopped),
            }
            for row in report.rounds)

    def _evidence(self, best: SearchResult,
                  report: BudgetedSearchReport) -> Dict[str, Any]:
        evidence: Dict[str, Any] = {
            "stopped_reason": report.stopped_reason,
            "best_value": _finite(best.value, "best_value"),
            "budgets": [row.budget for row in report.rounds],
            "rounds": list(self._rounds(report)),
            "prm_score": _finite(best.prm_score, "prm_score"),
            "verifier_score": best.verifier_score,
            "verifier_complete": best.verifier_complete,
            "detail": best.detail,
            "plan_detail": best.detail,
        }
        if self.evidence_of is not None:
            extra = self.evidence_of(best)
            if not isinstance(extra, Mapping):
                raise TypeError("evidence_of must return a mapping")
            evidence.update(dict(extra))
        return evidence

    def __call__(self, context: Any) -> LoopObservation:
        config = self.config
        budgets = self._budgets(context.budget_scale)
        search = BudgetedSearch(
            self.fabric,
            min_gain_per_expansion=config.min_gain_per_expansion,
            patience=config.patience)
        report = search.run(self.state(context), budgets=budgets)
        if report.best is None:
            evidence = {
                "stopped_reason": report.stopped_reason,
                "budgets": list(budgets),
                "rounds": list(self._rounds(report)),
            }
            return LoopObservation(
                0.0, cost=sum(budgets),
                error="budgeted search produced no result", evidence=evidence)
        best = report.best
        reward = _finite(best.value, "reward")
        if report.rounds:
            cost = _finite(report.rounds[-1].cumulative_expansions, "cost")
        else:
            cost = float(sum(budgets))
        return LoopObservation(
            reward=reward,
            cost=cost,
            risk=self._risk(best),
            artifact_id=(
                f"{config.artifact_prefix}:{context.seed}:{str(best.action)}"),
            evidence=self._evidence(best, report))
