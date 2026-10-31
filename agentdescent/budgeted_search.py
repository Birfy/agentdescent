"""Spend search compute until another expansion stops paying.

`SearchFabric` runs one UCT search at a fixed budget. A product needs the outer
question: should this task get 5, 20 or 60 expansions? `BudgetedSearch` runs
increasing budgets through the same fabric and stops when the value gain per
expansion falls below a bar, using the existing `MarginalGainStop` or a simple
patience rule.

    search = BudgetedSearch(fabric, patience=1, min_gain_per_expansion=1e-3)
    report = search.run(initial_state, budgets=(5, 20, 60))
    report.best.value, report.stopped_reason, report.rounds

The fabric's planner budget is changed between rounds; the state and simulator
are not copied. The report keeps every round and the value it bought, so a
caller can audit the spend rather than only seeing the winner.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

from .search_fabric import SearchFabric, SearchResult
from .stopping import MarginalGainStop

__all__ = ["SearchRound", "BudgetedSearchReport", "BudgetedSearch"]


@dataclass(frozen=True)
class SearchRound:
    budget: int
    result: SearchResult
    gain: Optional[float] = None
    gain_per_expansion: Optional[float] = None
    cumulative_expansions: int = 0
    stopped: bool = False


@dataclass(frozen=True)
class BudgetedSearchReport:
    rounds: Tuple[SearchRound, ...]
    best: Optional[SearchResult]
    stopped_reason: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rounds": [
                {"budget": row.budget, "value": row.result.value,
                 "action": row.result.action, "gain": row.gain,
                 "gain_per_expansion": row.gain_per_expansion,
                 "cumulative_expansions": row.cumulative_expansions,
                 "stopped": row.stopped}
                for row in self.rounds],
            "best_value": None if self.best is None else self.best.value,
            "stopped_reason": self.stopped_reason,
            "detail": self.detail,
        }


class BudgetedSearch:
    """Run a search at increasing budgets and stop when the gain plateaus."""

    def __init__(self, fabric: SearchFabric, *,
                 min_gain_per_expansion: float = 1e-4,
                 patience: int = 2,
                 stop: Optional[MarginalGainStop] = None,
                 seed: int = 0) -> None:
        if not isinstance(fabric, SearchFabric):
            raise TypeError("fabric must be a SearchFabric")
        gain = float(min_gain_per_expansion)
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("min_gain_per_expansion must be finite and >= 0")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be an int >= 1")
        if stop is not None and not isinstance(stop, MarginalGainStop):
            raise TypeError("stop must be a MarginalGainStop or None")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an int")
        self.fabric = fabric
        self.min_gain_per_expansion = gain
        self.patience = patience
        self.stop = stop
        self.seed = seed

    def run(self, state: Any, *, budgets: Sequence[int]) -> BudgetedSearchReport:
        planned = list(budgets)
        if not planned or any(
                isinstance(budget, bool) or not isinstance(budget, int)
                or budget < 1 for budget in planned):
            raise ValueError("budgets must be positive integers")
        planned = sorted(set(planned))
        if self.stop is not None:
            self.stop.reset()
        planner = self.fabric.planner
        original_budget = planner.budget
        original_rng = planner.rng.getstate()
        rounds = []
        best = None
        previous = None
        previous_budget = None
        cumulative = 0
        stagnant = 0
        stopped_reason = "budgets"
        try:
            for index, budget in enumerate(planned):
                planner.budget = budget
                planner.rng = random.Random(self.seed + index)
                result = self.fabric.plan(state)
                cumulative += budget
                gain = None if previous is None else result.value - previous
                delta_budget = (None if previous_budget is None
                                else budget - previous_budget)
                per_expansion = (None if gain is None or not delta_budget
                                 else gain / delta_budget)
                rounds.append(SearchRound(
                    budget, result, gain, per_expansion, cumulative))
                if best is None or result.value > best.value:
                    best = result
                stop_now = False
                if self.stop is not None:
                    stop_now = self.stop(SimpleNamespace(
                        calls=cumulative, held_out_reward=result.value))
                if per_expansion is not None and per_expansion <= self.min_gain_per_expansion:
                    stagnant += 1
                    if stagnant >= self.patience:
                        stop_now = True
                        stopped_reason = "patience"
                else:
                    stagnant = 0
                if stop_now:
                    if stopped_reason == "budgets":
                        stopped_reason = "stopping"
                    rounds[-1] = SearchRound(
                        budget, result, gain, per_expansion, cumulative,
                        stopped=True)
                    break
                previous = result.value
                previous_budget = budget
        finally:
            planner.budget = original_budget
            planner.rng.setstate(original_rng)
        detail = (f"{len(rounds)} round(s), best={None if best is None else best.value}, "
                  f"stopped={stopped_reason}")
        return BudgetedSearchReport(
            rounds=tuple(rounds), best=best, stopped_reason=stopped_reason,
            detail=detail)
