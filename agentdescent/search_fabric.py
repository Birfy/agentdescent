"""Process-supervised search: a PRM and verifier as the planner's value.

`planning.MCTSPlanner` needs a value function for non-terminal leaves.
`process_reward.ProcessRewardModel` is exactly that step-level scorer, but
nothing connected the two; callers had to write the adapter and remember that
the PRM needs a feature vector, not an opaque state. This module is that
adapter, plus an optional verifier vote:

    fabric = SearchFabric(
        simulator, prm, state_features,
        budget=200, verifier=ensemble, verifier_weight=0.25)
    result = fabric.plan(initial_state)
    result.action, result.value, result.prm_score, result.verifier_score

It also exposes the PRM as a reranker for candidates that never enter the
planner:

    fabric.rank(candidates) / fabric.select(candidates)

The fabric does not train the PRM, run the simulator or invent a feature map.
It owns the one contract those modules need: state -> features -> PRM score,
optionally blended with a verifier panel, then handed to UCT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from .planning import MCTSPlanner, PlanResult
from .process_reward import ProcessRewardModel
from .verifier_ensemble import EnsembleReport, VerifierEnsemble

__all__ = ["SearchResult", "SearchFabric"]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class SearchResult:
    """The planner's answer plus the scores that guided it."""

    plan: PlanResult
    value: float
    prm_score: float
    verifier_score: Optional[float] = None
    verifier_passed: Optional[bool] = None
    verifier_complete: Optional[bool] = None
    detail: str = ""

    @property
    def action(self) -> Any:
        return self.plan.action


class SearchFabric:
    """A planner whose leaf value is a PRM (optionally blended with a verifier).

    A verifier panel is evaluated at every leaf the UCT asks for, so its cost
    is paid once per leaf evaluation. ``rank``/``select`` are PRM-only.
    """

    def __init__(self, simulator: Any, prm: ProcessRewardModel,
                 feature_fn: Callable[[Any], Sequence[float]], *,
                 budget: int = 100, c: float = 1.4,
                 discount: float = 1.0, cost_weight: float = 0.0,
                 seed: int = 0,
                 verifier: Optional[VerifierEnsemble] = None,
                 verifier_weight: float = 0.5,
                 verifier_cost: Optional[float] = None) -> None:
        if not isinstance(prm, ProcessRewardModel):
            raise TypeError("prm must be a ProcessRewardModel")
        if not callable(feature_fn):
            raise TypeError("feature_fn must be callable")
        if verifier is not None and not isinstance(verifier, VerifierEnsemble):
            raise TypeError("verifier must be a VerifierEnsemble or None")
        weight = _finite(verifier_weight, "verifier_weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("verifier_weight must be in [0, 1]")
        if verifier is None:
            weight = 0.0
        if verifier_cost is not None:
            verifier_cost = _finite(verifier_cost, "verifier_cost")
            if verifier_cost < 0:
                raise ValueError("verifier_cost must be >= 0")
        self.prm = prm
        self.feature_fn = feature_fn
        self.verifier = verifier
        self.verifier_weight = weight
        self.verifier_cost = verifier_cost
        self.planner = MCTSPlanner(
            simulator, self.value, budget=budget, c=c, discount=discount,
            cost_weight=cost_weight, seed=seed)

    def prm_value(self, state: Any) -> float:
        return float(self.prm.score(self.feature_fn(state)))

    def verifier_value(self, state: Any) -> Optional[EnsembleReport]:
        if self.verifier is None:
            return None
        return self.verifier.evaluate(state, max_cost=self.verifier_cost)

    def value(self, state: Any) -> float:
        """The value function handed to UCT."""
        prm_score = self.prm_value(state)
        report = self.verifier_value(state)
        if report is None:
            return prm_score
        if not report.complete:
            # An incomplete verifier panel is not a green vote; fall back to the
            # PRM rather than treating a missing verifier as a pass.
            return prm_score
        return ((1.0 - self.verifier_weight) * prm_score
                + self.verifier_weight * float(report.score))

    def plan(self, state: Any) -> SearchResult:
        plan = self.planner.plan(state)
        target = plan.state if plan.state is not None else state
        prm_score = self.prm_value(target)
        report = self.verifier_value(target)
        verifier_score = None if report is None else float(report.score)
        verifier_complete = None if report is None else bool(report.complete)
        verifier_passed = (None if report is None else
                           bool(report.passed and report.complete))
        detail = (f"plan value={plan.value:.4f}, target prm={prm_score:.4f}"
                  + ("" if report is None else
                     f", verifier={report.score:.4f} ({report.confidence:.3f})"
                     + ("" if report.complete else ", verifier incomplete")))
        return SearchResult(
            plan=plan, value=plan.value, prm_score=prm_score,
            verifier_score=verifier_score, verifier_passed=verifier_passed,
            verifier_complete=verifier_complete, detail=detail)

    def rank(self, candidates: Sequence[Any]) -> Tuple[int, ...]:
        """PRM-only ranking; the verifier is not part of this shortcut."""
        features = [self.feature_fn(candidate) for candidate in candidates]
        return self.prm.rank(features)

    def select(self, candidates: Sequence[Any]) -> int:
        """PRM-only selection; the verifier is not part of this shortcut."""
        features = [self.feature_fn(candidate) for candidate in candidates]
        return self.prm.select(features)
