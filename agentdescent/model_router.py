"""Route each task to the cheapest model that is likely good enough.

An RSI loop calls models thousands of times. Sending every call to the strongest
model is the easiest way to waste a budget; sending every call to the cheapest
one is the easiest way to poison the archive with low-quality edits. Model
routing is the middle path: a contextual bandit over quality and cost.

This module is deliberately small:

* :class:`ModelSpec` is one model's cost, context limit, tags and quality prior;
* :class:`RouteContext` says what the task needs (tokens, tags, budget);
* :class:`ModelRouter` uses Thompson sampling on a Beta quality posterior and
  subtracts a normalized cost penalty; a small epsilon explores;
* ``observe`` updates the posterior and the measured cost.

    router = ModelRouter([
        ModelSpec("fast", cost=1, quality_prior=(2, 2)),
        ModelSpec("strong", cost=20, quality_prior=(6, 2)),
    ], cost_weight=0.4)
    decision = router.choose(RouteContext(tokens=2000))
    router.observe(decision.model, reward=0.9)

The router learns per model, not per task family; a caller with domain features
can wrap `choose` in its own policy. The point here is the cost/quality loop,
not a new model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ModelSpec",
    "RouteContext",
    "RouteDecision",
    "ModelReport",
    "RoutingReport",
    "ModelRouter",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cost: float
    max_context: int = 128_000
    tags: Tuple[str, ...] = ()
    quality_prior: Tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelSpec.name is required")
        cost = float(self.cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("ModelSpec.cost must be finite and >= 0")
        if self.max_context < 1:
            raise ValueError("ModelSpec.max_context must be >= 1")
        if len(self.quality_prior) != 2:
            raise ValueError("quality_prior must be (alpha, beta)")
        alpha, beta = (float(x) for x in self.quality_prior)
        if not math.isfinite(alpha) or not math.isfinite(beta) or alpha <= 0 or beta <= 0:
            raise ValueError("quality_prior alpha/beta must be positive and finite")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "quality_prior", (alpha, beta))


@dataclass(frozen=True)
class RouteContext:
    task_id: str = ""
    tokens: int = 0
    required_tags: Tuple[str, ...] = ()
    max_cost: Optional[float] = None
    difficulty: float = 0.5

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens must be >= 0")
        if self.max_cost is not None and (
                not math.isfinite(float(self.max_cost)) or float(self.max_cost) < 0):
            raise ValueError("max_cost must be finite and >= 0 or None")
        if not 0.0 <= float(self.difficulty) <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        object.__setattr__(self, "required_tags", tuple(str(tag) for tag in self.required_tags))


@dataclass(frozen=True)
class RouteDecision:
    model: str
    reason: str
    expected_quality: float
    sampled_quality: float
    expected_cost: float
    exploration: bool = False


@dataclass(frozen=True)
class ModelReport:
    model: str
    pulls: int
    mean_reward: float
    mean_cost: float
    posterior_mean: float
    detail: str = ""


@dataclass(frozen=True)
class RoutingReport:
    models: Tuple[ModelReport, ...]
    pulls: int
    mean_reward: float
    mean_cost: float
    detail: str = ""


class ModelRouter:
    """A cost-aware Thompson-sampling router."""

    def __init__(self, models: Sequence[ModelSpec], *,
                 cost_weight: float = 0.5, exploration: float = 0.05,
                 seed: int = 0) -> None:
        if not models:
            raise ValueError("ModelRouter needs at least one model")
        if len({model.name for model in models}) != len(models):
            raise ValueError("model names must be unique")
        if not math.isfinite(float(cost_weight)) or cost_weight < 0:
            raise ValueError("cost_weight must be finite and >= 0")
        if not math.isfinite(float(exploration)) or not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        self.models = tuple(models)
        self.cost_weight = float(cost_weight)
        self.exploration = float(exploration)
        self._rng = random.Random(seed)
        self._stats = {
            model.name: {"pulls": 0, "reward": 0.0, "cost": 0.0,
                         "alpha": model.quality_prior[0],
                         "beta": model.quality_prior[1]}
            for model in models
        }

    def _eligible(self, context: RouteContext) -> List[ModelSpec]:
        needed = set(context.required_tags)
        out = []
        for model in self.models:
            if context.tokens > model.max_context:
                continue
            if not needed.issubset(set(model.tags)):
                continue
            if context.max_cost is not None and model.cost > context.max_cost:
                continue
            out.append(model)
        if not out:
            raise ValueError("no model satisfies the context constraints")
        return out

    def choose(self, context: Optional[RouteContext] = None) -> RouteDecision:
        context = context or RouteContext()
        candidates = self._eligible(context)
        if self.exploration and self._rng.random() < self.exploration:
            model = self._rng.choice(candidates)
            stats = self._stats[model.name]
            posterior = stats["alpha"] / (stats["alpha"] + stats["beta"])
            return RouteDecision(model.name, "exploration", posterior, posterior,
                                 model.cost, True)
        max_cost = max(model.cost for model in candidates) or 1.0
        # Harder tasks make the router more willing to pay for quality; easy
        # tasks make price dominate. That is the context entering the bandit.
        effective_cost_weight = self.cost_weight * (1.0 - 0.8 * float(context.difficulty))
        best_model = None
        best_score = float("-inf")
        best_sampled = 0.0
        best_posterior = 0.0
        for model in candidates:
            stats = self._stats[model.name]
            sampled = self._rng.betavariate(stats["alpha"], stats["beta"])
            posterior = stats["alpha"] / (stats["alpha"] + stats["beta"])
            cost = model.cost / max_cost
            score = sampled - effective_cost_weight * cost
            if score > best_score:
                best_model, best_score = model, score
                best_sampled, best_posterior = sampled, posterior
        assert best_model is not None
        return RouteDecision(best_model.name, "thompson", best_posterior,
                             best_sampled, best_model.cost, False)

    def observe(self, model: str, *, reward: float, cost: Optional[float] = None) -> None:
        if model not in self._stats:
            raise KeyError(f"unknown model {model!r}")
        reward = float(reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be in [0, 1]")
        stats = self._stats[model]
        stats["pulls"] += 1
        stats["reward"] += reward
        stats["alpha"] += reward
        stats["beta"] += 1.0 - reward
        if cost is None:
            model_spec = next(m for m in self.models if m.name == model)
            cost = model_spec.cost
        cost = float(cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost must be finite and >= 0")
        stats["cost"] += cost

    def report(self) -> RoutingReport:
        rows = []
        total_pulls = 0
        total_reward = 0.0
        total_cost = 0.0
        for model in self.models:
            stats = self._stats[model.name]
            pulls = stats["pulls"]
            total_pulls += pulls
            total_reward += stats["reward"]
            total_cost += stats["cost"]
            rows.append(ModelReport(
                model=model.name, pulls=pulls,
                mean_reward=stats["reward"] / pulls if pulls else 0.0,
                mean_cost=stats["cost"] / pulls if pulls else 0.0,
                posterior_mean=stats["alpha"] / (stats["alpha"] + stats["beta"]),
                detail=f"alpha={stats['alpha']:.3f} beta={stats['beta']:.3f}"))
        return RoutingReport(
            models=tuple(rows), pulls=total_pulls,
            mean_reward=total_reward / total_pulls if total_pulls else 0.0,
            mean_cost=total_cost / total_pulls if total_pulls else 0.0,
            detail=f"{total_pulls} route(s), cost={total_cost:g}")
