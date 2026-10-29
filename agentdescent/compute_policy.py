"""Choose how much compute a task deserves, not just which model to call.

`model_router` answers "which model". An RSI loop also has to answer "which
strategy": a direct answer, a process-reward rerank, a tree search, or a
verifier ensemble. The strategies have different cost/quality profiles, and the
right choice depends on task difficulty, context size and the remaining budget.

This module is the action-level bandit on top of the model router:

* :class:`ComputeOption` is one strategy (`direct`, `prm_rerank`, `mcts`,
  `verifier_ensemble`) with a call multiplier, a quality prior and a difficulty
  band;
* :class:`ComputeController` samples an option quality, asks the injected
  `ModelRouter` for a concrete model, and subtracts the normalized cost;
* ``observe`` updates both the option posterior and the model posterior;
* the chosen option, model, expected quality/cost and exploration flag are one
  JSON-serialisable decision for the decision log.

    controller = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, calls=1),
        ComputeOption("prm", ComputeAction.PRM_RERANK, calls=2,
                      quality_prior=(4, 2), min_difficulty=0.3),
        ComputeOption("mcts", ComputeAction.MCTS, calls=8,
                      quality_prior=(6, 2), min_difficulty=0.6),
    ], router)
    decision = controller.choose(ComputeContext(difficulty=0.8, tokens=4000))
    controller.observe(decision, reward=0.9)

The controller does not run the strategy. Callers map ``decision.action`` to
their PRM/planner/ensemble; the controller owns cost/quality allocation and the
audit record.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .model_router import ModelRouter, RouteContext, RoutingReport

__all__ = [
    "ComputeAction",
    "ComputeOption",
    "ComputeContext",
    "ComputeDecision",
    "OptionReport",
    "ComputeReport",
    "ComputeController",
]


class ComputeAction:
    """The closed strategy vocabulary this policy allocates over."""

    DIRECT = "direct"
    PRM_RERANK = "prm_rerank"
    MCTS = "mcts"
    VERIFIER_ENSEMBLE = "verifier_ensemble"

    ALL = (DIRECT, PRM_RERANK, MCTS, VERIFIER_ENSEMBLE)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ComputeOption:
    """One strategy's cost/quality profile and when it is eligible."""

    name: str
    action: str
    calls: int = 1
    quality_prior: Tuple[float, float] = (1.0, 1.0)
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ComputeOption.name is required")
        if self.action not in ComputeAction.ALL:
            raise ValueError(
                f"unknown compute action {self.action!r}; expected one of "
                f"{ComputeAction.ALL}")
        if isinstance(self.calls, bool) or not isinstance(self.calls, int) \
                or self.calls < 1:
            raise ValueError("calls must be an int >= 1")
        if len(self.quality_prior) != 2:
            raise ValueError("quality_prior must be (alpha, beta)")
        alpha, beta = (float(x) for x in self.quality_prior)
        if not math.isfinite(alpha) or not math.isfinite(beta) \
                or alpha <= 0 or beta <= 0:
            raise ValueError("quality_prior alpha/beta must be positive and finite")
        low = _finite(self.min_difficulty, "min_difficulty")
        high = _finite(self.max_difficulty, "max_difficulty")
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("need 0 <= min_difficulty <= max_difficulty <= 1")
        object.__setattr__(self, "quality_prior", (alpha, beta))
        object.__setattr__(self, "min_difficulty", low)
        object.__setattr__(self, "max_difficulty", high)


@dataclass(frozen=True)
class ComputeContext:
    """What the task asks of the compute policy."""

    task_id: str = ""
    tokens: int = 0
    difficulty: float = 0.5
    max_cost: Optional[float] = None
    required_tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) \
                or self.tokens < 0:
            raise ValueError("tokens must be an int >= 0")
        difficulty = _finite(self.difficulty, "difficulty")
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        if self.max_cost is not None:
            cost = _finite(self.max_cost, "max_cost")
            if cost < 0:
                raise ValueError("max_cost must be >= 0 or None")
            object.__setattr__(self, "max_cost", cost)
        object.__setattr__(self, "difficulty", difficulty)
        if isinstance(self.required_tags, str):
            raise ValueError("required_tags must be a sequence, not a bare string")
        object.__setattr__(self, "required_tags",
                           tuple(str(tag) for tag in self.required_tags))


@dataclass(frozen=True)
class ComputeDecision:
    """One allocated strategy + model, with the numbers used to choose it."""

    option: str
    action: str
    model: str
    reason: str
    expected_quality: float
    sampled_quality: float
    expected_cost: float
    calls: int
    exploration: bool = False
    model_reason: str = ""
    detail: str = ""
    reservation_id: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "option": self.option,
            "action": self.action,
            "model": self.model,
            "reason": self.reason,
            "expected_quality": self.expected_quality,
            "sampled_quality": self.sampled_quality,
            "expected_cost": self.expected_cost,
            "calls": self.calls,
            "exploration": self.exploration,
            "model_reason": self.model_reason,
            "detail": self.detail,
            "reservation_id": self.reservation_id,
        }


@dataclass(frozen=True)
class OptionReport:
    option: str
    action: str
    pulls: int
    mean_reward: float
    mean_cost: float
    posterior_mean: float
    detail: str = ""


@dataclass(frozen=True)
class ComputeReport:
    options: Tuple[OptionReport, ...]
    router: RoutingReport
    pulls: int
    mean_reward: float
    mean_cost: float
    total_calls: int
    reserved_calls: int = 0
    detail: str = ""


class ComputeController:
    """A cost-aware Thompson-sampling bandit over compute strategies.

    ``choose`` calls the injected ``ModelRouter`` once per eligible option, so
    a shared router's RNG advances during allocation (same seed, same sequence).
    The router's cost is a per-call model price; a strategy's ``calls`` only
    multiplies it for the allocation decision, and ``observe`` forwards
    ``spent / calls`` back so the router's mean cost stays per call.

    Reward attribution is intentionally simple: a successful strategy also
    updates the model posterior. That confounds "this model is good" with "this
    strategy is good"; callers that need to separate them can pass a router
    dedicated to one strategy, or observe the model outside this controller.
    """

    def __init__(self, options: Sequence[ComputeOption], router: ModelRouter, *,
                 cost_weight: float = 0.5, exploration: float = 0.05,
                 max_calls: Optional[int] = None, seed: int = 0) -> None:
        if not options:
            raise ValueError("ComputeController needs at least one option")
        names = [option.name for option in options]
        if len(set(names)) != len(names):
            raise ValueError("option names must be unique")
        if not isinstance(router, ModelRouter):
            raise TypeError("router must be a ModelRouter")
        weight = _finite(cost_weight, "cost_weight")
        if weight < 0:
            raise ValueError("cost_weight must be >= 0")
        explore = _finite(exploration, "exploration")
        if not 0.0 <= explore <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if max_calls is not None:
            if isinstance(max_calls, bool) or not isinstance(max_calls, int) \
                    or max_calls < 1:
                raise ValueError("max_calls must be an int >= 1 or None")
        self.options = tuple(options)
        self.router = router
        self.cost_weight = weight
        self.exploration = explore
        self.max_calls = max_calls
        self._rng = random.Random(seed)
        self._stats = {
            option.name: {"pulls": 0, "reward": 0.0, "cost": 0.0,
                          "alpha": option.quality_prior[0],
                          "beta": option.quality_prior[1]}
            for option in self.options
        }
        self._pulls = 0
        self._reward = 0.0
        self._cost = 0.0
        self._calls = 0
        self._reserved_calls = 0
        self._reservations: Dict[str, int] = {}
        self._next_reservation = 0

    @property
    def total_calls(self) -> int:
        return self._calls

    @property
    def reserved_calls(self) -> int:
        return self._reserved_calls

    def _reserve(self, calls: int) -> str:
        reservation = f"r{self._next_reservation}"
        self._next_reservation += 1
        self._reservations[reservation] = calls
        self._reserved_calls += calls
        return reservation

    def _eligible(self, context: ComputeContext) -> List[Tuple[ComputeOption, object]]:
        out = []
        for option in self.options:
            if not option.min_difficulty <= context.difficulty <= option.max_difficulty:
                continue
            if (self.max_calls is not None
                    and self._calls + self._reserved_calls + option.calls
                    > self.max_calls):
                continue
            per_call_budget = (None if context.max_cost is None
                               else context.max_cost / option.calls)
            route_context = RouteContext(
                task_id=context.task_id, tokens=context.tokens,
                required_tags=context.required_tags,
                max_cost=per_call_budget, difficulty=context.difficulty)
            try:
                model = self.router.choose(route_context)
            except ValueError:
                continue
            cost = model.expected_cost * option.calls
            if context.max_cost is not None and cost > context.max_cost + 1e-12:
                continue
            out.append((option, model))
        if not out:
            if (self.max_calls is not None
                    and self._calls + self._reserved_calls >= self.max_calls):
                raise RuntimeError("compute budget exhausted")
            raise ValueError("no compute option satisfies the context")
        return out

    def choose(self, context: Optional[ComputeContext] = None) -> ComputeDecision:
        context = context or ComputeContext()
        candidates = self._eligible(context)
        max_cost = max(model.expected_cost * option.calls
                       for option, model in candidates) or 1.0
        if self.exploration and self._rng.random() < self.exploration:
            option, model = self._rng.choice(candidates)
            stats = self._stats[option.name]
            posterior = stats["alpha"] / (stats["alpha"] + stats["beta"])
            reservation = self._reserve(option.calls)
            return ComputeDecision(
                option=option.name, action=option.action, model=model.model,
                reason="exploration", expected_quality=posterior,
                sampled_quality=posterior,
                expected_cost=model.expected_cost * option.calls,
                calls=option.calls, exploration=True,
                model_reason=model.reason,
                detail=f"forced exploration of {option.name}",
                reservation_id=reservation)
        best = None
        best_score = float("-inf")
        best_sampled = 0.0
        best_posterior = 0.0
        for option, model in candidates:
            stats = self._stats[option.name]
            sampled = self._rng.betavariate(stats["alpha"], stats["beta"])
            posterior = stats["alpha"] / (stats["alpha"] + stats["beta"])
            cost = (model.expected_cost * option.calls) / max_cost
            score = sampled - self.cost_weight * cost
            if score > best_score:
                best = (option, model)
                best_score = score
                best_sampled = sampled
                best_posterior = posterior
        assert best is not None
        option, model = best
        reservation = self._reserve(option.calls)
        return ComputeDecision(
            option=option.name, action=option.action, model=model.model,
            reason="thompson", expected_quality=best_posterior,
            sampled_quality=best_sampled,
            expected_cost=model.expected_cost * option.calls,
            calls=option.calls, model_reason=model.reason,
            detail=(f"sampled {best_sampled:.3f}, cost "
                    f"{model.expected_cost * option.calls:g}"),
            reservation_id=reservation)

    def observe(self, decision: ComputeDecision, *, reward: float,
                cost: Optional[float] = None) -> None:
        if not isinstance(decision, ComputeDecision):
            raise TypeError("decision must be a ComputeDecision")
        if decision.option not in self._stats:
            raise KeyError(f"unknown compute option {decision.option!r}")
        value = _finite(reward, "reward")
        if not 0.0 <= value <= 1.0:
            raise ValueError("reward must be in [0, 1]")
        spent = (decision.expected_cost if cost is None
                 else _finite(cost, "cost"))
        if spent < 0:
            raise ValueError("cost must be >= 0")
        if decision.reservation_id:
            reserved = self._reservations.pop(decision.reservation_id, None)
            if reserved is None:
                raise ValueError(
                    f"unknown or already-settled reservation "
                    f"{decision.reservation_id!r}")
            self._reserved_calls -= reserved
        stats = self._stats[decision.option]
        stats["pulls"] += 1
        stats["reward"] += value
        stats["alpha"] += value
        stats["beta"] += 1.0 - value
        stats["cost"] += spent
        self._pulls += 1
        self._reward += value
        self._cost += spent
        self._calls += decision.calls
        if decision.model:
            self.router.observe(decision.model, reward=value,
                                cost=spent / decision.calls)

    def report(self) -> ComputeReport:
        rows = []
        for option in self.options:
            stats = self._stats[option.name]
            pulls = stats["pulls"]
            rows.append(OptionReport(
                option=option.name, action=option.action, pulls=pulls,
                mean_reward=stats["reward"] / pulls if pulls else 0.0,
                mean_cost=stats["cost"] / pulls if pulls else 0.0,
                posterior_mean=(stats["alpha"] / (stats["alpha"] + stats["beta"])),
                detail=f"alpha={stats['alpha']:.3f} beta={stats['beta']:.3f}"))
        return ComputeReport(
            options=tuple(rows), router=self.router.report(), pulls=self._pulls,
            mean_reward=self._reward / self._pulls if self._pulls else 0.0,
            mean_cost=self._cost / self._pulls if self._pulls else 0.0,
            total_calls=self._calls, reserved_calls=self._reserved_calls,
            detail=(f"{self._pulls} decision(s), {self._calls} call(s), "
                    f"cost={self._cost:g}"))
