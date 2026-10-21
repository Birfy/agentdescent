"""Learn-to-think: the framework evolves *when* to spend compute.

Search-R1 / ZigZag train a model to decide when to retrieve, search, or answer
directly, with verifiable rewards. The same decision exists in every evolution
run: spend a cheap rollout (direct), spend a deep tree search (search), or
spend a reflection pass (reflect) — and nothing ever learned *which* for a
given difficulty and budget. This module makes the routing rule an evolvable
artifact: a gated source ``Policy.route(difficulty, budget_left, reward)`` that
returns ``"direct"`` / ``"search"`` / ``"reflect"``. A problem is one workload
instance whose outcome depends on the route (each route has a quality model and
a cost model), and the meta reward is quality-per-cost — so the framework
learns, from its own measured returns, to search the hard tasks and answer the
easy ones directly, instead of paying a fixed price everywhere.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .evolution import EvolutionResult, Task, evolve
from .meta import (
    MetaOutcome, MetaReward, Problem, SourceSlot, auc, slot_reflector,
)

__all__ = [
    "ROUTE_SEED",
    "think_policy_spec",
    "think_problem",
    "quality_per_cost",
    "LearnToThinkReport",
    "learn_to_think",
]

#: The shipped route rule: answer directly unless the task is hard and the
#: budget allows a search. Deliberately conservative -- it refuses to search
#: when the budget is low, which is exactly the decision an evolved rule can
#: improve on (searching hard tasks pays more than the budget it costs).
ROUTE_SEED = """class Policy:
    # Answer directly unless the task is hard and the budget allows a search.
    def route(self, difficulty, budget_left, reward):
        if difficulty >= 0.7 and budget_left >= 0.5:
            return "search"
        if difficulty >= 0.4:
            return "reflect"
        return "direct"
"""

_ROUTES = ("direct", "search", "reflect")


def _gate_route(source: str) -> ast.Module:
    """The route rule's gate: one ``Policy`` class with a ``route`` method,
    built only from safe syntax (imports, calls, dunder names refused -- the
    same walk ``compile_policy_source`` runs)."""
    if not source or len(source) > 12_000:
        raise ValueError("route source is empty or too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("route source is not valid Python") from error
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "Policy":
        raise ValueError("route source must define exactly one class named Policy")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef)]
    if len(methods) != 1 or methods[0].name != "route":
        raise ValueError("Policy must define exactly one method, route")
    if len(methods[0].args.args) != 4:
        raise ValueError(
            "route(self, difficulty, budget_left, reward) must take exactly "
            "those four arguments")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            raise ValueError("imports are not allowed in a route rule")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attributes are forbidden")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("exec", "eval", "open", "__import__",
                                     "getattr", "setattr", "globals"):
            raise ValueError(f"forbidden call: {node.func.id}")
    return tree


def _validate_route(source: str) -> str:
    """The gate: ``_gate_route`` raises on anything unsafe, else the text passes."""
    source = (source or "").strip()
    _gate_route(source)
    return source + "\n"


def _build_route(source: str) -> Any:
    source = (source or "").strip()
    _gate_route(source)
    tree = ast.parse(source)
    builtins = dict(__build_class__=__builtins__["__build_class__"]
                    if isinstance(__builtins__, dict) else __builtins__.__build_class__)
    namespace: Dict[str, Any] = {"__builtins__": builtins, "__name__": "route"}
    exec(compile(tree, "<route-policy>", "exec"), namespace, namespace)
    policy = namespace["Policy"]()
    for d, b, r in [(0.2, 0.5, 0.5), (0.9, 0.8, 0.1), (0.5, 0.2, 0.5)]:
        choice = policy.route(d, b, r)
        if choice not in _ROUTES:
            raise ValueError(f"route() returned {choice!r}, not one of {_ROUTES}")
    return policy


def think_policy_spec(seed: str = ROUTE_SEED) -> SourceSlot:
    """The evolvable spec: the value is a ``Policy.route`` source, gated and
    compiled by :func:`_build_route` (a smoke test runs the route on an easy, a
    hard-with-budget and a hard-without-budget input)."""
    return SourceSlot(initial_value=seed.strip() + "\n", validate=_validate_route,
                      build=_build_route,
                      description=(
                          "The value is one class `Policy` with a method "
                          "`route(self, difficulty, budget_left, reward)` "
                          "returning exactly 'direct', 'search' or 'reflect'. "
                          "No imports, dunder access or forbidden calls. Reply "
                          "with the complete class in one ```python fence."))


#: ``route -> (cost_multiplier, quality_gain)``: the workload's cost/quality
#: models. "search" is expensive and lifts quality the most, "direct" is free
#: but only answers easy tasks, "reflect" sits between.
_COST_AND_GAIN = {"direct": (0.3, 0.0), "reflect": (0.6, 0.2), "search": (1.0, 0.5)}


def think_problem(
    *,
    lam: float = 0.1,
    base_quality: Optional[Callable[[float], float]] = None,
) -> Problem:
    """A :class:`~agentdescent.meta.Problem` over one workload instance.

    ``(policy, seed) -> MetaOutcome``: difficulty is drawn from ``seed``
    (``seed % 10 / 10``) and the budget from ``seed % 3`` (``0.4`` when the
    seed is a multiple of three, ``1.0`` otherwise). The route rule chooses,
    and the outcome's reward is ``quality - lam * cost`` where quality = base
    quality plus the route's gain and cost = the route's cost multiplier. A
    route whose cost exceeds the budget **overdraws**: its compute is cut
    short, so its gain collapses to zero. The budget therefore matters -- on a
    low budget the cheap direct answer is the only one that pays, while on a
    high budget the expensive search is worth it on hard tasks -- and the
    optimum route depends on both the difficulty and the budget. ``base_quality``
    maps a difficulty to a base quality (``1 - difficulty`` when omitted).
    ``detail`` carries the route, the difficulty and the raw cost, so the
    report can show what the rule decided and what it paid."""

    base = base_quality or (lambda d: 1.0 - d)

    def problem(policy: Any, seed: int) -> MetaOutcome:
        difficulty = (seed % 10) / 10.0
        budget_left = 1.0 if seed % 3 else 0.4
        reward = 0.4
        try:
            route = policy.route(difficulty, budget_left, reward)
            if route not in _ROUTES:
                route = "direct"
        except Exception:  # noqa: BLE001 - a broken rule is a direct answer
            route = "direct"
        mult, gain = _COST_AND_GAIN[route]
        if mult > budget_left:
            gain = 0.0     # the route overdrew its budget: compute cut short
        quality = min(1.0, base(difficulty) + gain)
        value = quality - lam * mult
        return MetaOutcome(curve=[value, value], final=value, rollouts=1,
                           detail={"route": route, "difficulty": difficulty,
                                   "cost": mult, "budget_left": budget_left})

    return problem


def quality_per_cost(outcome: MetaOutcome) -> float:
    """The outcome's reward, read back (already quality-per-cost)."""
    return min(1.0, max(0.0, auc(outcome)))


@dataclass(frozen=True)
class LearnToThinkReport:
    """What :func:`learn_to_think` found: the evolved route rule and how its
    quality-per-cost compares with the default on validation problems."""

    seed_value: str
    value: str
    seed_score: float
    evolved_score: float
    gain: float
    routes_on: Dict[str, str]


def learn_to_think(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    seeds: Sequence[int] = (0, 1, 2, 3),
    val_seeds: Sequence[int] = (4, 5, 6, 7),
    seed_source: str = ROUTE_SEED,
    **evolve_kwargs: Any,
) -> Tuple[LearnToThinkReport, EvolutionResult]:
    """Evolve the route rule on quality-per-cost.

    A bespoke ``evolve()`` over ``think_problem`` tasks (the artifact is the
    route source, the propose rewrites it, the reward is the outcome's
    quality-per-cost); then the evolved rule and the default are both scored on
    ``val_problems`` and :class:`LearnToThinkReport` compares them, with the
    route each chose on an easy and a hard input.

    ``problems`` / ``val_problems`` may be a mapping or a sequence; a sequence
    is normalised to ``{"p0": ...}`` before use, so ``dict(problems)`` and the
    ``.values()`` scoring never crash on a list.
    """
    if propose is None and model is None:
        raise ValueError("learn_to_think() needs propose= or model=")
    spec = think_policy_spec(seed=seed_source)
    problems = _as_problem_map(problems)
    val_problems = _as_problem_map(val_problems)
    named = dict(problems)
    tasks = [Task(id=f"{name}:{seed}", prompt=f"{name} (seed {seed})",
                  meta={"problem": name, "seed": seed})
             for seed in seeds for name in named]
    reflector = propose if propose is not None else slot_reflector(model, spec)
    spec.compile(spec.render(spec.initial()))      # the seed must pass its gate

    def run(rendered: str, task: Task) -> str:
        try:
            policy = spec.compile(rendered)
        except ValueError as error:
            return MetaOutcome(detail={"error": f"compile: {error}"}).to_json()
        outcome = named[task.meta["problem"]](policy, int(task.meta["seed"]))
        return outcome.to_json()

    def reward(task: Task, output: str) -> float:
        try:
            return float(quality_per_cost(MetaOutcome.from_json(output)))
        except (TypeError, ValueError, KeyError):
            return 0.0

    result = evolve(
        tasks, reward, run=run, propose=reflector, strategy=spec,
        initial_state=spec.initial(), self_verify=False, solved_threshold=1.1,
        **evolve_kwargs)
    value = result.rendered or spec.render(spec.initial())
    seed_value = spec.render(spec.initial())

    seed_score = _mean_score(spec.compile(seed_value), val_problems, val_seeds)
    evolved_score = _mean_score(spec.compile(value), val_problems, val_seeds)
    evolved_policy = spec.compile(value)
    easy_route = _route_on(evolved_policy, 0.0, 1.0, 0.4)
    hard_route = _route_on(evolved_policy, 0.9, 1.0, 0.4)
    report = LearnToThinkReport(
        seed_value=seed_value, value=value,
        seed_score=seed_score, evolved_score=evolved_score,
        gain=evolved_score - seed_score,
        routes_on={"easy": easy_route, "hard": hard_route})
    try:
        result.meta = {
            "seed_score": seed_score, "evolved_score": evolved_score,
            "gain": evolved_score - seed_score,
            "routes_on": {"easy": easy_route, "hard": hard_route},
        }
    except AttributeError:
        pass
    return report, result


def _as_problem_map(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
) -> Mapping[str, Problem]:
    """A sequence of problems becomes ``{"p0": ..., "p1": ...}``; a mapping
    passes through unchanged, so every problem is scored by name."""
    if isinstance(problems, Mapping):
        return problems
    return {f"p{i}": p for i, p in enumerate(problems)}


def _mean_score(policy: Any, problems: Mapping[str, Problem],
                seeds: Sequence[int]) -> float:
    total = 0.0
    n = 0
    for problem in problems.values():
        for seed in seeds:
            total += float(quality_per_cost(problem(policy, seed)))
            n += 1
    return total / n if n else 0.0


def _route_on(policy: Any, difficulty: float, budget: float, reward: float) -> str:
    try:
        choice = policy.route(difficulty, budget, reward)
        return choice if choice in _ROUTES else "direct"
    except Exception:  # noqa: BLE001 - a broken rule answers directly
        return "direct"
