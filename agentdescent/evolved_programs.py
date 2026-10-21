"""Evolved decision code that beats the shipped implementation on quality-per-cost.

``priority_selection`` (#350's surface) already evolves the ``selection``
priority rule behind an AST gate, but its reward is quality alone — an evolved
rule that costs more than the shipped seed can win on AUC and lose on latency,
invisibly. This is the AlphaEvolve / FunSearch idea applied to the decision
plane: the reward includes **cost**, so the search finds a rule that is as good
*and cheaper* than the hand-written default, and the report measures the
evolved rule against the shipped seed on both axes.

The cost of a candidate rule is counted from its own source — a deterministic,
attributable proxy for what evaluating it costs (an arithmetic node is a
constant unit of work), recorded in the outcome's ``detail`` and used by the
reward as ``quality - lam * (cost / baseline)``. The problem is a fixed grid of
priority inputs whose reference ordering is the shipped seed's own — so the
baseline scores ``quality = 1`` at ``cost = baseline``, and a rule must be
cheaper at equal quality (or better at equal cost) to beat it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .evolution import EvolutionResult, Task
from .meta import (
    MetaOutcome, MetaReward, Problem, SlotSpec, auc, meta_evolve,
    meta_validate, priority_selection, PRIORITY_SEED,
)

__all__ = [
    "source_ops",
    "priority_grid_problem",
    "cost_adjusted_reward",
    "FastRuleReport",
    "evolve_fast_rule",
]

#: A fixed grid of priority inputs, spanning the root before any expansion, a
#: lone node, deep nodes and unrated priors -- the same regime
#: ``compile_priority`` refuses a rule on.
_GRID = [
    (0.5, 0, 0, 1.0, 0, 1),
    (0.0, 0, 1, 0.5, 0, 2),
    (1.0, 3, 7, 0.25, 1, 4),
    (0.33, 12, 40, 0.01, 6, 25),
    (0.9, 1, 300, 0.0, 12, 120),
]


def source_ops(source: str) -> int:
    """The deterministic cost of a candidate rule: arithmetic/call nodes in its
    source, counted from the AST. An attributable unit of work -- evaluating a
    rule costs what its expression tree costs, and counting it from the source
    is reproducible where wallclock is not."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 10 ** 6
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)):
            count += 1
        elif isinstance(node, ast.Name) and node.id in ("sqrt", "log", "log1p",
                                                        "exp", "tanh", "pow"):
            count += 1
    return count


def _ordering(policy: Any) -> List[int]:
    """The grid order the policy's priority induces (best first)."""
    scored = [(float(policy.priority(*point)), i) for i, point in enumerate(_GRID)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in scored]


def _reference_ordering() -> List[int]:
    """The shipped seed's own ordering of the grid -- the baseline to beat."""
    seed = priority_selection().compile(PRIORITY_SEED)
    return _ordering(seed)


def _kendall_tau(a: Sequence[int], b: Sequence[int]) -> float:
    """Normalised Kendall tau of two orderings, in ``[-1, 1]``.

    Each ordering is a permutation of the same item ids (best first); the
    statistic compares, for every item pair, whether both orderings agree on
    which of the two ranks higher. Identical orderings score 1, exact reverses
    -1."""
    n = len(a)
    rank_a = {item: pos for pos, item in enumerate(a)}
    rank_b = {item: pos for pos, item in enumerate(b)}
    pairs = ((x, y) for x in range(n) for y in range(x + 1, n))
    pos = sum(1 for x, y in pairs
              if (rank_a[x] < rank_a[y]) == (rank_b[x] < rank_b[y]))
    total = n * (n - 1) / 2
    return (2 * pos / total) - 1.0 if total else 0.0


def priority_grid_problem(
    *,
    reference: Optional[Sequence[int]] = None,
) -> Problem:
    """A :class:`~agentdescent.meta.Problem` over the priority grid.

    ``(policy, seed) -> MetaOutcome`` where the outcome's reward is the
    policy's ordering agreement with ``reference`` (the shipped seed's own by
    default, normalised to ``[0, 1]``) and ``detail`` carries ``ops`` (the
    candidate's source cost) and ``wallclock``. The grid itself is fixed: the
    module-level ``_GRID`` is the regime, so there is no ``points`` knob.
    ``reference`` must be a permutation of the grid indices ``0..n-1``.
    """
    ref = list(reference) if reference is not None else _reference_ordering()
    if sorted(ref) != list(range(len(_GRID))):
        raise ValueError("reference must be a permutation of the grid indices "
                         f"0..{len(_GRID) - 1}")

    def problem(policy: Any, seed: int) -> MetaOutcome:
        order = _ordering(policy)
        quality = max(0.0, (_kendall_tau(order, ref) + 1.0) / 2.0)
        source = getattr(policy, "source", "")
        return MetaOutcome(curve=[quality, quality], final=quality,
                           rollouts=len(order),
                           detail={"ops": source_ops(source)})

    return problem


def cost_adjusted_reward(
    lam: float = 0.05,
    baseline: float = 1.0,
    cost_key: str = "ops",
) -> MetaReward:
    """``quality - lam * (cost / baseline)``, clipped to ``[0, 1]``.

    At the baseline's own cost this scores ``quality - lam``, so a candidate
    must be *cheaper at equal quality* (or better at equal cost) to outscore
    it. The cost is read from the outcome's ``detail[cost_key]``."""

    def reward(outcome: MetaOutcome) -> float:
        quality = min(1.0, max(0.0, auc(outcome)))
        cost = float(outcome.detail.get(cost_key, baseline))
        value = quality - lam * (cost / max(float(baseline), 1e-9))
        return min(1.0, max(0.0, value))

    return reward


@dataclass(frozen=True)
class FastRuleReport:
    """How an evolved rule does against the shipped seed, on both axes."""

    seed_value: str
    value: str
    seed_ops: int
    evolved_ops: int
    seed_quality: float
    evolved_quality: float
    #: The cost-adjusted gain on the validation problems: positive means the
    #: evolved rule beats the shipped seed on quality-per-cost.
    gain: float


def evolve_fast_rule(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    lam: float = 0.05,
    seeds: Sequence[int] = (0, 1, 2, 3),
    val_seeds: Sequence[int] = (4, 5, 6, 7),
    seed_source: str = PRIORITY_SEED,
    **evolve_kwargs: Any,
) -> Tuple[FastRuleReport, EvolutionResult]:
    """Evolve a ``selection`` priority rule on quality-per-cost, beat the seed.

    ``meta_evolve`` with ``spec=priority_selection()``, the problems scoring
    grid agreement and the reward :func:`cost_adjusted_reward`; then the
    evolved rule and the shipped seed are both scored on ``val_problems`` on
    both axes, and :class:`FastRuleReport` says which won on quality-per-cost.

    ``problems`` / ``val_problems`` may be a mapping or a sequence; a sequence
    is normalised to ``{"p0": ...}`` before use, so every problem is always
    scored by name and ``val_problems`` never crashes on ``.values()``.
    """
    if propose is None and model is None:
        raise ValueError("evolve_fast_rule() needs propose= or model=")
    problems = _as_problem_map(problems)
    val_problems = _as_problem_map(val_problems)
    spec = priority_selection(seed=seed_source)
    baseline = source_ops(seed_source)
    result = meta_evolve(
        problems, slot="selection", spec=spec, propose=propose, model=model,
        meta_reward=cost_adjusted_reward(lam=lam, baseline=baseline),
        seeds=seeds, **evolve_kwargs)
    value = result.rendered or spec.render(spec.initial())
    seed_value = spec.render(spec.initial())

    seed_policy = spec.compile(seed_value)
    evolved_policy = spec.compile(value)
    seed_quality = _mean_quality(seed_policy, val_problems, val_seeds)
    evolved_quality = _mean_quality(evolved_policy, val_problems, val_seeds)
    seed_ops = source_ops(seed_value)
    evolved_ops = source_ops(value)
    gain = (evolved_quality - lam * (evolved_ops / baseline)) \
        - (seed_quality - lam * (seed_ops / baseline))

    report = FastRuleReport(
        seed_value=seed_value, value=value,
        seed_ops=seed_ops, evolved_ops=evolved_ops,
        seed_quality=seed_quality, evolved_quality=evolved_quality, gain=gain)
    try:
        result.meta = {
            "seed_ops": seed_ops, "evolved_ops": evolved_ops,
            "seed_quality": seed_quality, "evolved_quality": evolved_quality,
            "cost_adjusted_gain": gain,
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


def _mean_quality(policy: Any, problems: Mapping[str, Problem],
                  seeds: Sequence[int]) -> float:
    total = 0.0
    n = 0
    for problem in problems.values():
        for seed in seeds:
            total += float(auc(problem(policy, seed)))
            n += 1
    return total / n if n else 0.0
