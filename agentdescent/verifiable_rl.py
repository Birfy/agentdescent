"""Verifiable-reward policy improvement: the framework trains on truth it can check.

DeepSeek-R1 and RAGEN improve a policy with RL over trajectories where the
reward is **verifiable** — exact match, a checker, a held-out fact — not a
learned proxy. The meta stack here defaults to ``auc`` (a proxy); this module
makes the verifiable reward first-class: :func:`verifiable_problem` wraps a
``(value, task) -> output`` problem so its outcome carries the output and the
gold in ``detail``, :func:`verifiable_reward` scores exact match (1.0 / 0.0 —
a rule that cannot be checked is not rewarded), and :func:`policy_improve`
evolves a decision slot with that reward, reporting the per-round verifiable
reward curve (the trajectory-level signal RL trains on) and the before/after on
held-out verifiable facts.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evolution import EvolutionResult, Task
from .meta import MetaOutcome, MetaReward, SlotSpec, meta_evolve

__all__ = ["verifiable_problem", "verifiable_reward", "policy_improve",
           "VerifiableReport"]


def verifiable_problem(
    problem_fn: Callable[[Any, Task], Any],
    gold: Callable[[Task], Any],
    *,
    task_for: Callable[[int], Task],
) -> Callable[[Any, int], MetaOutcome]:
    """Wrap a ``(value, task) -> output`` problem so the outcome's reward is
    verifiable: ``1.0`` on exact match with ``gold``, ``0.0`` otherwise, with
    the output and gold carried in ``detail`` for :func:`verifiable_reward`."""

    def wrapped(value: Any, seed: int) -> MetaOutcome:
        task = task_for(seed)
        output = problem_fn(value, task)
        gold_value = gold(task)
        ok = str(output).strip() == str(gold_value).strip()
        return MetaOutcome(curve=[1.0 if ok else 0.0],
                           final=1.0 if ok else 0.0, rollouts=1,
                           detail={"output": output, "gold": gold_value})

    return wrapped


def verifiable_reward() -> MetaReward:
    """The RL signal: ``1.0`` when the outcome's output exactly matches its
    ``gold``, ``0.0`` otherwise. A rule that cannot be checked is not rewarded
    -- the DeepSeek-R1 verifiability contract."""

    def reward(outcome: MetaOutcome) -> float:
        detail = outcome.detail or {}
        output, gold_value = detail.get("output"), detail.get("gold")
        if output is None or gold_value is None:
            return 0.0
        return 1.0 if str(output).strip() == str(gold_value).strip() else 0.0

    return reward


@dataclass(frozen=True)
class VerifiableReport:
    """What :func:`policy_improve` did: the per-round held-out verifiable
    reward curve (the trajectory-level signal RL trains on) and the
    before/after on held-out verifiable facts."""

    value: str
    reward_curve: List[float]
    seed_reward: float
    improved_reward: float


def policy_improve(
    problems: Mapping[str, Callable],
    val_problems: Mapping[str, Callable],
    *,
    slot: str,
    spec: SlotSpec,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    seeds: Sequence[int] = (0, 1, 2, 3),
    val_seeds: Sequence[int] = (4, 5),
    **evolve_kwargs: Any,
) -> Tuple[VerifiableReport, EvolutionResult]:
    """Evolve a decision slot on verifiable reward and report the curve.

    ``meta_evolve`` with :func:`verifiable_reward` as the meta reward — every
    rollout's outcome is checked against its gold, so only verifiable progress
    counts. The report carries the per-round mean verifiable reward (the
    trajectory-level curve) plus the before/after on the held-out verifiable
    facts.
    """
    if propose is None and model is None:
        raise ValueError("policy_improve() needs propose= or model=")
    vr = verifiable_reward()
    result = meta_evolve(
        problems, slot=slot, spec=spec, propose=propose, model=model,
        meta_reward=vr, seeds=seeds, **evolve_kwargs)
    value = result.rendered or spec.render(spec.initial())
    seed_value = spec.render(spec.initial())

    # before/after on the held-out verifiable facts, never the training set
    seed_reward = _mean_verifiable(spec, seed_value, val_problems, val_seeds)
    improved_reward = _mean_verifiable(spec, value, val_problems, val_seeds)
    curve = [round(float(h.held_out_reward), 4) for h in result.history]
    report = VerifiableReport(
        value=value, reward_curve=curve,
        seed_reward=seed_reward, improved_reward=improved_reward)
    try:
        result.meta = {"seed_reward": seed_reward,
                       "improved_reward": improved_reward,
                       "reward_curve": curve}
    except AttributeError:
        pass
    return report, result


def _mean_verifiable(spec: SlotSpec, rendered: str,
                     problems: Mapping[str, Callable],
                     seeds: Sequence[int]) -> float:
    total = 0.0
    n = 0
    try:
        value = spec.compile(rendered)
    except ValueError:
        return 0.0
    vr = verifiable_reward()
    for problem in problems.values():
        for seed in seeds:
            total += float(vr(problem(value, seed)))
            n += 1
    return total / n if n else 0.0
