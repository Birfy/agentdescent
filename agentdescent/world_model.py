"""World model: the framework learns an internal simulator and dreams before it acts.

Dreamer / world-model RL train a simulator from experience and use it for cheap
rollouts, spending real cost only where it pays. Every evolution run in this
framework pays the real cost of every rollout; nothing learns an internal model
of its domain. This module is the Dreamer loop applied to the framework: a
:class:`LearnedSimulator` is fit on observed ``(state, reward)`` pairs, a
:class:`~agentdescent.meta.Problem` can run on the simulator instead of the real
inner run, and ``imagine_then_evolve`` spends most of its budget in the dream —
evolving against the simulator — then verifies the result on the real
problem. ``simulator_score`` measures the simulator against the real problem,
so the framework knows when to trust the dream and when to wake up.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from .evolution import EvolutionResult, Task, evolve
from .meta import (
    MetaOutcome, Problem, SlotSpec, auc, meta_evolve,
    meta_validate, slot_reflector,
)

__all__ = ["LearnedSimulator", "simulator_problem", "simulator_score",
           "ImagineReport", "imagine_then_evolve"]


@dataclass
class LearnedSimulator:
    """A pure-Python nearest-neighbour simulator from observed rollouts.

    ``fit((features, reward))`` stores the examples; ``predict(features)``
    returns the mean reward of the ``k`` nearest examples (Euclidean) -- a
    deterministic, dependency-free model that needs no linear algebra and
    cannot extrapolate beyond what it has seen, which is exactly the honesty a
    simulator of a measured domain should have.
    """

    k: int = 1
    examples: List[Tuple[List[float], float]] = field(default_factory=list)

    def fit(self, pairs: Sequence[Tuple[List[float], float]]) -> None:
        self.examples = [(list(f), float(r)) for f, r in pairs]

    def predict(self, features: Sequence[float]) -> float:
        if not self.examples:
            return 0.0
        feats = list(features)
        dists = sorted(
            ((sum((a - b) ** 2 for a, b in zip(feats, f)), r)
             for f, r in self.examples),
            key=lambda d: d[0])
        k = max(1, min(self.k, len(dists)))
        return sum(r for _, r in dists[:k]) / k

    @property
    def n_examples(self) -> int:
        return len(self.examples)


def _features(value: Any, seed: int) -> List[float]:
    """The simulator's view of a rollout: the value's scalar (a ``cap:N`` or
    ``c: X`` number, read from the rendered policy or its own attribute) and
    the task's difficulty. A caller can supply its own extractor to
    ``simulator_problem``; this is the shipped one for the capability-capped
    and UCB-parameter domains."""
    import re
    scalar = 0.0
    if hasattr(value, "render"):
        rendered = value.render()
    elif hasattr(value, "source"):
        rendered = value.source
    else:
        rendered = str(value)
    m = re.search(r"cap:(\d+)", rendered or "")
    if m:
        scalar = float(m.group(1))
    else:
        m2 = re.search(r"c: ([0-9.]+)", rendered or "")
        if m2:
            scalar = float(m2.group(1))
        elif hasattr(value, "c"):
            scalar = float(value.c)
    difficulty = float(seed % 10)
    return [scalar, difficulty]


def simulator_problem(
    simulator: LearnedSimulator,
    *,
    features: Callable[[Any, int], List[float]] = _features,
    baseline: float = 0.0,
) -> Problem:
    """A :class:`~agentdescent.meta.Problem` that runs the simulator.

    ``(value, seed) -> MetaOutcome``: the value renders, ``features`` extracts
    the simulator's inputs, ``predict`` scores it, and the outcome's reward is
    that predicted quality. Cheap by construction -- no inner run, no model
    call -- which is what makes "dream then act" affordable."""

    def problem(value: Any, seed: int) -> MetaOutcome:
        predicted = simulator.predict(features(value, seed))
        return MetaOutcome(curve=[predicted, predicted], final=predicted,
                           rollouts=1,
                           detail={"predicted": predicted, "seed": seed})

    return problem


def simulator_score(
    simulator: LearnedSimulator,
    real_problem: Problem,
    *,
    value: Any,
    seeds: Sequence[int] = range(10),
    features: Callable[[Any, int], List[float]] = _features,
) -> Dict[str, float]:
    """The simulator against the real problem: how much the dream is worth.

    Scores ``value`` on ``real_problem`` and on the simulator over ``seeds``,
    and returns ``mean_abs_error`` and ``rank_agreement`` (normalised Kendall
    tau of the predicted vs real orderings). Near-zero error and high agreement
    mean the dream can be trusted for this value; otherwise the framework
    should wake up and pay the real cost."""
    real: List[Tuple[float, float]] = []
    for seed in seeds:
        outcome = real_problem(value, seed)
        real.append((simulator.predict(features(value, seed)),
                     float(auc(outcome))))
    errors = [abs(p - r) for p, r in real]
    preds = [p for p, _ in real]
    trues = [r for _, r in real]
    # tau-b: ties count half, so a constant predictor cannot claim perfect trust
    pos = 0.0
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            dp = (preds[i] > preds[j]) - (preds[i] < preds[j])
            dt = (trues[i] > trues[j]) - (trues[i] < trues[j])
            if dp == dt and dp != 0:
                pos += 1.0
            elif dp == 0 or dt == 0:
                pos += 0.5
    total = len(real) * (len(real) - 1) / 2
    return {
        "mean_abs_error": statistics.fmean(errors) if errors else 0.0,
        "rank_agreement": pos / total if total else 0.5,
        "n": len(real),
    }


@dataclass(frozen=True)
class ImagineReport:
    """What :func:`imagine_then_evolve` did: the verified value, its real gain
    over the seed, the simulator's trust score, and how much of the budget was
    spent dreaming vs waking."""

    seed_value: str
    value: str
    oracle_gain: float
    simulator_error: float
    simulator_agreement: float
    dreamed_rollouts: int
    verified_rollouts: int


def imagine_then_evolve(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slot: str,
    spec: SlotSpec,
    simulator: LearnedSimulator,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    seeds: Sequence[int] = (0, 1, 2, 3),
    val_seeds: Sequence[int] = (4, 5),
    **evolve_kwargs: Any,
) -> Tuple[ImagineReport, EvolutionResult]:
    """Evolve against the simulator, verify the result on the real problem.

    ``meta_evolve`` on the simulator-based problems (cheap), then the ``value``
    is scored on the **real** ``val_problems``; the report compares the evolved
    value's real gain over the seed with the simulator's own trust score, so
    "the dream was worth it" is measured, not assumed. ``dreamed_rollouts`` and
    ``verified_rollouts`` are **actual** call counts (simulator ``predict``s in
    the search plus trust measurement, and real problem invocations in
    validation plus trust), not nominal sizes.
    """
    if propose is None and model is None:
        raise ValueError("imagine_then_evolve() needs propose= or model=")
    seed_value = spec.render(spec.initial())
    val_map = (dict(val_problems) if isinstance(val_problems, Mapping)
               else {f"p{i}": p for i, p in enumerate(val_problems)})

    sim_calls: Dict[str, int] = {"n": 0}
    real_calls: Dict[str, int] = {"n": 0}
    val_counted = {name: _count_problem(p, real_calls)
                   for name, p in val_map.items()}
    original_predict = simulator.predict
    simulator.predict = _count_predict(original_predict, sim_calls)
    try:
        dream = {name: simulator_problem(simulator) for name in problems}
        result = meta_evolve(
            dream, slot=slot, spec=spec, propose=propose, model=model,
            seeds=seeds, **evolve_kwargs)
        value = result.rendered or seed_value

        compiled = spec.compile(value)
        trust = simulator_score(simulator, next(iter(val_counted.values())),
                                value=compiled, seeds=val_seeds)
        real_gain = _mean_gain(spec, seed_value, value, val_counted, val_seeds)

        report = ImagineReport(
            seed_value=seed_value, value=value, oracle_gain=real_gain,
            simulator_error=trust["mean_abs_error"],
            simulator_agreement=trust["rank_agreement"],
            dreamed_rollouts=sim_calls["n"],
            verified_rollouts=real_calls["n"])
        try:
            result.meta = {
                "oracle_gain": real_gain,
                "simulator_error": trust["mean_abs_error"],
                "simulator_agreement": trust["rank_agreement"],
                "dreamed_rollouts": sim_calls["n"],
                "verified_rollouts": real_calls["n"],
            }
        except AttributeError:
            pass
        return report, result
    finally:
        simulator.predict = original_predict   # do not leak the counting wrapper


def _count_predict(original: Callable[[Sequence[float]], float],
                   counter: Dict[str, int]) -> Callable[[Sequence[float]], float]:
    """Wrap ``predict`` so every simulator call -- in the dream and in the trust
    measurement -- is counted."""

    def counted(features: Sequence[float]) -> float:
        counter["n"] += 1
        return original(features)

    return counted


def _count_problem(problem: Problem, counter: Dict[str, int]) -> Problem:
    def counted(value: Any, seed: int) -> MetaOutcome:
        counter["n"] += 1
        return problem(value, seed)

    return counted


def _mean_gain(spec: SlotSpec, before: str, after: str,
               val_problems: Mapping[str, Problem], seeds: Sequence[int]) -> float:
    validation = meta_validate(spec, before, after, val_problems,
                               seeds=seeds, meta_reward=auc)
    gains = [row["gain"] for row in validation.values() if row["n"] > 0]
    return statistics.fmean(gains) if gains else 0.0
