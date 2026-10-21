"""Self red-teaming: the framework attacks its own value to find the OOD gap.

Red-teaming / adversarial search finds the inputs where a system fails; a
self-improving framework should do this to *itself* before anyone else does.
This module is that: ``red_team`` mutates the training tasks outward and scores
the current value on the mutations, collecting the failures -- the
out-of-distribution region where the value does not generalise; and
``red_team_evolve`` runs the ordinary ``evolve()``, attacks its own result, and
retrains on the failures it found, closing the OOD gap. The report shows the
attack's findings (how many failures, at what distance) and the coverage before
and after the retraining, so "the framework found its own blind spot and fixed
it" is measured, not claimed.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evolution import EvolutionResult, Task, evolve
from .meta import MetaOutcome, SlotSpec

__all__ = ["red_team", "RedTeamReport", "red_team_evolve"]


def red_team(
    value: Any,
    score: Callable[[Any, Task], float],
    mutate: Callable[[Task, int], Task],
    seed_tasks: Sequence[Task],
    *,
    budget: int = 20,
    solved: float = 0.999,
    seeds: Sequence[int] = (0,),
) -> List[Tuple[Task, float]]:
    """Attack ``value`` by mutating the seed tasks outward and scoring each
    mutation. Returns the ``(task, score)`` failures -- the OOD region where
    the value does not generalise. A deterministic attacker: the same mutations
    in the same order, so the attack is reproducible."""
    results = _attack_results(value, score, mutate, seed_tasks, budget, seeds)
    return [(task, s) for task, s in results if s < solved]


def _attack_results(
    value: Any,
    score: Callable[[Any, Task], float],
    mutate: Callable[[Task, int], Task],
    seed_tasks: Sequence[Task],
    budget: int,
    seeds: Sequence[int],
    start: int = 0,
) -> List[Tuple[Task, float]]:
    """Every probe of one attack pass -- successes and failures alike -- so a
    caller can measure coverage over the *complete* attack set. ``start``
    offsets the mutation indices, for a fresh probe set that a run never
    retrained on."""
    out: List[Tuple[Task, float]] = []
    for k in range(budget):
        i = start + k
        task = mutate(seed_tasks[i % len(seed_tasks)], i)
        s = max(score(value, task) for _ in seeds)
        out.append((task, s))
    return out


@dataclass(frozen=True)
class RedTeamReport:
    """What a :func:`red_team_evolve` round did: the attack's findings and the
    coverage before and after the retraining."""

    value: str
    failures_found: int
    coverage_before: float
    coverage_after: float


def red_team_evolve(
    problems: Mapping[str, Task],
    *,
    run: Callable[[str, Task], Any],
    reward: Callable[[Task, Any], float],
    propose: Callable[[str, Task, Any, float], Optional[str]],
    mutate: Callable[[Task, int], Task],
    strategy: Any,
    solved: float = 0.999,
    attack_budget: int = 20,
    retrain_rounds: int = 2,
    **evolve_kwargs: Any,
) -> Tuple[RedTeamReport, EvolutionResult]:
    """Evolve, attack the result, retrain on the failures it found.

    Phase 1 runs ``evolve()`` on ``problems`` (at least four). Phase 2
    red-teams the result: ``mutate`` perturbs the seed tasks outward and
    ``score``-s the value, collecting the failures. Phase 3 retrains on the
    union of the original problems and the failures for ``retrain_rounds``
    rounds, **starting from phase 1's evolved state** (unless the caller
    supplied an ``initial_state``), so retraining refines the attacked result
    instead of restarting from the seed. The report shows the failures found
    and the coverage -- the fraction of a probe set the value solves, at
    ``solved`` -- before (on the attack that found the failures) and after (on
    a **fresh** attack the run never retrained on), so memorising the failures
    does not count as closing the OOD gap.
    """
    if len(problems) < 4:
        raise ValueError("red_team_evolve() needs at least 4 problems to "
                         "split train/held-out")
    tasks = list(problems.values())

    def run_task(rendered: str, task: Task) -> Any:
        return run(rendered, task)

    def reward_fn(task: Task, output: Any) -> float:
        return float(reward(task, output))

    def score_of(rendered: str, task: Task) -> float:
        return float(reward(task, run(rendered, task)))

    kwargs = dict(evolve_kwargs)
    first_rounds = int(kwargs.pop("rounds", 1))
    first = evolve(tasks, reward_fn, run=run_task, propose=propose,
                   strategy=strategy, rounds=first_rounds, **kwargs)
    value = first.rendered or strategy.render(strategy.initial())

    attack = _attack_results(value, score_of, mutate, tasks,
                             attack_budget, (0,))
    failures = [(task, s) for task, s in attack if s < solved]
    probe_tasks = [task for task, _ in attack]
    coverage_before = _coverage(value, probe_tasks, score_of, solved)
    extended = tasks + [task for task, _ in failures]

    second_kwargs = dict(kwargs)
    if "initial_state" not in second_kwargs:
        second_kwargs["initial_state"] = first.state
    second = evolve(extended, reward_fn, run=run_task, propose=propose,
                    strategy=strategy, rounds=retrain_rounds, **second_kwargs)
    value = second.rendered or strategy.render(strategy.initial())
    # a FRESH attack on the retrained value: probes it never retrained on, so
    # memorising the failures does not count as closing the OOD gap
    fresh = _attack_results(value, score_of, mutate, tasks,
                            attack_budget, (0,), start=attack_budget)
    coverage_after = _coverage(value, [task for task, _ in fresh],
                               score_of, solved)

    report = RedTeamReport(
        value=value, failures_found=len(failures),
        coverage_before=coverage_before, coverage_after=coverage_after)
    try:
        second.meta = {"failures_found": len(failures),
                       "coverage_before": coverage_before,
                       "coverage_after": coverage_after}
    except AttributeError:
        pass
    return report, second


def _coverage(value: str, probe_tasks: Sequence[Task],
              score_of: Callable[[str, Task], float],
              solved: float) -> float:
    """Fraction of the attack's probe tasks the value solves, at ``solved``.
    A set with no probes is vacuously covered (1.0)."""
    if not probe_tasks:
        return 1.0
    solved_now = sum(1 for t in probe_tasks if score_of(value, t) >= solved)
    return solved_now / len(probe_tasks)
