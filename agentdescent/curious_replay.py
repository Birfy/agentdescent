"""Failure-weighted replay: the framework replays the past it is still failing.

Curious Replay (ICLR 2025) replays under-explored regions of experience,
weighted by learning progress, for model-based adaptation. The trajectory
ledger records every past rollout; nothing here ever re-examined them. This
module makes the past an explicit training set with **failure weights**: a
replayed example carries a weight that is high while the artifact is still
failing it (its mean score is far below the ceiling) and low once it is
mastered -- so the evolution spends its learning signal on the past it has not
yet conquered. (This is a *failure/need* signal, not learning-progress: a
permanently stuck example stays heavily weighted until it is solved.)

``curious_replay_evolve`` runs the ordinary ``evolve()`` over replay tasks --
one task per replayed example, weighted by its **snapshot** weight, with the
mastered and still-curious cohorts interleaved so both land on both sides of
the held-out cut -- and the report shows the weights and the improvement on
the mastered vs still-curious past.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evolution import EvolutionResult, Task, evolve
from .meta import MetaOutcome, SlotSpec, slot_reflector

__all__ = ["ReplayExample", "curiosity_weight", "curious_replay_evolve",
           "CuriousReplayReport"]


@dataclass(frozen=True)
class ReplayExample:
    """One replayed past example: the task it came from, its recorded outcome,
    and the rolling statistics that decide how curious the artifact still is."""

    task: Any
    #: The artifact's scores on this task so far: the spread is the uncertainty
    #: that drives curiosity.
    scores: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def uncertainty(self) -> float:
        """The spread of the artifact's own scores on this example -- high
        while still learning it, low once mastered (or permanently stuck)."""
        if len(self.scores) < 2:
            return 1.0
        return statistics.pstdev(self.scores)

    @property
    def mastered(self) -> bool:
        return bool(self.scores) and self.mean >= 0.999

    def record(self, score: float) -> None:
        self.scores.append(float(score))


def curiosity_weight(example: ReplayExample, *, floor: float = 0.05) -> float:
    """How much replay weight this example should carry -- **failure-weighted**:
    high while the artifact is still failing it (mean far below the ceiling),
    falling as it improves, floor once mastered. ``floor`` keeps a mastered
    example replayable without dominating the still-curious past. (This is a
    failure/need signal, not learning-progress: a permanently stuck example
    stays heavily weighted until it is solved.)"""
    if example.mastered:
        return floor
    return floor + (1.0 - floor) * (1.0 - example.mean)


def _interleave(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Alternate two index lists so a positional held-out cut sees both."""
    out: List[int] = []
    i = j = 0
    while i < len(a) or j < len(b):
        if i < len(a):
            out.append(a[i]); i += 1
        if j < len(b):
            out.append(b[j]); j += 1
    return out


@dataclass(frozen=True)
class CuriousReplayReport:
    """What :func:`curious_replay_evolve` did: the weights, the gains on the
    still-curious vs mastered past, and the value produced."""

    value: str
    weights: Dict[str, float]
    curious_gain: float
    mastered_gain: float


def curious_replay_evolve(
    examples: Sequence[ReplayExample],
    *,
    spec: SlotSpec,
    run: Callable[[str, ReplayExample], float],
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    seeds: Sequence[int] = (0, 1),
    floor: float = 0.05,
    **evolve_kwargs: Any,
) -> Tuple[CuriousReplayReport, EvolutionResult]:
    """Evolve the artifact on its own past, weighted by curiosity.

    One task per replayed example, whose **held-out gate scores raw
    performance** -- forgetting the mastered past costs its full unweighted
    penalty, so a candidate that trades mastered examples for curious ones is
    rejected. The snapshot curiosity weight rides in ``task.meta["weight"]`` as
    the reflector/proposer signal. The cohorts, weights and per-example
    baselines are snapshotted **before** any candidate rolls out (the run
    re-scores and records into the examples, so the *next* call's curiosity
    reflects this run, but the report and the held-out gate must not drift with
    it). One task per example keeps ``evolve()``'s positional held-out split
    over **distinct** examples, and the task list **interleaves the mastered
    and still-curious cohorts** so both land on both sides of the cut -- the
    held-out set cannot be curious-only, so a candidate that forgets the
    mastered past is rejected by the gate. The report shows the snapshot
    weights and the gain on the still-curious vs mastered past, so "the replay
    helped where it was curious" is measured."""
    if propose is None and model is None:
        raise ValueError("curious_replay_evolve() needs propose= or model=")
    baseline = {i: e.mean for i, e in enumerate(examples)}
    mastered = {i: e.mastered for i, e in enumerate(examples)}
    weights = {i: curiosity_weight(e, floor=floor) for i, e in enumerate(examples)}
    reflector = propose if propose is not None else slot_reflector(model, spec)
    seed_value = spec.render(spec.initial())
    order = _interleave([i for i in range(len(examples)) if not mastered[i]],
                        [i for i in range(len(examples)) if mastered[i]])
    tasks = [Task(id=f"replay:{i}", prompt=f"replay example {i}",
                  meta={"weight": weights[i], "index": i})
             for i in order]

    def reward(task: Task, output: str) -> float:
        # unweighted: the held-out gate must treat every example equally, or
        # a mastered example (weight ~0.05) is cheap to forget
        try:
            return float(MetaOutcome.from_json(output).final)
        except (TypeError, ValueError, KeyError):
            return 0.0

    def run_task(rendered: str, task: Task) -> str:
        idx = int(task.meta["index"])
        score = float(run(rendered, examples[idx]))
        examples[idx].record(score)
        return MetaOutcome(
            curve=[score], final=score, rollouts=1).to_json()

    result = evolve(
        tasks, reward, run=run_task, propose=reflector, strategy=spec,
        initial_state=spec.initial(), self_verify=False, solved_threshold=1.1,
        **evolve_kwargs)
    value = result.rendered or seed_value
    curious_idx = [i for i in range(len(examples)) if not mastered[i]]
    mastered_idx = [i for i in range(len(examples)) if mastered[i]]
    curious_gain = _gain(value, [(i, baseline[i]) for i in curious_idx],
                         examples, run)
    mastered_gain = _gain(value, [(i, baseline[i]) for i in mastered_idx],
                          examples, run)
    report = CuriousReplayReport(
        value=value,
        weights={e.task if isinstance(e.task, str) else str(i): w
                 for i, e in enumerate(examples) for w in (weights[i],)},
        curious_gain=curious_gain, mastered_gain=mastered_gain)
    try:
        result.meta = {"curious_gain": curious_gain,
                       "mastered_gain": mastered_gain}
    except AttributeError:
        pass
    return report, result


def _gain(value: str, pairs: Sequence[Tuple[int, float]],
          examples: Sequence[ReplayExample],
          run: Callable[[str, ReplayExample], float]) -> float:
    """The mean ``run(value, example) - baseline`` over a cohort, where the
    baseline is the example's **pre-run** mean, not the post-run one."""
    if not pairs:
        return 0.0
    gains = [float(run(value, examples[i])) - base for i, base in pairs]
    return statistics.fmean(gains)
