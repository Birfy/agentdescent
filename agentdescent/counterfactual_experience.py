"""Turn counterfactual credit into experience the meta-policy can replay.

`counterfactual.ReplayLab` attributes a score to individual trace steps;
`experience.ExperienceEngine` learns from per-action rewards. Nothing converted
one into the other. This module does that conversion:

    trace -> ReplayLab.shapley -> {edit_id: credit}
          -> ExperienceRecord -> ExperienceEngine.ingest

A step's Shapley credit becomes its reward; credits below ``min_credit`` are
dropped (they did not help), and an empty reward map is refused instead of
silently creating a zero-information experience. The engine remains the only
replay/update owner.

    bridge = CounterfactualExperience(lab, engine)
    report = bridge.ingest(trace, features=(0.2, 0.8))
    report.record.rewards, report.credit.total_credit
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .counterfactual import ReplayLab, ReplayTrace, TraceStep
from .credit import CreditReport
from .experience import ExperienceEngine, ExperienceRecord, ExperienceUpdate

__all__ = ["CounterfactualExperienceReport", "CounterfactualExperience"]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class CounterfactualExperienceReport:
    """The credit report, the record it produced and where it was ingested."""

    record: ExperienceRecord
    credit: CreditReport
    observed: int
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "credit": {
                "method": self.credit.method,
                "gain": self.credit.gain,
                "credits": [
                    {"edit_id": row.edit_id, "credit": row.credit,
                     "share": row.share}
                    for row in self.credit.credits
                ],
            },
            "observed": self.observed,
            "detail": self.detail,
        }


class CounterfactualExperience:
    """Convert Shapley step credit into an `ExperienceRecord` and ingest it."""

    def __init__(self, lab: ReplayLab, engine: ExperienceEngine,
                 action_of: Callable[[TraceStep], str], *,
                 min_credit: Optional[float] = None,
                 require_positive_gain: bool = False,
                 permutations: int = 64, seed: int = 0,
                 exact_limit: int = 7) -> None:
        if not isinstance(lab, ReplayLab):
            raise TypeError("lab must be a ReplayLab")
        if not isinstance(engine, ExperienceEngine):
            raise TypeError("engine must be an ExperienceEngine")
        if not callable(action_of):
            raise TypeError("action_of must be callable")
        threshold = None if min_credit is None else _finite(min_credit, "min_credit")
        if not isinstance(require_positive_gain, bool):
            raise TypeError("require_positive_gain must be a bool")
        if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
            raise ValueError("permutations must be an int >= 1")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an int")
        if isinstance(exact_limit, bool) or not isinstance(exact_limit, int) or exact_limit < 0:
            raise ValueError("exact_limit must be an int >= 0")
        self.lab = lab
        self.engine = engine
        self.action_of = action_of
        self.min_credit = threshold
        self.require_positive_gain = require_positive_gain
        self.permutations = permutations
        self.seed = seed
        self.exact_limit = exact_limit

    def ingest(self, trace: ReplayTrace, features: Sequence[float], *,
               accepted: bool = True, cost: float = 0.0,
               artifact_id: str = "",
               metadata: Optional[Mapping[str, Any]] = None) -> CounterfactualExperienceReport:
        if not isinstance(trace, ReplayTrace):
            raise TypeError("trace must be a ReplayTrace")
        if not trace.steps:
            raise ValueError("trace has no steps to attribute")
        cost = _finite(cost, "cost")
        if cost < 0:
            raise ValueError("cost must be >= 0")
        if len(features) != self.engine.learner.policy.dim:
            raise ValueError(
                f"features have {len(features)} dimension(s), learner expects "
                f"{self.engine.learner.policy.dim}")
        credit = self.lab.shapley(
            trace, permutations=self.permutations, seed=self.seed,
            exact_limit=self.exact_limit)
        if self.require_positive_gain and credit.gain <= 0:
            raise ValueError(f"trace gain {credit.gain:.6g} is not positive")
        by_id = {step.step_id: step for step in trace.steps}
        rewards: Dict[str, float] = {}
        for row in credit.credits:
            if self.min_credit is not None and row.credit < self.min_credit:
                continue
            action = self.action_of(by_id[row.edit_id])
            if action not in self.engine.learner.actions:
                raise ValueError(
                    f"step {row.edit_id!r} maps to action {action!r}, which is "
                    f"not in learner actions {self.engine.learner.actions}")
            rewards[action] = rewards.get(action, 0.0) + float(row.credit)
        if not rewards:
            raise ValueError("no step credit survived the filter")
        data = dict(metadata or {})
        data.update({"shapley_method": credit.method,
                     "baseline": credit.baseline, "full": credit.full,
                     "samples": credit.samples,
                     "gain": credit.gain})
        record = ExperienceRecord(
            run_id=trace.trace_id, features=features, rewards=rewards,
            cost=cost, accepted=accepted, artifact_id=str(artifact_id),
            metadata=data)
        self.engine.ingest(record)
        detail = (f"{len(record.rewards)} action reward(s), "
                  f"gain={credit.gain:+.4f}, replay={len(self.engine.store)}")
        return CounterfactualExperienceReport(
            record=record, credit=credit,
            observed=len(self.engine.store), detail=detail)

    def update(self, *, validation: Sequence[ExperienceRecord],
               anchors: Sequence[ExperienceRecord]) -> ExperienceUpdate:
        return self.engine.update(validation=validation, anchors=anchors)
