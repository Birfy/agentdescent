# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is adapted from google-research/rrsi/rrsi/selection.py at
# be50316e1db05914068a973f322770ef08ed7ba1. It is distributed under the
# Apache License, Version 2.0.
"""Validated, offline implementation of the RRSI round-selection mechanism.

This module intentionally contains only Algorithm 2's score/cost selection
step. It is not a complete RRSI run loop or an AgentDescent ``evolve`` adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


STRUCTURAL_COMPONENTS = frozenset({"client_tool", "skill", "memory", "subagent"})


@dataclass(frozen=True)
class Measurement:
    """One candidate's aggregate score and policy-token estimate."""

    score: float
    cost: Optional[float]


@dataclass(frozen=True)
class SelectionConfig:
    """Algorithm 2 parameters from an RRSI instance configuration."""

    beta0: float
    beta1: float
    w_s: float
    w_c: float
    w_n: float


@dataclass(frozen=True)
class Candidate:
    """A candidate with component tags and pre-evaluation screening result."""

    variant: str
    components: Tuple[str, ...] = ()
    measurement: Optional[Measurement] = None
    gate_failure: Optional[str] = None


@dataclass
class Decision:
    variant: str
    admissible: bool
    reason: str
    score: Optional[float] = None
    cost: Optional[float] = None
    delta_score: Optional[float] = None
    delta_cost: Optional[float] = None
    novelty: int = 0
    guards: List[str] = field(default_factory=list)


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_measurement(value: Optional[Measurement]) -> bool:
    if value is None or not _finite_number(value.score) or not 0.0 <= value.score <= 1.0:
        return False
    return value.cost is None or (
        _finite_number(value.cost) and value.cost >= 0.0
    )


def _valid_config(config: SelectionConfig, delta: float) -> bool:
    return (
        _finite_number(delta)
        and 0.0 <= delta <= 1.0
        and all(_finite_number(v) for v in (
            config.beta0, config.beta1, config.w_s, config.w_c, config.w_n
        ))
        and config.beta0 >= 0.0
        and config.beta1 >= 0.0
        and config.w_s >= 0.0
        and config.w_c >= 0.0
        and config.w_n >= 0.0
    )


def _novelty(components: Sequence[str], incumbent_counts: Dict[str, int]) -> int:
    # Set semantics and default-zero counts mirror the pinned upstream helper.
    return sum(1 for name in set(components)
               if name in STRUCTURAL_COMPONENTS and incumbent_counts.get(name, 0) == 0)


def _relative_cost_change(candidate: Optional[float], incumbent: Optional[float]) -> float:
    """Match pinned upstream: if either token estimate is unavailable/zero, dC=0."""
    if candidate is None or incumbent is None or candidate == 0 or incumbent == 0:
        return 0.0
    # This rearrangement avoids overflowing candidate - incumbent when both
    # token counts are large. A non-finite ratio is rejected by the caller.
    return candidate / incumbent - 1.0


def _cost_rule(delta_score: float, delta_cost: float, novelty: int,
               delta: float, config: SelectionConfig) -> Tuple[bool, str]:
    if delta_score > delta:
        budget = config.beta0 + config.beta1 * delta_score
        if not math.isfinite(budget):
            return False, "invalid cost budget (non-finite arithmetic)"
        ok = delta_cost <= budget
        return ok, "large-gain cost rule passed" if ok else "large-gain cost rule failed"

    shaped = (config.w_s * delta_score - config.w_c * delta_cost
              + config.w_n * novelty)
    if not math.isfinite(shaped):
        return False, "invalid within-band value (non-finite arithmetic)"
    ok = shaped > 0.0
    return ok, "within-band rule passed" if ok else "within-band rule failed"


def judge(candidate: Candidate, incumbent: Measurement, best_score: float,
          delta: float, config: SelectionConfig,
          incumbent_counts: Dict[str, int],
          guards: Optional[Sequence[str]] = None) -> Decision:
    """Evaluate one candidate against the incumbent and historical-best floor.

    Invalid, non-finite and out-of-domain measurements are explicitly rejected.
    A critic/smoke veto remains decisive even if a measurement was attached.
    """
    rejected = candidate.gate_failure
    if rejected:
        return Decision(candidate.variant, False, rejected)
    if not _valid_measurement(incumbent):
        return Decision(candidate.variant, False, "invalid incumbent measurement")
    if not _finite_number(best_score) or not 0.0 <= best_score <= 1.0:
        return Decision(candidate.variant, False, "invalid historical-best score")
    if not _valid_config(config, delta):
        return Decision(candidate.variant, False, "invalid selection configuration")
    measurement = candidate.measurement
    if not _valid_measurement(measurement):
        return Decision(candidate.variant, False, "invalid or missing candidate measurement")

    assert measurement is not None and incumbent is not None
    score_delta = measurement.score - incumbent.score
    cost_delta = _relative_cost_change(measurement.cost, incumbent.cost)
    if not math.isfinite(score_delta) or not math.isfinite(cost_delta):
        return Decision(candidate.variant, False, "invalid measurement arithmetic")

    nov = _novelty(candidate.components, incumbent_counts)
    decision = Decision(candidate.variant, False, "", score=measurement.score,
                        cost=measurement.cost, delta_score=score_delta,
                        delta_cost=cost_delta, novelty=nov,
                        guards=list(guards or ()))
    floor = best_score - delta
    if measurement.score < floor:
        decision.reason = "below historical-best noise floor"
        return decision

    ok, why = _cost_rule(score_delta, cost_delta, nov, delta, config)
    if not ok:
        decision.reason = why
        return decision
    if decision.guards:
        decision.reason = "domain guard violated: " + "; ".join(decision.guards)
        return decision
    decision.admissible = True
    decision.reason = why
    return decision


def select_round(candidates: Sequence[Candidate], incumbent: Measurement,
                 best_score: float, delta: float, config: SelectionConfig,
                 incumbent_counts: Dict[str, int],
                 guard_fn: Optional[Callable[[Measurement, Optional[Measurement]], Sequence[str]]] = None
                 ) -> Tuple[Optional[Candidate], List[Decision]]:
    """Return the highest-scoring admissible candidate; stable ties keep first.

    All candidates represent alternatives from one incumbent. The caller owns
    evaluation, screening, history persistence and updating the best score.
    """
    decisions: List[Decision] = []
    for candidate in candidates:
        guards: Sequence[str] = ()
        # Guards may perform arithmetic on both measurements. Preserve judge's
        # invalid-input rejection contract by only calling them after both
        # values pass the same validation used by judge.
        if (guard_fn and not candidate.gate_failure
                and _valid_measurement(incumbent)
                and _valid_measurement(candidate.measurement)):
            guards = guard_fn(incumbent, candidate.measurement)
        decisions.append(judge(candidate, incumbent, best_score, delta, config,
                               incumbent_counts, guards))

    winner: Optional[Candidate] = None
    winning_score = -math.inf
    for candidate, decision in zip(candidates, decisions):
        if decision.admissible and decision.score is not None and decision.score > winning_score:
            winner, winning_score = candidate, decision.score
    return winner, decisions


def update_best_score(best_score: float, winner: Optional[Candidate]) -> float:
    """Keep S* monotone across rounds; a no-winner round preserves it."""
    if not _finite_number(best_score) or not 0.0 <= best_score <= 1.0:
        raise ValueError("best_score must be finite and in [0, 1]")
    if winner is None:
        return best_score
    if not _valid_measurement(winner.measurement):
        raise ValueError("winner must have a valid measurement")
    assert winner.measurement is not None
    return max(best_score, winner.measurement.score)
