"""Replay a session with one decision changed, and measure the difference.

A trajectory tells you what happened. It does not tell you what would have
happened if the agent had skipped a tool, removed an edit, or used a different
context. That counterfactual is the difference between "this run scored 0.7"
and "this edit is why the run scored 0.7".

This module is a thin, deterministic replay lab over a trace:

* a trace is a sequence of :class:`TraceStep` records (edit, tool, context,
  decision);
* a replay evaluator is any ``Sequence[TraceStep] -> float`` callable (a cached
  recorded run, a sandbox, a deterministic simulator);
* an :class:`Intervention` removes, replaces or disables one step;
* :class:`ReplayLab` exposes the baseline, single interventions, leave-one-out
  attribution, and Shapley attribution through the existing
  :mod:`agentdescent.credit` primitive.

The evaluator is called only through the lab, so a caller can memoise it by the
canonical trace signature. The lab itself is deliberately not a model runner:
that keeps the causal semantics testable and lets the same trace be replayed
against a different model or sandbox.

    lab = ReplayLab(replay)
    base = lab.baseline(trace)
    report = lab.leave_one_out(trace)
    shapley = lab.shapley(trace)
    shapley.ranking[0].edit_id
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .credit import Edit, CreditReport, leave_one_out_credit, shapley_credit

__all__ = [
    "TraceStep",
    "ReplayTrace",
    "Intervention",
    "CounterfactualResult",
    "CounterfactualReport",
    "ReplayLab",
    "apply_interventions",
]

Evaluator = Callable[[Sequence["TraceStep"]], float]


@dataclass(frozen=True)
class TraceStep:
    """One replayable step in a session."""

    step_id: str
    kind: str = "edit"
    payload: Any = None
    cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("TraceStep.step_id must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("TraceStep.kind must be a non-empty string")
        if not math.isfinite(float(self.cost)) or float(self.cost) < 0:
            raise ValueError("TraceStep.cost must be finite and >= 0")


@dataclass(frozen=True)
class ReplayTrace:
    """A session's ordered steps and its recorded score."""

    trace_id: str
    steps: Tuple[TraceStep, ...]
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("ReplayTrace.trace_id must be a non-empty string")
        steps = tuple(self.steps)
        if any(not isinstance(step, TraceStep) for step in steps):
            raise TypeError("ReplayTrace.steps must contain TraceStep objects")
        if len({step.step_id for step in steps}) != len(steps):
            raise ValueError("ReplayTrace.step_id values must be unique")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("ReplayTrace.score must be finite")
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True)
class Intervention:
    """One change to a trace: remove, replace or disable a step."""

    step_id: str
    action: str = "remove"
    replacement: Optional[TraceStep] = None

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("Intervention.step_id must be a non-empty string")
        if self.action not in ("remove", "replace", "disable"):
            raise ValueError("action must be remove, replace or disable")
        if self.action == "replace" and self.replacement is None:
            raise ValueError("replace requires a replacement TraceStep")
        if self.action != "replace" and self.replacement is not None:
            raise ValueError("replacement is only valid for action='replace'")
        if self.replacement is not None and self.replacement.step_id != self.step_id:
            raise ValueError("replacement must keep the same step_id")


@dataclass(frozen=True)
class CounterfactualResult:
    """One counterfactual's score and effect.

    ``cost`` is the cost of the **remaining** trace after intervention, not the
    cost saved. ``interventions`` carries the real tuple; ``intervention`` is a
    convenience for the single-intervention case (``None`` for a bundle).
    """

    interventions: Tuple[Intervention, ...]
    score: float
    delta: float
    cost: float
    detail: str = ""

    @property
    def intervention(self) -> Optional[Intervention]:
        return self.interventions[0] if len(self.interventions) == 1 else None


@dataclass(frozen=True)
class CounterfactualReport:
    """Baseline, leave-one-out, and Shapley views of one trace."""

    trace_id: str
    baseline: float
    results: Tuple[CounterfactualResult, ...] = ()
    shapley: Optional[CreditReport] = None
    detail: str = ""

    @property
    def largest_effect(self) -> Optional[CounterfactualResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda row: abs(row.delta))


def _check_score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"evaluator returned a non-finite score: {value!r}")
    return score


def apply_interventions(trace: ReplayTrace,
                        interventions: Iterable[Intervention]) -> Tuple[TraceStep, ...]:
    """Apply interventions in order, preserving the other steps.

    ``disable`` keeps the step in place with ``metadata['disabled']=True`` so a
    replay evaluator can implement "the tool existed but returned nothing".
    A step id may appear at most once; conflicting duplicates raise ``ValueError``.
    """
    by_id = {step.step_id: step for step in trace.steps}
    pending = {}
    for intervention in interventions:
        if intervention.step_id not in by_id:
            raise KeyError(f"unknown step {intervention.step_id!r} in {trace.trace_id!r}")
        if intervention.step_id in pending:
            raise ValueError(f"conflicting interventions for {intervention.step_id!r}")
        pending[intervention.step_id] = intervention
    steps: List[TraceStep] = []
    for step in trace.steps:
        intervention = pending.get(step.step_id)
        if intervention is None:
            steps.append(step)
            continue
        if intervention.action == "remove":
            continue
        if intervention.action == "replace":
            steps.append(intervention.replacement)  # type: ignore[arg-type]
            continue
        metadata = dict(step.metadata)
        metadata["disabled"] = True
        steps.append(TraceStep(step.step_id, step.kind, step.payload, step.cost,
                              metadata))
    return tuple(steps)


class ReplayLab:
    """Counterfactual replay over a deterministic evaluator."""

    def __init__(self, evaluate: Evaluator, *,
                 cache: Optional[Dict[Any, float]] = None,
                 key: Optional[Callable[[TraceStep], Any]] = None) -> None:
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        self.evaluate = evaluate
        self.cache = cache if cache is not None else {}
        self.key = key
        self.calls = 0

    def _step_key(self, step: TraceStep) -> Any:
        if self.key is not None:
            return self.key(step)
        try:
            hash(step.payload)
        except TypeError:
            return None                 # unhashable payload: do not cache
        metadata = []
        for key, value in sorted(step.metadata.items(), key=lambda item: str(item[0])):
            try:
                hash(value)
            except TypeError:
                return None             # unhashable metadata value: do not cache
            metadata.append((str(key), value))
        return (step.step_id, step.kind, ("hash", step.payload), step.cost,
                tuple(metadata))

    def signature(self, steps: Sequence[TraceStep]) -> Optional[Tuple[Any, ...]]:
        keys = [self._step_key(step) for step in steps]
        if any(key is None for key in keys):
            return None
        return tuple(keys)

    def score(self, steps: Sequence[TraceStep]) -> float:
        key = self.signature(steps)
        if key is not None and key in self.cache:
            return float(self.cache[key])
        self.calls += 1
        value = _check_score(self.evaluate(tuple(steps)))
        if key is not None:
            self.cache[key] = value
        return value

    def baseline(self, trace: ReplayTrace) -> float:
        return self.score(trace.steps)

    def counterfactual(self, trace: ReplayTrace,
                       interventions: Iterable[Intervention]) -> CounterfactualResult:
        base = self.baseline(trace)
        interventions = tuple(interventions)
        if not interventions:
            raise ValueError("counterfactual() needs at least one intervention")
        steps = apply_interventions(trace, interventions)
        score = self.score(steps)
        delta = base - score
        cost = sum(step.cost for step in steps)
        detail = (f"{len(interventions)} intervention(s), "
                  f"baseline={base:.6g}, replay={score:.6g}")
        return CounterfactualResult(interventions=interventions, score=score,
                                    delta=delta, cost=cost, detail=detail)

    def leave_one_out(self, trace: ReplayTrace) -> CounterfactualReport:
        base = self.baseline(trace)
        results = []
        for step in trace.steps:
            result = self.counterfactual(
                trace, [Intervention(step.step_id, "remove")])
            results.append(result)
        detail = (f"{len(trace.steps)} step(s), baseline={base:.6g}, "
                  f"calls={self.calls}")
        return CounterfactualReport(trace_id=trace.trace_id, baseline=base,
                                    results=tuple(results), detail=detail)

    def shapley(self, trace: ReplayTrace, *, permutations: int = 64,
                seed: int = 0, exact_limit: int = 7) -> CreditReport:
        """Shapley attribution over steps, using ``agentdescent.credit``."""
        edits = [Edit(step.step_id, payload=step) for step in trace.steps]

        def evaluate_subset(subset: Sequence[Edit]) -> float:
            return self.score([edit.payload for edit in subset])

        return shapley_credit(edits, evaluate_subset, permutations=permutations,
                              seed=seed, exact_limit=exact_limit)
