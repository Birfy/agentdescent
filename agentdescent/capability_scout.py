"""From failure tasks to a reviewed, admitted tool, not a loose proposal.

`TaskPipeline` produces failed/held-out tasks; `tool_synthesis` can propose a
tool; `CapabilityFoundry` knows how to admit one. Nothing joined them. This
module is that short loop:

    failed TaskCandidates -> ToolSynthesisRequest -> ToolSynthesizer
        -> selected ToolCandidate -> caller's CapabilityRequest
        -> CapabilityFoundry.admit -> promoted / deferred / rejected

The scout does not invent a tool or a capability policy. It packages the
failure prompts as synthesis examples, runs the existing synthesizer, and hands
the selected candidate to the caller's request builder before admission.

    scout = CapabilityScout(synthesizer, foundry)
    report = scout.run(failed_tasks, goal="parse a CSV line",
                       make_request=build_capability_request)
    report.verdict, report.outcome.artifact_id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from .capability_foundry import (
    CapabilityFoundry,
    CapabilityOutcome,
    CapabilityRequest,
)
from .task_foundry import TaskCandidate
from .tool_synthesis import (
    ToolCandidate,
    ToolSynthesisReport,
    ToolSynthesisRequest,
    ToolSynthesizer,
)

__all__ = ["CapabilityScoutReport", "CapabilityScout"]

RequestBuilder = Callable[[ToolCandidate], Optional[CapabilityRequest]]


@dataclass(frozen=True)
class CapabilityScoutReport:
    """The failure tasks, synthesis trail and admission outcome."""

    goal: str
    tasks: int
    synthesis: ToolSynthesisReport
    outcome: Optional[CapabilityOutcome]
    verdict: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "tasks": self.tasks,
            "synthesis": {
                "candidates": len(self.synthesis.candidates),
                "accepted": len(self.synthesis.accepted),
                "selected": (None if self.synthesis.selected is None
                             else self.synthesis.selected.spec.name),
            },
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
            "verdict": self.verdict,
            "detail": self.detail,
        }


class CapabilityScout:
    """Turn failure tasks into a reviewed tool via the existing seams."""

    def __init__(self, synthesizer: ToolSynthesizer,
                 foundry: CapabilityFoundry, *, max_examples: int = 5,
                 max_candidates: int = 4) -> None:
        if not isinstance(synthesizer, ToolSynthesizer):
            raise TypeError("synthesizer must be a ToolSynthesizer")
        if not isinstance(foundry, CapabilityFoundry):
            raise TypeError("foundry must be a CapabilityFoundry")
        for name, value in (("max_examples", max_examples),
                            ("max_candidates", max_candidates)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1")
        self.synthesizer = synthesizer
        self.foundry = foundry
        self.max_examples = max_examples
        self.max_candidates = max_candidates

    def run(self, tasks: Sequence[TaskCandidate], *, goal: str,
            make_request: RequestBuilder) -> CapabilityScoutReport:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal is required")
        if not callable(make_request):
            raise TypeError("make_request must be callable")
        rows = list(tasks)
        if not rows:
            raise ValueError("CapabilityScout.run needs at least one task")
        if any(not isinstance(task, TaskCandidate) for task in rows):
            raise TypeError("tasks must contain TaskCandidate values")
        examples = tuple(task.prompt for task in rows[:self.max_examples])
        request = ToolSynthesisRequest(
            goal, examples=examples,
            constraints={"task_ids": [task.task_id for task in rows]},
            max_candidates=self.max_candidates)
        synthesis = self.synthesizer.synthesize(request)
        if synthesis.selected is None:
            detail = f"no valid candidate from {len(synthesis.candidates)} attempt(s)"
            return CapabilityScoutReport(
                goal, len(rows), synthesis, None, "no-candidate", detail)
        request_object = make_request(synthesis.selected)
        if request_object is None:
            return CapabilityScoutReport(
                goal, len(rows), synthesis, None, "no-request",
                "make_request returned None")
        if not isinstance(request_object, CapabilityRequest):
            raise TypeError("make_request must return CapabilityRequest or None")
        outcome = self.foundry.admit(request_object)
        return CapabilityScoutReport(
            goal, len(rows), synthesis, outcome, outcome.status,
            f"selected={synthesis.selected.spec.name}; {outcome.status}")
