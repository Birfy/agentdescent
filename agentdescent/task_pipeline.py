"""Turn production traces into a validated train/holdout task stream.

`data_flywheel` cleans and deduplicates raw traffic; `task_foundry` validates,
deduplicates, checks contamination and splits a task stream. This module is the
short pipeline between them, so the loop can start from real failures instead
of a hand-written task list:

    traces (TraceSample) -> DataFlywheel -> failure rows -> failure_seeds
        -> TaskFoundry -> train / holdout

    pipeline = TaskPipeline(DataFlywheel(), TaskFoundry(
        FoundryConfig(holdout_fraction=0.2),
        generator=expand, validator=check, contamination=overlap))
    report = pipeline.run(trace_rows)
    report.foundry.train, report.foundry.holdout, report.verdict

The pipeline does not generate or score tasks itself. It owns the hand-off and
the audit trail: how many traces were rejected by the flywheel, how many
accepted traces became seeds, and what the foundry did with each seed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from .data_flywheel import DataFlywheel, FlywheelReport, TraceSample
from .task_foundry import (
    FoundryReport,
    TaskCandidate,
    TaskFoundry,
    failure_seeds,
)

__all__ = ["TaskPipelineConfig", "TaskPipelineReport", "TaskPipeline"]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _prompt(task: Any) -> str:
    if isinstance(task, str):
        return task
    try:
        return json.dumps(task, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(task)


@dataclass(frozen=True)
class TaskPipelineConfig:
    """How traces become seeds, and when the result is usable."""

    failure_threshold: float = 0.5
    max_seeds: int = 100
    require_holdout: bool = True

    def __post_init__(self) -> None:
        threshold = _finite(self.failure_threshold, "failure_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("failure_threshold must be in [0, 1]")
        if isinstance(self.max_seeds, bool) or not isinstance(
                self.max_seeds, int) or self.max_seeds < 1:
            raise ValueError("max_seeds must be an int >= 1")
        if not isinstance(self.require_holdout, bool):
            raise TypeError("require_holdout must be a bool")
        object.__setattr__(self, "failure_threshold", threshold)


@dataclass(frozen=True)
class TaskPipelineReport:
    """The flywheel audit trail, the foundry report and the verdict."""

    flywheel: FlywheelReport
    foundry: FoundryReport
    seeds: Tuple[TaskCandidate, ...]
    verdict: str
    detail: str = ""

    @property
    def train(self) -> Tuple[TaskCandidate, ...]:
        return self.foundry.train

    @property
    def holdout(self) -> Tuple[TaskCandidate, ...]:
        return self.foundry.holdout

    @property
    def accepted(self) -> Tuple[TaskCandidate, ...]:
        return self.foundry.accepted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flywheel": {
                "accepted": self.flywheel.accepted_count,
                "rejected": dict(self.flywheel.rejection_counts()),
                "clusters": self.flywheel.clusters,
            },
            "seeds": len(self.seeds),
            "foundry": self.foundry.to_dict(),
            "verdict": self.verdict,
            "detail": self.detail,
        }


class TaskPipeline:
    """Mine traces, turn accepted failures into seeds, then run the foundry."""

    def __init__(self, flywheel: DataFlywheel, foundry: TaskFoundry, *,
                 config: Optional[TaskPipelineConfig] = None) -> None:
        if not isinstance(flywheel, DataFlywheel):
            raise TypeError("flywheel must be a DataFlywheel")
        if not isinstance(foundry, TaskFoundry):
            raise TypeError("foundry must be a TaskFoundry")
        self.flywheel = flywheel
        self.foundry = foundry
        self.config = config or TaskPipelineConfig()

    def run(self, traces: Iterable[TraceSample]) -> TaskPipelineReport:
        rows = list(traces)
        if any(not isinstance(row, TraceSample) for row in rows):
            raise TypeError("traces must contain TraceSample values")
        flywheel = self.flywheel.mine(rows)
        seeds = self._seeds(flywheel)
        foundry = self.foundry.run(seeds)
        if not seeds or not foundry.accepted:
            verdict = "empty"
        elif foundry.verdict in ("insufficient", "filtered", "no-train"):
            # Respect min_tasks / contamination / no-train: a pipeline must not
            # turn the foundry's explicit refusal into "ready".
            verdict = foundry.verdict
        elif not foundry.holdout:
            verdict = ("no-holdout" if self.config.require_holdout
                       else foundry.verdict)
        else:
            verdict = foundry.verdict
        detail = (
            f"{flywheel.accepted_count} accepted trace(s), {len(seeds)} seed(s), "
            f"{len(foundry.train)} train / {len(foundry.holdout)} holdout, "
            f"verdict={verdict}")
        return TaskPipelineReport(
            flywheel=flywheel, foundry=foundry, seeds=seeds, verdict=verdict,
            detail=detail)

    def _seeds(self, report: FlywheelReport) -> Tuple[TaskCandidate, ...]:
        rows = []
        for sample in report.accepted:
            if sample.reward >= self.config.failure_threshold:
                continue
            if sample.task is None or (
                    isinstance(sample.task, str) and not sample.task.strip()):
                continue
            rows.append({
                "task_id": sample.trace_id,
                "prompt": _prompt(sample.task),
                "reward": sample.reward,
                "output": sample.output,
                "metadata": dict(sample.metadata),
            })
        seeds = failure_seeds(rows, threshold=self.config.failure_threshold,
                              source="flywheel")
        # Flywheel order is deterministic; truncation keeps the first N.
        return seeds[:self.config.max_seeds]
