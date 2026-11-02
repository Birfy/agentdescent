"""Close the loop from production failures to an archived capability.

`TaskPipeline` turns traces into a validated task stream, `CapabilityScout`
turns failed tasks into a reviewed tool, and `CapabilityArchive` stores a
capability in a quality-diversity library with semantic deduplication. Each
seam was useful on its own; this module composes them so one call has a single,
auditable outcome:

    production traces -> TaskPipeline -> accepted tasks
        -> CapabilityScout -> promoted capability
        -> CapabilityArchive -> admitted / duplicate / not-elite

The loop owns only the hand-off and the verdict. It does not generate tasks,
synthesize tools, validate capabilities or deduplicate artifacts; those stay in
the modules that already know how to do them. A pipeline that cannot produce
usable tasks, or a scout that cannot admit a capability, is returned as an
explicit ``pipeline-*`` or ``capability-*`` verdict with no archive write.

    loop = CapabilityLoop(pipeline, scout, archive)
    report = loop.run(
        traces, goal="parse a CSV line", make_request=build_request,
        descriptor=(0, 0), kind="tool", tags=("csv",))
    report.admitted, report.verdict, report.archive.artifact_id
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .capability_archive import ArchiveAddResult, CapabilityArchive
from .capability_scout import CapabilityScout, CapabilityScoutReport
from .data_flywheel import TraceSample
from .task_pipeline import TaskPipeline, TaskPipelineReport

__all__ = ["CapabilityLoopReport", "CapabilityLoop"]

_UNUSABLE_PIPELINE_VERDICTS = (
    "empty", "insufficient", "filtered", "no-train", "no-holdout")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _descriptor(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
        raise TypeError("descriptor must be a sequence of ints")
    cells = tuple(value)
    if not cells:
        raise ValueError("descriptor must be a non-empty sequence of ints")
    for index, cell in enumerate(cells):
        if isinstance(cell, bool) or not isinstance(cell, int):
            raise TypeError(f"descriptor[{index}] must be an int")
    return cells


def _strings(value: Any, name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
        raise TypeError(f"{name} must be a sequence of strings, not text")
    items = tuple(value)
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{index}] must be a string")
    return items


def _available_id(archive: CapabilityArchive, name: str, source: str) -> str:
    if name not in archive.graph:
        return name
    digest = hashlib.sha256(
        (name + "\x00" + source).encode("utf-8")).hexdigest()[:8]
    candidate = f"{name}~{digest}"
    if candidate not in archive.graph:
        return candidate
    index = 1
    while f"{name}~{digest}-{index}" in archive.graph:
        index += 1
    return f"{name}~{digest}-{index}"


@dataclass(frozen=True)
class CapabilityLoopReport:
    """The pipeline trail, scout outcome, archive result and final verdict."""

    pipeline: TaskPipelineReport
    scout: Optional[CapabilityScoutReport]
    archive: Optional[ArchiveAddResult]
    verdict: str
    detail: str = ""

    @property
    def admitted(self) -> bool:
        return self.archive is not None and self.archive.added

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline.to_dict(),
            "scout": None if self.scout is None else self.scout.to_dict(),
            "archive": None if self.archive is None else {
                "artifact_id": self.archive.artifact_id,
                "added": self.archive.added,
                "elite": self.archive.elite,
                "duplicate_of": self.archive.duplicate_of,
                "detail": self.archive.detail,
            },
            "verdict": self.verdict,
            "detail": self.detail,
            "admitted": self.admitted,
        }


class CapabilityLoop:
    """Compose the trace, task, capability and archive seams into one call."""

    def __init__(self, pipeline: TaskPipeline, scout: CapabilityScout,
                 archive: CapabilityArchive) -> None:
        if not isinstance(pipeline, TaskPipeline):
            raise TypeError("pipeline must be a TaskPipeline")
        if not isinstance(scout, CapabilityScout):
            raise TypeError("scout must be a CapabilityScout")
        if not isinstance(archive, CapabilityArchive):
            raise TypeError("archive must be a CapabilityArchive")
        self.pipeline = pipeline
        self.scout = scout
        self.archive = archive

    def run(self, traces: Iterable[TraceSample], *, goal: str,
            make_request: Callable[[Any], Any], descriptor: Sequence[int],
            kind: str = "tool", tags: Sequence[str] = (),
            parents: Sequence[str] = (),
            metadata: Optional[Mapping[str, Any]] = None
            ) -> CapabilityLoopReport:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal is required")
        if not callable(make_request):
            raise TypeError("make_request must be callable")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind must be a non-empty string")
        tag_values = _strings(tags, "tags")
        parent_values = _strings(parents, "parents")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        cells = _descriptor(descriptor)

        pipeline = self.pipeline.run(traces)
        if not pipeline.accepted or pipeline.verdict in _UNUSABLE_PIPELINE_VERDICTS:
            return CapabilityLoopReport(
                pipeline, None, None, f"pipeline-{pipeline.verdict}",
                f"pipeline verdict {pipeline.verdict!r} cannot seed a capability: "
                f"{pipeline.detail}")

        tasks = pipeline.accepted or pipeline.train
        if not tasks:
            return CapabilityLoopReport(
                pipeline, None, None, "pipeline-empty",
                "pipeline accepted no tasks to scout")

        scout = self.scout.run(tasks, goal=goal, make_request=make_request)
        outcome = scout.outcome
        if outcome is None or not outcome.accepted:
            status = scout.verdict if outcome is None else outcome.status
            return CapabilityLoopReport(
                pipeline, scout, None, f"capability-{status}",
                f"scout did not admit a capability: {scout.detail}")

        selected = scout.synthesis.selected
        if selected is None:
            return CapabilityLoopReport(
                pipeline, scout, None, "capability-no-candidate",
                "scout reported an admission without a selected candidate")

        spec = selected.spec
        score = selected.validation.score
        if not score and outcome.holdout_gain is not None:
            score = outcome.holdout_gain
        if score is None:
            score = 0.0
        score = _finite(score, "score")

        artifact_metadata = dict(metadata or {})
        artifact_metadata.update({
            "pipeline_verdict": pipeline.verdict,
            "scout_verdict": scout.verdict,
            "gap_id": outcome.gap_id,
            "validation_score": selected.validation.score,
            "holdout_gain": outcome.holdout_gain,
        })
        artifact_id = _available_id(self.archive, spec.name, spec.source)
        result = self.archive.add(
            artifact_id=artifact_id, text=spec.source, score=score,
            descriptor=cells, kind=kind, tags=tag_values,
            parents=parent_values, metadata=artifact_metadata)

        if result.added:
            verdict = "admitted"
            detail = f"archived {artifact_id!r} as a {kind} capability"
        elif result.duplicate_of is not None:
            verdict = "duplicate"
            detail = (f"capability for {spec.name!r} duplicates "
                      f"{result.duplicate_of!r}: {result.detail}")
        else:
            verdict = "not-elite"
            detail = (f"capability {artifact_id!r} did not become the cell "
                      f"elite: {result.detail}")
        return CapabilityLoopReport(pipeline, scout, result, verdict, detail)
