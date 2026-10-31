"""Keep a quality-diversity capability library, not a pile of near-copies.

`quality_diversity.QualityDiversityArchive` keeps one elite per behaviour cell;
`artifact_graph.ArtifactGraph` makes the artifacts retrievable; `semantic_dedup`
knows when two artifacts are the same behaviour under a new name. Each module
was useful alone, but nothing kept them in sync. `CapabilityArchive` is the
storage seam:

* a near-duplicate is rejected before it occupies a QD cell or the graph;
* only an artifact that becomes the cell's elite is registered in the graph,
  so the graph and the QD archive describe the same population;
* retrieval goes through the graph, selection through the QD archive;
* the report combines coverage/QD-score with graph shape.

    archive = CapabilityArchive(
        QualityDiversityArchive(bins=(4, 3)), ArtifactGraph())
    result = archive.add("csv-tool", text, score=0.8,
                         descriptor=(2, 1), kind="tool")
    archive.retrieve("parse csv", k=3)
    archive.report().illumination.coverage, archive.report().graph.artifacts
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .artifact_graph import Artifact, ArtifactGraph, GraphReport, SearchHit
from .quality_diversity import (
    Elite,
    IlluminationReport,
    QualityDiversityArchive,
    illumination_report,
)
from .semantic_dedup import simhash, simhash_similarity

__all__ = [
    "ArchiveAddResult",
    "CapabilityArchiveReport",
    "CapabilityArchive",
]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ArchiveAddResult:
    """What happened to one candidate capability."""

    artifact_id: str
    added: bool
    elite: bool = False
    duplicate_of: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class CapabilityArchiveReport:
    """QD coverage plus the retrieval graph's shape."""

    illumination: IlluminationReport
    graph: GraphReport
    duplicates: int = 0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "illumination": asdict(self.illumination),
            "graph": asdict(self.graph),
            "duplicates": self.duplicates,
            "detail": self.detail,
        }


class CapabilityArchive:
    """A QD archive, an artifact graph and semantic dedup kept in sync."""

    def __init__(self, archive: QualityDiversityArchive,
                 graph: ArtifactGraph, *,
                 dedup_threshold: float = 0.9) -> None:
        if not isinstance(archive, QualityDiversityArchive):
            raise TypeError("archive must be a QualityDiversityArchive")
        if not isinstance(graph, ArtifactGraph):
            raise TypeError("graph must be an ArtifactGraph")
        threshold = _finite(dedup_threshold, "dedup_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("dedup_threshold must be in [0, 1]")
        self.archive = archive
        self.graph = graph
        self.dedup_threshold = threshold
        self._fingerprints: Dict[str, int] = {
            artifact.artifact_id: simhash(artifact.text)
            for artifact in graph.artifacts
        }
        self._live: Dict[str, int] = dict(self._fingerprints)
        self._duplicates = 0

    def add(self, artifact_id: str, text: str, *, score: float,
            descriptor: Sequence[int], kind: str = "skill",
            tags: Sequence[str] = (), parents: Sequence[str] = (),
            metadata: Optional[Mapping[str, Any]] = None,
            generation: int = 0) -> ArchiveAddResult:
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_id is required")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required")
        if artifact_id in self.graph:
            raise ValueError(f"artifact {artifact_id!r} already exists")
        score = _finite(score, "score")
        # Validate the descriptor before touching the duplicate counter/archive.
        previous = self.archive.get(descriptor)
        artifact = Artifact(
            artifact_id=artifact_id, text=text, kind=kind, score=score,
            tags=tuple(tags), parents=tuple(parents),
            metadata=dict(metadata or {}))
        fingerprint = simhash(text)
        duplicate_of = self._nearest(fingerprint)
        if duplicate_of is not None:
            self._duplicates += 1
            return ArchiveAddResult(
                artifact_id, False, duplicate_of=duplicate_of,
                detail=f"near-duplicate of {duplicate_of!r}")
        # QD carries one parent pointer; the graph keeps the full parent list.
        elite = self.archive.add(
            artifact=artifact_id, score=score, descriptor=descriptor,
            parent_id=(parents[-1] if parents else None),
            generation=generation, metadata=metadata)
        if not elite:
            return ArchiveAddResult(
                artifact_id, False,
                detail="score did not replace the cell's elite")
        try:
            self.graph.add(artifact)
        except Exception:
            # Roll the QD cell back so a failed graph write does not evict the
            # previous elite or leave an unregistered cell.
            self.archive.remove(descriptor)
            if previous is not None:
                self.archive.add(
                    previous.artifact, previous.score, descriptor,
                    parent_id=previous.parent_id,
                    generation=previous.generation,
                    metadata=previous.metadata)
            raise
        if previous is not None and previous.artifact != artifact_id:
            self.graph.remove(previous.artifact)
            self._fingerprints.pop(previous.artifact, None)
            self._live.pop(previous.artifact, None)
        self._fingerprints[artifact_id] = fingerprint
        self._live[artifact_id] = fingerprint
        return ArchiveAddResult(artifact_id, True, elite=True,
                                detail="registered as the cell elite")

    def _nearest(self, fingerprint: int) -> Optional[str]:
        best = None
        best_score = self.dedup_threshold
        for artifact_id, candidate in self._live.items():
            similarity = simhash_similarity(fingerprint, candidate)
            if similarity >= best_score:
                best, best_score = artifact_id, similarity
        return best

    def retrieve(self, query: str, *, k: int = 5) -> Tuple[SearchHit, ...]:
        return self.graph.search(query, k=k)

    def select(self, *, mode: str = "uniform",
               seed: Optional[int] = None) -> Optional[Elite]:
        rng = None if seed is None else random.Random(seed)
        return self.archive.select(rng, mode=mode)

    def report(self) -> CapabilityArchiveReport:
        illumination = illumination_report(self.archive)
        graph = self.graph.report()
        detail = (f"{illumination.detail}; {graph.detail}; "
                  f"{self._duplicates} duplicate(s) refused")
        return CapabilityArchiveReport(
            illumination=illumination, graph=graph,
            duplicates=self._duplicates, detail=detail)
