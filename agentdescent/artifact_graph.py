"""Retrieve and compose artifacts across runs by what they actually do.

The archive knows what was adopted; the consolidation store knows what survived
a run. Neither answers the question a new session asks first: **which prior
skills, prompts or harness fragments are relevant to *this* task, and can I
compose them without pulling in the whole library?**

This module is a small, dependency-free retrieval graph over artifacts. It is
deliberately not an embedding service: a Unicode-aware IDF-weighted overlap
ranker over the artifact text is deterministic, works offline, and is enough to
beat a linear scan of the archive. The graph remains the place where
parent/child edges, trust and composition live; embeddings can be layered on by
a caller without changing this storage.

    graph = ArtifactGraph()
    graph.add(Artifact("skill-1", "use pytest fixtures ...", kind="skill", score=0.8))
    hits = graph.search("pytest fixture isolation", k=3)
    composed = graph.compose("pytest fixture isolation", k=2, max_chars=2000)

The graph is version-agnostic: parents are artifact ids, so it composes with
`lineage`, `quality_diversity` and the consolidation archive without importing
them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "Artifact",
    "SearchHit",
    "Composition",
    "GraphReport",
    "ArtifactGraph",
]

_CJK_RANGE = "\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af"
_SEGMENT = re.compile(
    rf"[{_CJK_RANGE}]+|[^\W{_CJK_RANGE}]+", re.UNICODE)
_CJK = re.compile(rf"[{_CJK_RANGE}]")


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    for segment in _SEGMENT.findall(str(text).lower()):
        if _CJK.search(segment):
            out.append(segment)
            if len(segment) > 1:
                out.extend(segment[i:i + 2] for i in range(len(segment) - 1))
        else:
            out.append(segment)
    return out


@dataclass(frozen=True)
class Artifact:
    """One reusable prompt, skill, code fragment or harness module."""

    artifact_id: str
    text: str
    kind: str = "artifact"
    score: float = 0.0
    tags: Tuple[str, ...] = ()
    parents: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("Artifact.artifact_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Artifact.text must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("Artifact.kind must be a non-empty string")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("Artifact.score must be finite")
        object.__setattr__(self, "score", score)
        if isinstance(self.tags, str) or isinstance(self.parents, str):
            raise ValueError("Artifact.tags and parents must be sequences, not bare strings")
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "parents", tuple(str(parent) for parent in self.parents))
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("Artifact.parents must be unique")


@dataclass(frozen=True)
class SearchHit:
    """One ranked artifact, with the terms that matched."""

    artifact: Artifact
    score: float
    matched_terms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Composition:
    """Artifacts selected for one task and the text they produce."""

    hits: Tuple[SearchHit, ...]
    text: str
    coverage: float
    missing_terms: Tuple[str, ...] = ()
    skipped: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphReport:
    """Shape and trust distribution of the graph."""

    artifacts: int
    edges: int
    roots: Tuple[str, ...]
    leaves: Tuple[str, ...]
    kinds: Dict[str, int]
    tags: Dict[str, int]
    detail: str = ""


class ArtifactGraph:
    """A deterministic retrieval graph over reusable artifacts."""

    def __init__(self, *, trust_weight: float = 0.0,
                 tokenizer: Optional[Callable[[str], Sequence[str]]] = None) -> None:
        if not math.isfinite(float(trust_weight)) or trust_weight < 0:
            raise ValueError("trust_weight must be finite and >= 0")
        self.trust_weight = float(trust_weight)
        self.tokenizer = tokenizer or _tokens
        self._artifacts: Dict[str, Artifact] = {}
        self._tokens: Dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._children: Dict[str, Set[str]] = {}

    # -- mutation ------------------------------------------------------------

    def add(self, artifact: Artifact) -> None:
        if not isinstance(artifact, Artifact):
            raise TypeError(f"add() expects Artifact, got {type(artifact).__name__}")
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"duplicate artifact id {artifact.artifact_id!r}")
        if artifact.artifact_id in artifact.parents:
            raise ValueError(f"artifact {artifact.artifact_id!r} cannot be its own parent")
        # Reject a parent edge that would close a cycle through known children.
        stack = [artifact.artifact_id]
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in artifact.parents:
                raise ValueError(f"artifact {artifact.artifact_id!r} would create a cycle")
            stack.extend(self._children.get(current, ()))
        counts = Counter(self.tokenizer(artifact.text))
        self._artifacts[artifact.artifact_id] = artifact
        self._tokens[artifact.artifact_id] = counts
        for token in counts:
            self._df[token] += 1
        for parent in artifact.parents:
            self._children.setdefault(parent, set()).add(artifact.artifact_id)

    def extend(self, artifacts: Iterable[Artifact]) -> "ArtifactGraph":
        for artifact in artifacts:
            self.add(artifact)
        return self

    # -- graph ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._artifacts)

    def __contains__(self, artifact_id: object) -> bool:
        return artifact_id in self._artifacts

    def __getitem__(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError:
            raise KeyError(f"unknown artifact {artifact_id!r}") from None

    @property
    def artifacts(self) -> Tuple[Artifact, ...]:
        return tuple(self._artifacts[key] for key in sorted(self._artifacts))

    def children(self, artifact_id: str) -> Tuple[str, ...]:
        self[artifact_id]
        return tuple(sorted(self._children.get(artifact_id, ())))

    def parents(self, artifact_id: str) -> Tuple[str, ...]:
        return tuple(sorted(self[artifact_id].parents))

    def roots(self) -> Tuple[str, ...]:
        return tuple(sorted(
            key for key, artifact in self._artifacts.items()
            if not any(parent in self._artifacts for parent in artifact.parents)))

    def leaves(self) -> Tuple[str, ...]:
        return tuple(sorted(key for key in self._artifacts
                            if not self._children.get(key)))

    def neighbors(self, artifact_id: str, *, direction: str = "both") -> Tuple[str, ...]:
        if direction not in ("parents", "children", "both"):
            raise ValueError("direction must be parents, children or both")
        found = set()
        if direction in ("parents", "both"):
            found.update(self.parents(artifact_id))
        if direction in ("children", "both"):
            found.update(self.children(artifact_id))
        return tuple(sorted(found))

    # -- retrieval -----------------------------------------------------------

    def _idf(self, token: str) -> float:
        total = max(1, len(self._artifacts))
        df = self._df.get(token, 0)
        return math.log((total + 1.0) / (df + 1.0)) + 1.0

    def search(self, query: str, *, k: int = 5, kind: Optional[str] = None,
               tags: Optional[Iterable[str]] = None,
               min_score: float = 0.0,
               exclude: Iterable[str] = ()) -> Tuple[SearchHit, ...]:
        """Rank artifacts by IDF-weighted query overlap.

        Filtering happens before ranking. Ties break by trust score and then
        artifact id, so the result is reproducible.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        query_tokens = self.tokenizer(query)
        if not query_tokens:
            raise ValueError("query must contain at least one token")
        if min_score < 0 or not math.isfinite(min_score):
            raise ValueError("min_score must be finite and >= 0")
        wanted_tags = {str(tag) for tag in tags} if tags is not None else None
        excluded = {str(item) for item in exclude}
        unique_query = sorted(set(query_tokens))
        hits: List[SearchHit] = []
        for artifact_id in sorted(self._artifacts):
            if artifact_id in excluded:
                continue
            artifact = self._artifacts[artifact_id]
            if kind is not None and artifact.kind != kind:
                continue
            if wanted_tags is not None and not wanted_tags.issubset(set(artifact.tags)):
                continue
            counts = self._tokens[artifact_id]
            matched = [token for token in unique_query if counts.get(token, 0)]
            if not matched:
                continue
            lexical = sum(self._idf(token) for token in matched) / math.sqrt(
                max(1, len(unique_query)) * max(1, sum(counts.values())))
            score = lexical * (1.0 + self.trust_weight * max(0.0, artifact.score))
            if score < min_score:
                continue
            matched.sort(key=lambda token: (-self._idf(token), token))
            hits.append(SearchHit(artifact=artifact, score=score,
                                  matched_terms=tuple(matched)))
        hits.sort(key=lambda hit: (-hit.score, -hit.artifact.score,
                                   hit.artifact.artifact_id))
        return tuple(hits[:k])

    def compose(self, query: str, *, k: int = 3,
                max_chars: Optional[int] = None,
                separator: str = "\n\n",
                **search_kwargs: Any) -> Composition:
        """Select hits and concatenate them under a character budget.

        ``coverage`` is the fraction of unique query terms present in the
        selected artifacts. A hit that would exceed ``max_chars`` is skipped
        (and reported in ``skipped``) rather than truncating a skill in the
        middle.
        """
        if max_chars is not None and max_chars < 1:
            raise ValueError("max_chars must be >= 1 or None")
        search_k = k if max_chars is None else max(k * 4, k)
        hits = self.search(query, k=search_k, **search_kwargs)
        selected: List[SearchHit] = []
        skipped: List[str] = []
        parts: List[str] = []
        length = 0
        for hit in hits:
            if len(selected) >= k:
                break
            extra = len(hit.artifact.text) + (len(separator) if parts else 0)
            if max_chars is not None and length + extra > max_chars:
                skipped.append(hit.artifact.artifact_id)
                continue
            selected.append(hit)
            parts.append(hit.artifact.text)
            length += extra
        query_terms = set(self.tokenizer(query))
        covered: Set[str] = set()
        for hit in selected:
            covered.update(query_terms & set(self._tokens[hit.artifact.artifact_id]))
        coverage = len(covered) / len(query_terms) if query_terms else 0.0
        return Composition(hits=tuple(selected), text=separator.join(parts),
                           coverage=coverage,
                           missing_terms=tuple(sorted(query_terms - covered)),
                           skipped=tuple(skipped))

    # -- read models ---------------------------------------------------------

    def report(self) -> GraphReport:
        edges = sum(len(children) for children in self._children.values())
        kinds: Counter = Counter(artifact.kind for artifact in self._artifacts.values())
        tags: Counter = Counter(tag for artifact in self._artifacts.values()
                                for tag in artifact.tags)
        detail = (f"{len(self)} artifacts, {edges} edges, "
                  f"{len(self.roots())} roots, {len(self.leaves())} leaves")
        return GraphReport(artifacts=len(self), edges=edges,
                           roots=self.roots(), leaves=self.leaves(),
                           kinds=dict(kinds), tags=dict(tags), detail=detail)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifacts": [
                {"artifact_id": artifact.artifact_id, "kind": artifact.kind,
                 "score": artifact.score, "tags": list(artifact.tags),
                 "parents": list(artifact.parents), "text": artifact.text,
                 "metadata": dict(artifact.metadata)}
                for artifact in self.artifacts
            ]
        }
