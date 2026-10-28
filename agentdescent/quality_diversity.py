"""Quality-diversity search for self-evolving artifacts.

A scalar archive answers "what is the best artifact so far?" and throws away the
rest. That is the wrong question for recursive self-improvement: the trajectory
between two artifacts can matter more than either endpoint, and an artifact that
is mediocre on the current metric may be the parent of the next breakthrough.
MAP-Elites makes the trade-off explicit by keeping one elite per **behaviour
cell** and reporting how much of the behaviour space is illuminated.

This module is deliberately storage-only. It does not call a model, run a
task, or decide what a descriptor means; the caller supplies a descriptor per
candidate (for example ``(diff_size_bucket, uses_tools, language)``). That keeps
the archive usable with every strategy in the library -- prompts, code,
directory trees, verifier rules -- without importing any of them.

    archive = QualityDiversityArchive(bins=(4, 3))
    archive.add(artifact=candidate, score=0.71, descriptor=(2, 1))
    report = illumination_report(archive)
    print(report.coverage, report.qd_score)

The archive is deterministic: ``select`` takes an explicit ``random.Random`` and
the default selection mode is uniform over filled cells, not a hidden clock or
global RNG. ``visits`` counts selections so a caller can combine coverage with
recency (``mode="novelty"``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Elite",
    "QualityDiversityArchive",
    "IlluminationReport",
    "descriptor_distance",
    "illumination_report",
]


Cell = Tuple[int, ...]


def _validate_bins(bins: Sequence[int]) -> Tuple[int, ...]:
    if not bins:
        raise ValueError("bins must name at least one behaviour dimension")
    out = []
    for value in bins:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"bin sizes must be ints, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"bin sizes must be >= 1, got {value}")
        out.append(value)
    return tuple(out)


@dataclass(frozen=True)
class Elite:
    """One occupied behaviour cell and the artifact that owns it.

    ``artifact`` is opaque to the archive. ``parent_id`` and ``generation`` are
    carried so a caller can reconstruct a lineage without a second store, and
    ``metadata`` is a free-form bag (run id, slot, source hash, ...).
    """

    artifact: Any
    score: float
    descriptor: Cell
    parent_id: Optional[str] = None
    generation: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score != self.score or self.score in (float("inf"), float("-inf")):
            raise ValueError("Elite.score must be finite")
        if self.generation < 0:
            raise ValueError("Elite.generation must be >= 0")


class QualityDiversityArchive:
    """One elite per behaviour cell, with MAP-Elites replacement.

    Parameters
    ----------
    bins:
        Number of cells along each behaviour dimension. A descriptor must be an
        integer tuple of the same length, with ``0 <= value < bins[i]``.
    seed:
        Seed for the archive's own ``random.Random``. Selection still accepts an
        explicit RNG; the stored one makes ``select()`` reproducible by default.
    """

    def __init__(self, bins: Sequence[int], *, seed: int = 0) -> None:
        self.bins = _validate_bins(bins)
        self._cells: Dict[Cell, Elite] = {}
        self._visits: Dict[Cell, int] = {}
        self._rng = random.Random(seed)

    # -- descriptors ---------------------------------------------------------

    def _check_descriptor(self, descriptor: Sequence[int]) -> Cell:
        if isinstance(descriptor, (str, bytes)):
            raise TypeError("descriptor must be a sequence of ints, not text")
        cell = tuple(descriptor)  # type: ignore[arg-type]
        if len(cell) != len(self.bins):
            raise ValueError(
                f"descriptor has {len(cell)} dimensions, archive has {len(self.bins)}")
        for i, (value, size) in enumerate(zip(cell, self.bins)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"descriptor[{i}] must be int, got {type(value).__name__}")
            if value < 0 or value >= size:
                raise ValueError(
                    f"descriptor[{i}]={value} outside [0, {size - 1}]")
        return cell

    # -- mutation ------------------------------------------------------------

    def add(self, artifact: Any, score: float, descriptor: Sequence[int], *,
            parent_id: Optional[str] = None, generation: int = 0,
            metadata: Optional[Mapping[str, Any]] = None) -> bool:
        """Insert ``artifact`` and return whether it became the cell's elite.

        A cell is replaced only by a strictly higher score. Ties keep the first
        arrival so a deterministic stream of candidates is not reordered by an
        arbitrary comparison of opaque artifacts.
        """
        cell = self._check_descriptor(descriptor)
        elite = Elite(artifact=artifact, score=float(score), descriptor=cell,
                      parent_id=parent_id, generation=int(generation),
                      metadata=dict(metadata or {}))
        current = self._cells.get(cell)
        if current is not None and elite.score <= current.score:
            return False
        self._cells[cell] = elite
        self._visits.setdefault(cell, 0)
        return True

    def remove(self, descriptor: Sequence[int]) -> bool:
        """Evict one cell; return whether it was occupied."""
        cell = self._check_descriptor(descriptor)
        self._visits.pop(cell, None)
        return self._cells.pop(cell, None) is not None

    def clear(self) -> None:
        self._cells.clear()
        self._visits.clear()

    # -- read models ---------------------------------------------------------

    @property
    def filled(self) -> int:
        return len(self._cells)

    @property
    def total(self) -> int:
        total = 1
        for size in self.bins:
            total *= size
        return total

    @property
    def coverage(self) -> float:
        return self.filled / self.total

    @property
    def qd_score(self) -> float:
        """Sum of elite scores: MAP-Elites' ``QD-score``."""
        return sum(elite.score for elite in self._cells.values())

    @property
    def best(self) -> Optional[Elite]:
        if not self._cells:
            return None
        return max(self._cells.values(), key=lambda elite: elite.score)

    @property
    def elites(self) -> Tuple[Elite, ...]:
        return tuple(self._cells[key] for key in sorted(self._cells))

    def get(self, descriptor: Sequence[int]) -> Optional[Elite]:
        return self._cells.get(self._check_descriptor(descriptor))

    def visits(self, descriptor: Sequence[int]) -> int:
        return self._visits.get(self._check_descriptor(descriptor), 0)

    # -- selection -----------------------------------------------------------

    def select(self, rng: Optional[random.Random] = None, *,
               mode: str = "uniform") -> Optional[Elite]:
        """Pick a parent from the occupied cells.

        ``uniform`` samples a filled cell, ``best`` returns the top elite,
        ``novelty`` weights each cell by ``1 / (1 + visits)`` so repeated picks
        move on, and ``score`` samples proportional to elite score. The selected
        cell's visit counter is incremented in every mode except ``best``.
        """
        if not self._cells:
            return None
        rng = rng or self._rng
        cells: List[Cell] = sorted(self._cells)
        if mode == "best":
            return self.best
        if mode == "uniform":
            cell = rng.choice(cells)
        elif mode == "novelty":
            weights = [1.0 / (1 + self._visits.get(c, 0)) for c in cells]
            cell = rng.choices(cells, weights=weights, k=1)[0]
        elif mode == "score":
            weights = [max(0.0, self._cells[c].score) for c in cells]
            if sum(weights) <= 0:
                cell = rng.choice(cells)
            else:
                cell = rng.choices(cells, weights=weights, k=1)[0]
        else:
            raise ValueError(
                f"unknown selection mode {mode!r}; use uniform, best, novelty or score")
        self._visits[cell] = self._visits.get(cell, 0) + 1
        return self._cells[cell]

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the archive's non-opaque fields.

        ``artifact`` is not serialised: it belongs to the caller's store. A
        reloaded archive is a *skeleton* for layout inspection, not a second
        artifact store.
        """
        return {
            "bins": list(self.bins),
            "cells": [
                {
                    "descriptor": list(elite.descriptor),
                    "score": elite.score,
                    "parent_id": elite.parent_id,
                    "generation": elite.generation,
                    "metadata": dict(elite.metadata),
                }
                for elite in self.elites
            ],
        }


@dataclass(frozen=True)
class IlluminationReport:
    """How much of the behaviour space the archive has illuminated."""

    filled: int
    total: int
    coverage: float
    qd_score: float
    best_score: Optional[float]
    verdict: str
    detail: str = ""


def illumination_report(archive: QualityDiversityArchive) -> IlluminationReport:
    """Summarise coverage and QD-score with a coarse verdict.

    The thresholds are descriptive, not a gate: ``empty`` (no cells),
    ``sparse`` (< 25%), ``partial`` (< 75%) and ``illuminated`` (>= 75%).
    """
    coverage = archive.coverage
    if archive.filled == 0:
        verdict = "empty"
    elif coverage < 0.25:
        verdict = "sparse"
    elif coverage < 0.75:
        verdict = "partial"
    else:
        verdict = "illuminated"
    best = archive.best
    detail = (f"{archive.filled}/{archive.total} cells; "
              f"QD-score={archive.qd_score:.4f}; "
              f"best={best.score:.4f}" if best else
              f"{archive.filled}/{archive.total} cells; QD-score=0")
    return IlluminationReport(
        filled=archive.filled,
        total=archive.total,
        coverage=coverage,
        qd_score=archive.qd_score,
        best_score=None if best is None else best.score,
        verdict=verdict,
        detail=detail,
    )


def descriptor_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Manhattan distance between two behaviour descriptors.

    The archive does not require a metric, but callers writing a novelty bonus
    do; this keeps the common one in one place and validates equal length.
    """
    a = tuple(left)
    b = tuple(right)
    if len(a) != len(b):
        raise ValueError("descriptors must have the same number of dimensions")
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))
