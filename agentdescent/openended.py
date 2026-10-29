"""Open-ended task/solver co-evolution with novelty and a difficulty frontier.

A fixed curriculum stops exploring once it has a few good tasks. POET-style
open-ended learning keeps generating **new environments** and keeps the ones
that are novel and learnable, so the solver is pulled toward its frontier
instead of replaying the same distribution.

This is a small, deterministic slice:

* :class:`TaskGenome` is a task with a behaviour descriptor and a difficulty;
* a caller-supplied generator mutates a parent into children;
* a caller-supplied solver scores each child;
* the archive keeps novel tasks, caps its size by score+novelty, and reports a
  difficulty frontier around ``target_difficulty``.

It composes with `TaskFoundry` (validation/dedup/contamination) and
`quality_diversity` (descriptor archive); it does not run a model itself.

    report = open_ended_coevolve(generate, solve, OpenEndedConfig(
        generations=8, population=6, target_difficulty=0.6))
    report.frontier, report.best
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "TaskGenome",
    "OpenEndedConfig",
    "ArchiveEntry",
    "OpenEndedReport",
    "open_ended_coevolve",
    "descriptor_distance",
]

Generator = Callable[[Optional["TaskGenome"], int], "TaskGenome"]
Solver = Callable[["TaskGenome"], float]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class TaskGenome:
    task_id: str
    payload: Any = None
    difficulty: float = 0.5
    descriptor: Tuple[float, ...] = (0.5,)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("TaskGenome.task_id is required")
        difficulty = _finite(self.difficulty, "difficulty")
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        descriptor = tuple(_finite(value, "descriptor") for value in self.descriptor)
        if not descriptor:
            raise ValueError("descriptor must not be empty")
        object.__setattr__(self, "difficulty", difficulty)
        object.__setattr__(self, "descriptor", descriptor)


@dataclass(frozen=True)
class OpenEndedConfig:
    generations: int = 8
    population: int = 6
    novelty_threshold: float = 0.05
    target_difficulty: float = 0.6
    frontier_width: float = 0.15
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    max_archive: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.generations < 1 or self.population < 1:
            raise ValueError("generations and population must be >= 1")
        if not math.isfinite(float(self.novelty_threshold)) or self.novelty_threshold < 0:
            raise ValueError("novelty_threshold must be finite and >= 0")
        if not 0.0 <= float(self.target_difficulty) <= 1.0:
            raise ValueError("target_difficulty must be in [0, 1]")
        if not 0.0 <= float(self.frontier_width) <= 1.0:
            raise ValueError("frontier_width must be in [0, 1]")
        if not 0.0 <= float(self.min_difficulty) <= float(self.max_difficulty) <= 1.0:
            raise ValueError("difficulty bounds must satisfy 0 <= min <= max <= 1")
        if self.max_archive < 1:
            raise ValueError("max_archive must be >= 1")


@dataclass(frozen=True)
class ArchiveEntry:
    task: TaskGenome
    score: float
    novelty: float
    generation: int

    @property
    def fitness(self) -> float:
        return self.score + 0.5 * self.novelty


@dataclass(frozen=True)
class OpenEndedReport:
    entries: Tuple[ArchiveEntry, ...]
    frontier: Tuple[ArchiveEntry, ...]
    best: Optional[ArchiveEntry]
    generations: int
    coverage: float
    detail: str = ""


def descriptor_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("descriptors must have the same dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _novelty(task: TaskGenome, archive: Sequence[ArchiveEntry]) -> float:
    if not archive:
        return 1.0
    return min(descriptor_distance(task.descriptor, entry.task.descriptor)
               for entry in archive)


def open_ended_coevolve(generator: Generator, solver: Solver,
                        config: Optional[OpenEndedConfig] = None) -> OpenEndedReport:
    cfg = config or OpenEndedConfig()
    if not callable(generator) or not callable(solver):
        raise TypeError("generator and solver must be callable")
    rng = random.Random(cfg.seed)
    first = generator(None, 0)
    if not isinstance(first, TaskGenome):
        raise TypeError("generator must return TaskGenome")
    first_score = float(solver(first))
    if not math.isfinite(first_score):
        raise ValueError("solver returned a non-finite score")
    archive: List[ArchiveEntry] = [ArchiveEntry(first, first_score, 1.0, 0)]
    for generation in range(1, cfg.generations + 1):
        parents = sorted(archive, key=lambda entry: (-entry.fitness, entry.task.task_id))
        pool = parents[: max(1, min(len(parents), cfg.population))]
        for index in range(cfg.population):
            parent = pool[rng.randrange(len(pool))].task
            child = generator(parent, generation * 10_000 + index)
            if not isinstance(child, TaskGenome):
                raise TypeError("generator must return TaskGenome")
            if not cfg.min_difficulty <= child.difficulty <= cfg.max_difficulty:
                continue
            score = float(solver(child))
            if not math.isfinite(score):
                raise ValueError("solver returned a non-finite score")
            novelty = _novelty(child, archive)
            if novelty < cfg.novelty_threshold:
                continue
            archive.append(ArchiveEntry(child, score, novelty, generation))
            archive.sort(key=lambda entry: (-entry.fitness, entry.task.task_id))
            del archive[cfg.max_archive:]
    frontier = tuple(
        entry for entry in archive
        if abs(entry.task.difficulty - cfg.target_difficulty) <= cfg.frontier_width)
    best = max(archive, key=lambda entry: (entry.score, entry.task.task_id))
    coverage = len(archive) / (1 + cfg.generations * cfg.population)
    return OpenEndedReport(
        entries=tuple(archive), frontier=frontier, best=best,
        generations=cfg.generations, coverage=coverage,
        detail=(f"{len(archive)} archived, {len(frontier)} on frontier, "
                f"best={best.score:.4f}, coverage={coverage:.3f}"))
