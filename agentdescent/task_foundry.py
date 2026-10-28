"""Turn real failures into a validated, uncontaminated task stream.

A curriculum generator is only as good as its seeds. Generating tasks from a
model's imagination produces plausible-looking problems with no ground truth;
generating them from **failed rollouts, failed tests, and rejected candidates**
keeps the curriculum anchored to the capability the loop is actually missing.

This module is the assembly line between those seeds and an `evolve()` task set:

1. **mine** -- :func:`failure_seeds` turns low-reward trajectory rows into
   :class:`TaskCandidate` seeds (the caller may also supply its own seeds);
2. **generate** -- a caller-supplied generator expands each seed into zero or
   more candidates;
3. **validate** -- a validation callable scores whether the candidate has a
   usable answer/check;
4. **deduplicate** -- exact normalized prompts, plus an optional near-duplicate
   similarity hook (for example the `semantic_dedup` module once merged);
5. **contamination** -- a caller-supplied overlap score against the frozen
   evaluation set;
6. **learnability** -- a callable can reject tasks that are already solved or
   impossible, so the curriculum stays in the zone of proximal development;
7. **split** -- accepted tasks are deterministically partitioned into train and
   holdout by a stable hash, not input order.

The foundry stores no tasks itself. The report is the audit trail; the caller
turns the accepted train/holdout candidates into its own `Task` objects.

    foundry = TaskFoundry(config, generator=gen, validator=check,
                          contamination=overlap, learnability=solve_rate)
    report = foundry.run(failure_seeds(trajectory_rows))
    train_tasks, holdout_tasks = report.train, report.holdout
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evolvable import stable_hash

__all__ = [
    "TaskCandidate",
    "TaskRecord",
    "FoundryConfig",
    "FoundryReport",
    "TaskFoundry",
    "failure_seeds",
]

_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS.sub(" ", str(text).strip().lower())


@dataclass(frozen=True)
class TaskCandidate:
    """One candidate task before validation."""

    task_id: str
    prompt: str
    payload: Any = None
    source: str = ""
    difficulty: float = 0.5
    parent_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("TaskCandidate.task_id must be a non-empty string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("TaskCandidate.prompt must be a non-empty string")
        value = float(self.difficulty)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("TaskCandidate.difficulty must be in [0, 1]")


@dataclass(frozen=True)
class FoundryConfig:
    """Thresholds for the assembly line.

    ``per_seed`` bounds generator calls per seed; ``max_candidates`` bounds the
    whole run. ``holdout_fraction`` is a fraction of *accepted* tasks, assigned
    by stable hash. ``near_duplicate_threshold`` is only used when a similarity
    callable is supplied; exact normalized duplicates are always rejected.
    """

    per_seed: int = 2
    max_candidates: int = 100
    min_validation: float = 0.5
    min_learnability: float = 0.0
    max_learnability: float = 1.0
    max_contamination: float = 0.0
    near_duplicate_threshold: float = 0.95
    holdout_fraction: float = 0.2
    holdout_salt: str = ""
    min_tasks: int = 1
    max_difficulty: float = 1.0

    def __post_init__(self) -> None:
        for name in ("per_seed", "max_candidates", "min_tasks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an int >= 0")
        if self.per_seed < 1:
            raise ValueError("per_seed must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        for name in ("min_validation", "min_learnability", "max_learnability",
                     "max_contamination", "near_duplicate_threshold",
                     "holdout_fraction", "max_difficulty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if float(self.holdout_fraction) >= 1.0:
            raise ValueError("holdout_fraction must be < 1")


@dataclass(frozen=True)
class TaskRecord:
    """One candidate's fate in the foundry."""

    candidate: TaskCandidate
    validation: Optional[float]
    learnability: Optional[float]
    contamination: Optional[float]
    accepted: bool
    reason: str
    duplicate_of: str = ""
    split: str = "rejected"


@dataclass(frozen=True)
class FoundryReport:
    """The accepted train/holdout split and the audit trail."""

    records: Tuple[TaskRecord, ...]
    train: Tuple[TaskCandidate, ...]
    holdout: Tuple[TaskCandidate, ...]
    verdict: str
    detail: str = ""

    @property
    def accepted(self) -> Tuple[TaskCandidate, ...]:
        return self.train + self.holdout

    @property
    def rejected(self) -> Tuple[TaskRecord, ...]:
        return tuple(row for row in self.records if not row.accepted)

    @property
    def holdout_ratio(self) -> float:
        total = len(self.accepted)
        return len(self.holdout) / total if total else 0.0

    def rejection_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self.rejected:
            counts[row.reason] = counts.get(row.reason, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": [task.task_id for task in self.train],
            "holdout": [task.task_id for task in self.holdout],
            "rejected": [{"task_id": row.candidate.task_id, "reason": row.reason}
                         for row in self.rejected],
            "verdict": self.verdict,
            "detail": self.detail,
        }


Generator = Callable[[TaskCandidate, int], Optional[TaskCandidate]]
Validator = Callable[[TaskCandidate], Any]
Contamination = Callable[[TaskCandidate], float]
Learnability = Callable[[TaskCandidate], float]
Similarity = Callable[[TaskCandidate, TaskCandidate], float]


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} returned a non-finite score")
    return max(0.0, min(1.0, score))


def failure_seeds(records: Iterable[Mapping[str, Any]], *,
                  threshold: float = 0.5, prompt_key: str = "prompt",
                  reward_key: str = "reward", id_key: str = "task_id",
                  source: str = "failure") -> Tuple[TaskCandidate, ...]:
    """Extract low-reward rows as candidate seeds, deterministically.

    Rows without a prompt are skipped; a missing id gets a stable id derived
    from the source and prompt. This is intentionally permissive because
    trajectory stores differ in their field names; the caller can pass its own
    seeds instead.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    seeds = []
    for index, row in enumerate(records):
        prompt = row.get(prompt_key)
        if not prompt or reward_key not in row:
            continue
        raw_reward = row.get(reward_key)
        if raw_reward is None:
            continue
        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {raw_reward!r}")
        if reward >= threshold:
            continue
        task_id = row.get(id_key) or f"{source}-{stable_hash((source, prompt)):016x}"
        difficulty = min(1.0, max(0.0, 1.0 - reward))
        seeds.append(TaskCandidate(
            task_id=str(task_id), prompt=str(prompt), payload=dict(row),
            source=source, difficulty=difficulty,
            metadata={"source_index": index, "reward": reward}))
    return tuple(seeds)


class TaskFoundry:
    """Generate, validate, deduplicate, filter and split task candidates."""

    def __init__(self, config: FoundryConfig, *,
                 generator: Generator,
                 validator: Validator,
                 contamination: Optional[Contamination] = None,
                 learnability: Optional[Learnability] = None,
                 similarity: Optional[Similarity] = None) -> None:
        if not callable(generator) or not callable(validator):
            raise TypeError("generator and validator must be callable")
        for name, fn in (("contamination", contamination),
                         ("learnability", learnability),
                         ("similarity", similarity)):
            if fn is not None and not callable(fn):
                raise TypeError(f"{name} must be callable or None")
        self.config = config
        self.generator = generator
        self.validator = validator
        self.contamination = contamination or (lambda _candidate: 0.0)
        self.learnability = learnability or (lambda _candidate: 1.0)
        self.similarity = similarity

    def run(self, seeds: Iterable[TaskCandidate]) -> FoundryReport:
        config = self.config
        records: List[TaskRecord] = []
        accepted: List[TaskCandidate] = []
        fingerprints: Dict[str, str] = {}
        seen_ids = set()
        for seed in seeds:
            for attempt in range(config.per_seed):
                if len(records) >= config.max_candidates:
                    break
                try:
                    candidate = self.generator(seed, attempt)
                except Exception as exc:  # noqa: BLE001 - record, don't abort
                    records.append(TaskRecord(seed, None, None, None, False,
                                              f"generator-error:{type(exc).__name__}"))
                    continue
                if candidate is None:
                    continue
                if not isinstance(candidate, TaskCandidate):
                    raise TypeError(
                        f"generator returned {type(candidate).__name__}, expected TaskCandidate")
                if candidate.task_id in seen_ids:
                    records.append(TaskRecord(candidate, None, None, None, False,
                                              "duplicate-id", duplicate_of=candidate.task_id))
                    continue
                seen_ids.add(candidate.task_id)
                if candidate.difficulty > config.max_difficulty:
                    records.append(TaskRecord(candidate, None, None, None, False,
                                              "too-difficult"))
                    continue
                try:
                    validation = _score(self.validator(candidate), "validator")
                except Exception as exc:  # noqa: BLE001
                    records.append(TaskRecord(candidate, None, None, None, False,
                                              f"validator-error:{type(exc).__name__}"))
                    continue
                if validation < config.min_validation:
                    records.append(TaskRecord(candidate, validation, None, None,
                                              False, "invalid"))
                    continue
                fingerprint = _normalise(candidate.prompt)
                if fingerprint in fingerprints:
                    records.append(TaskRecord(candidate, validation, None, None,
                                              False, "duplicate",
                                              duplicate_of=fingerprints[fingerprint]))
                    continue
                duplicate_of = ""
                if self.similarity is not None:
                    try:
                        for existing in accepted:
                            similarity = float(self.similarity(candidate, existing))
                            if not math.isfinite(similarity):
                                raise ValueError("similarity returned a non-finite score")
                            if similarity >= config.near_duplicate_threshold:
                                duplicate_of = existing.task_id
                                break
                    except Exception as exc:  # noqa: BLE001
                        records.append(TaskRecord(candidate, validation, None, None,
                                                  False,
                                                  f"similarity-error:{type(exc).__name__}"))
                        continue
                if duplicate_of:
                    records.append(TaskRecord(candidate, validation, None, None,
                                              False, "near-duplicate",
                                              duplicate_of=duplicate_of))
                    continue
                try:
                    contamination = _score(self.contamination(candidate), "contamination")
                except Exception as exc:  # noqa: BLE001
                    records.append(TaskRecord(candidate, validation, None, None,
                                              False,
                                              f"contamination-error:{type(exc).__name__}"))
                    continue
                if contamination > config.max_contamination:
                    records.append(TaskRecord(candidate, validation, None, contamination,
                                              False, "contaminated"))
                    continue
                try:
                    learnability = _score(self.learnability(candidate), "learnability")
                except Exception as exc:  # noqa: BLE001
                    records.append(TaskRecord(candidate, validation, None, contamination,
                                              False,
                                              f"learnability-error:{type(exc).__name__}"))
                    continue
                if learnability < config.min_learnability:
                    records.append(TaskRecord(candidate, validation, learnability,
                                              contamination, False, "unlearnable"))
                    continue
                if learnability > config.max_learnability:
                    records.append(TaskRecord(candidate, validation, learnability,
                                              contamination, False, "too-easy"))
                    continue
                fingerprints[fingerprint] = candidate.task_id
                accepted.append(candidate)
                records.append(TaskRecord(candidate, validation, learnability,
                                          contamination, True, "accepted"))

        accepted.sort(key=lambda task: (stable_hash(task.task_id), task.task_id))
        percent = int(round(float(config.holdout_fraction) * 10_000))
        holdout_ids = {
            task.task_id for task in accepted
            if percent > 0 and stable_hash((config.holdout_salt, task.task_id)) % 10_000 < percent
        }
        train = tuple(task for task in accepted if task.task_id not in holdout_ids)
        holdout = tuple(task for task in accepted if task.task_id in holdout_ids)
        records = [
            TaskRecord(row.candidate, row.validation, row.learnability,
                       row.contamination, row.accepted, row.reason,
                       row.duplicate_of,
                       ("holdout" if row.candidate.task_id in holdout_ids else "train")
                       if row.accepted else "rejected")
            for row in records
        ]
        if len(accepted) < config.min_tasks:
            verdict = "insufficient"
        elif any(row.reason in ("contaminated", "unlearnable", "too-easy",
                                "near-duplicate")
                 or row.reason.startswith(("generator-error", "similarity-error",
                                           "contamination-error", "learnability-error"))
                 for row in records):
            verdict = "filtered"
        elif config.holdout_fraction > 0 and not holdout:
            verdict = "no-holdout"
        elif not train:
            verdict = "no-train"
        else:
            verdict = "ready"
        detail = (f"{len(records)} candidate(s), {len(accepted)} accepted, "
                  f"{len(train)} train / {len(holdout)} holdout")
        return FoundryReport(records=tuple(records), train=train, holdout=holdout,
                             verdict=verdict, detail=detail)
