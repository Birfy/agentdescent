"""Cross-run experience: remember, recall, and update without forgetting.

`continual_meta` already updates a meta-policy with replay and validation.
What it does not own is the *cross-run* store: which runs produced which
experience, how to recall similar history for the next run, and a hard guard
that rejects an update which regresses a frozen anchor set.

This module is that small coordinator:

* :class:`ExperienceRecord` is one run's context + per-action rewards;
* :class:`ExperienceStore` is an append-only JSONL memory (optional on disk)
  with nearest-neighbour recall;
* :class:`ExperienceEngine` feeds records into a
  :class:`~agentdescent.continual_meta.ContinualMetaLearner`, updates it on a
  validation slice, and **rolls the update back** when a frozen anchor slice
  loses more than ``anchor_tolerance``.

    engine = ExperienceEngine(learner, ExperienceStore("experience.jsonl"))
    engine.ingest(record)
    hits = engine.recall(features, k=3)
    update = engine.update(validation=[...], anchors=[...])
    update.accepted, update.rolled_back, update.reason

The experience store is append-only and bounded only by the caller's retention
policy; the learner's replay buffer is the bounded hot set and is warmed from
the store on construction. A rollback restores the exact policy weights the
learner had before the rejected update. The learned policy itself is not
persisted here; callers that need it across processes should serialize the
learner's weights separately.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .continual_meta import ContinualMetaLearner, ContinualReport, MetaExample

__all__ = [
    "ExperienceRecord",
    "ExperienceHit",
    "ExperienceUpdate",
    "ExperienceReport",
    "ExperienceStore",
    "ExperienceEngine",
]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ExperienceRecord:
    """One run's experience: context features and the reward of each action."""

    run_id: str
    features: Tuple[float, ...]
    rewards: Mapping[str, float]
    cost: float = 0.0
    accepted: bool = True
    artifact_id: str = ""
    created_at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id is required")
        features = tuple(_finite(value, "feature") for value in self.features)
        if not features:
            raise ValueError("features must not be empty")
        if not isinstance(self.rewards, Mapping) or not self.rewards:
            raise ValueError("rewards must be a non-empty mapping")
        rewards = {}
        for action, value in self.rewards.items():
            if not isinstance(action, str) or not action.strip():
                raise ValueError("reward action names must be non-empty strings")
            rewards[action] = _finite(value, f"reward[{action!r}]")
        cost = _finite(self.cost, "cost")
        if cost < 0:
            raise ValueError("cost must be >= 0")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        if not isinstance(self.artifact_id, str):
            raise TypeError("artifact_id must be a string")
        created = _finite(self.created_at, "created_at")
        if created < 0:
            raise ValueError("created_at must be >= 0")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_meta_example(self) -> MetaExample:
        return MetaExample(self.features, self.rewards, self.cost,
                           self.created_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "features": list(self.features),
            "rewards": dict(self.rewards),
            "cost": self.cost,
            "accepted": self.accepted,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperienceRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("experience record must be an object")
        accepted = payload.get("accepted", True)
        if not isinstance(accepted, bool):
            raise ValueError("accepted must be a bool")
        try:
            return cls(
                run_id=str(payload["run_id"]),
                features=tuple(payload["features"]),
                rewards=dict(payload["rewards"]),
                cost=payload.get("cost", 0.0),
                accepted=payload.get("accepted", True),
                artifact_id=str(payload.get("artifact_id", "")),
                created_at=payload.get("created_at", 0.0),
                metadata=payload.get("metadata") or {})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid experience record: {exc}") from None


@dataclass(frozen=True)
class ExperienceHit:
    """One recalled record and its distance from the query."""

    record: ExperienceRecord
    distance: float


@dataclass(frozen=True)
class ExperienceUpdate:
    """The validation result plus the anchor guard's verdict."""

    version: int
    accepted: bool
    before: float
    after: float
    drift: float
    anchor_before: Optional[float] = None
    #: The anchor score the candidate policy would have produced. It is kept
    #: even when ``rolled_back`` is true, so the regression is visible.
    anchor_after: Optional[float] = None
    rolled_back: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ExperienceReport:
    rows: int
    learner: ContinualReport
    skipped: int = 0
    detail: str = ""


class ExperienceStore:
    """An append-only, optionally persistent experience memory."""

    def __init__(self, path: Optional[str] = None) -> None:
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError("path must be a non-empty string or None")
        self.path = (None if path is None else
                     os.path.abspath(os.path.expanduser(path)))
        self._lock = threading.Lock()
        self._rows: List[ExperienceRecord] = []
        self.skipped = 0
        if self.path is not None:
            self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._rows.append(
                            ExperienceRecord.from_dict(json.loads(line)))
                    except (ValueError, TypeError):
                        self.skipped += 1
        except FileNotFoundError:
            return

    def add(self, record: ExperienceRecord) -> ExperienceRecord:
        if not isinstance(record, ExperienceRecord):
            raise TypeError("add() expects ExperienceRecord")
        with self._lock:
            if self.path is not None:
                directory = os.path.dirname(self.path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record.to_dict(), sort_keys=True,
                                           allow_nan=False) + "\n")
            self._rows.append(record)
        return record

    def recall(self, features: Sequence[float], *, k: int = 5) -> Tuple[ExperienceHit, ...]:
        """Nearest records with the same feature dimension; others are skipped."""
        query = tuple(_finite(value, "feature") for value in features)
        if not query:
            raise ValueError("features must not be empty")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be an int >= 1")
        candidates = []
        for record in self._rows:
            if len(record.features) != len(query):
                continue
            distance = math.sqrt(sum((a - b) ** 2
                                     for a, b in zip(record.features, query)))
            candidates.append(ExperienceHit(record, distance))
        if not candidates:
            raise ValueError("no experience record has the query dimension")
        candidates.sort(key=lambda hit: (hit.distance, hit.record.run_id))
        return tuple(candidates[:k])

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[ExperienceRecord]:
        return iter(tuple(self._rows))

    @property
    def rows(self) -> Tuple[ExperienceRecord, ...]:
        return tuple(self._rows)


class ExperienceEngine:
    """Feed cross-run records into a continual learner with a forgetting guard."""

    def __init__(self, learner: ContinualMetaLearner, store: ExperienceStore, *,
                 require_anchors: bool = True,
                 anchor_tolerance: float = 0.0) -> None:
        if not isinstance(learner, ContinualMetaLearner):
            raise TypeError("learner must be a ContinualMetaLearner")
        if not isinstance(store, ExperienceStore):
            raise TypeError("store must be an ExperienceStore")
        if not isinstance(require_anchors, bool):
            raise TypeError("require_anchors must be a bool")
        tolerance = _finite(anchor_tolerance, "anchor_tolerance")
        if tolerance < 0:
            raise ValueError("anchor_tolerance must be >= 0")
        self.learner = learner
        self.store = store
        self.require_anchors = require_anchors
        self.anchor_tolerance = tolerance
        self.replay_from_store()

    def replay_from_store(self) -> int:
        """Warm the learner replay from the store without re-writing rows."""
        warmed = 0
        for record in self.store:
            self.learner.observe(record.to_meta_example())
            warmed += 1
        return warmed

    def ingest(self, record: ExperienceRecord) -> ExperienceRecord:
        if not isinstance(record, ExperienceRecord):
            raise TypeError("ingest() expects ExperienceRecord")
        self.store.add(record)
        self.learner.observe(record.to_meta_example())
        return record

    def recall(self, features: Sequence[float], *, k: int = 5) -> Tuple[ExperienceHit, ...]:
        return self.store.recall(features, k=k)

    def update(self, *, validation: Sequence[ExperienceRecord],
               anchors: Sequence[ExperienceRecord]) -> ExperienceUpdate:
        validation_rows = list(validation)
        anchor_rows = list(anchors)
        if not validation_rows:
            raise ValueError("validation must not be empty")
        if self.require_anchors and not anchor_rows:
            raise ValueError("anchors must not be empty when require_anchors is on")
        if any(not isinstance(row, ExperienceRecord)
               for row in validation_rows + anchor_rows):
            raise TypeError("validation and anchors must contain ExperienceRecord")
        anchor_examples = [row.to_meta_example() for row in anchor_rows]
        anchor_before = (self.learner.policy.expected_reward(anchor_examples)
                         if anchor_examples else None)
        meta = self.learner.step([row.to_meta_example()
                                  for row in validation_rows])
        anchor_after = (self.learner.policy.expected_reward(anchor_examples)
                        if anchor_examples and meta.accepted else anchor_before)
        rolled_back = False
        reason = meta.reason
        accepted = meta.accepted
        if (meta.accepted and anchor_before is not None
                and anchor_after is not None
                and anchor_after < anchor_before - self.anchor_tolerance):
            rolled_back = self.learner.rollback()
            if rolled_back:
                accepted = False
                # Keep the candidate's anchor score in the report; the caller
                # can see exactly how far it regressed before the rollback.
                reason = (f"forgetting guard: anchor candidate "
                          f"{anchor_after:.6g} < before {anchor_before:.6g} - "
                          f"tolerance {self.anchor_tolerance:.6g}")
        return ExperienceUpdate(
            version=self.learner.version, accepted=accepted,
            before=meta.before, after=meta.after, drift=meta.drift,
            anchor_before=anchor_before, anchor_after=anchor_after,
            rolled_back=rolled_back, reason=reason)

    def act(self, features: Sequence[float], *, epsilon: float = 0.0) -> str:
        return self.learner.act(features, epsilon=epsilon)

    def report(self) -> ExperienceReport:
        learned = self.learner.report()
        return ExperienceReport(
            rows=len(self.store), learner=learned, skipped=self.store.skipped,
            detail=(f"{len(self.store)} experience row(s), "
                    f"{self.store.skipped} skipped, {learned.detail}"))
