"""Compose credit, preference and experience into one adaptive cycle.

Three modules already own the pieces of cross-cycle learning:

* :class:`~agentdescent.counterfactual_experience.CounterfactualExperience`
  turns Shapley step credit into an :class:`ExperienceRecord`;
* :class:`~agentdescent.preference_replay.PreferenceReplay` uses a learned
  preference model to rank which historical records are worth replaying;
* :class:`~agentdescent.experience.ExperienceEngine` is the only owner of
  ingestion, updates and the forgetting guard.

This module composes them without reimplementing any of their scoring:

    adaptive = AdaptiveExperience(engine, replay=replay, bridge=bridge)
    report = adaptive.cycle(records, trace=trace, features=(0.2, 0.8))
    report.ingested, report.plan.source, report.update.accepted

`ingest` deduplicates by ``run_id`` against the engine's store. `plan` chooses
validation anchors from the preference-ranked replay, a feature recall, or the
last historical rows, falling back to self-validation so the first cycle can
warm up. `cycle` performs both and delegates the guarded update to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .counterfactual_experience import CounterfactualExperience
from .experience import ExperienceEngine, ExperienceRecord, ExperienceUpdate
from .preference_replay import PreferenceReplay

__all__ = [
    "ExperiencePlan",
    "AdaptiveExperienceReport",
    "AdaptiveExperience",
]


@dataclass(frozen=True)
class ExperiencePlan:
    """The validation slice, the anchors chosen for it and why."""

    validation: Tuple[ExperienceRecord, ...]
    anchors: Tuple[ExperienceRecord, ...]
    scores: Mapping[str, float]
    source: str
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "validation", tuple(self.validation))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        if not isinstance(self.scores, Mapping):
            raise TypeError("scores must be a mapping")
        object.__setattr__(
            self, "scores",
            {str(key): float(value) for key, value in self.scores.items()})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation": [record.to_dict() for record in self.validation],
            "anchors": [record.to_dict() for record in self.anchors],
            "scores": dict(self.scores),
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AdaptiveExperienceReport:
    """What a cycle ingested, the anchors it planned and the guarded update."""

    ingested: Tuple[str, ...]
    plan: ExperiencePlan
    update: ExperienceUpdate
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ingested", tuple(self.ingested))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ingested": list(self.ingested),
            "plan": self.plan.to_dict(),
            "update": {
                "version": self.update.version,
                "accepted": self.update.accepted,
                "before": self.update.before,
                "after": self.update.after,
                "drift": self.update.drift,
                "anchor_before": self.update.anchor_before,
                "anchor_after": self.update.anchor_after,
                "rolled_back": self.update.rolled_back,
                "reason": self.update.reason,
            },
            "detail": self.detail,
        }


def _records(values: Iterable[ExperienceRecord], name: str) -> Tuple[ExperienceRecord, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be an iterable of ExperienceRecord")
    rows = tuple(values)
    if any(not isinstance(row, ExperienceRecord) for row in rows):
        raise TypeError(f"{name} must contain ExperienceRecord objects")
    return rows


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an int >= 1")


class AdaptiveExperience:
    """Run one ingest -> select -> guarded-update cycle over the engine."""

    def __init__(self, engine: ExperienceEngine, *,
                 replay: Optional[PreferenceReplay] = None,
                 bridge: Optional[CounterfactualExperience] = None) -> None:
        if not isinstance(engine, ExperienceEngine):
            raise TypeError("engine must be an ExperienceEngine")
        if replay is not None and not isinstance(replay, PreferenceReplay):
            raise TypeError("replay must be a PreferenceReplay or None")
        if bridge is not None and not isinstance(bridge, CounterfactualExperience):
            raise TypeError("bridge must be a CounterfactualExperience or None")
        self.engine = engine
        self.replay = replay
        self.bridge = bridge

    def ingest(self, records: Iterable[ExperienceRecord]) -> Tuple[str, ...]:
        """Ingest new records, deduplicating by run_id against the store."""
        rows = _records(records, "records")
        known = {row.run_id for row in self.engine.store.rows}
        ingested = []
        for row in rows:
            if row.run_id in known:
                continue
            self.engine.ingest(row)
            known.add(row.run_id)
            ingested.append(row.run_id)
        return tuple(ingested)

    def _preference_scores(
            self, anchors: Sequence[ExperienceRecord]) -> Dict[str, float]:
        if self.replay is None:
            return {}
        scores = {}
        for row in anchors:
            features = (row.features if self.replay.featurize is None
                        else self.replay.featurize(row))
            try:
                scores[row.run_id] = float(self.replay.model.score(features))
            except (TypeError, ValueError):
                continue
        return scores

    def plan(self, validation: Sequence[ExperienceRecord], *, k: int = 5,
             features: Optional[Sequence[float]] = None) -> ExperiencePlan:
        """Choose anchors for the current validation slice."""
        rows = _records(validation, "validation")
        if not rows:
            raise ValueError("validation must not be empty")
        _check_k(k)

        anchors: Tuple[ExperienceRecord, ...] = ()
        scores: Dict[str, float] = {}
        source = ""
        if self.replay is not None:
            anchors = tuple(self.replay.rank(k=k))
            if anchors:
                source = "preference"
                scores = self._preference_scores(anchors)
        elif features is not None:
            try:
                hits = self.engine.recall(features, k=k)
            except ValueError:
                hits = ()
            anchors = tuple(hit.record for hit in hits)
            if anchors:
                source = "recall"
        else:
            current = {row.run_id for row in rows}
            history = [row for row in self.engine.store.rows
                       if row.run_id not in current]
            anchors = tuple(history[-k:])
            if anchors:
                source = "history"

        if not anchors:
            anchors = rows[:1]
            source = "self-validation"
            scores = {}
        detail = (f"{len(rows)} validation row(s), {len(anchors)} anchor(s), "
                  f"source={source}")
        return ExperiencePlan(
            validation=rows, anchors=anchors, scores=scores,
            source=source, detail=detail)

    def cycle(self, records: Iterable[ExperienceRecord] = (), *,
              validation: Optional[Sequence[ExperienceRecord]] = None,
              anchors: Optional[Sequence[ExperienceRecord]] = None,
              trace: Any = None,
              features: Optional[Sequence[float]] = None,
              k: int = 5, accepted: bool = True, cost: float = 0.0,
              artifact_id: str = "",
              metadata: Optional[Mapping[str, Any]] = None) -> AdaptiveExperienceReport:
        """Ingest, select validation and anchors, then update the engine."""
        _check_k(k)
        if trace is not None and self.bridge is None:
            raise ValueError("trace requires a CounterfactualExperience bridge")
        if trace is not None and features is None:
            raise ValueError("features are required when a trace is given")

        incoming = _records(records, "records")
        new_ids = self.ingest(incoming)
        new_set = set(new_ids)
        fresh = [row for row in incoming if row.run_id in new_set]
        ingested = list(new_ids)

        if trace is not None:
            bridge_report = self.bridge.ingest(
                trace, features, accepted=accepted, cost=cost,
                artifact_id=artifact_id, metadata=metadata)
            record = bridge_report.record
            if record.run_id not in ingested:
                ingested.append(record.run_id)
            fresh.append(record)

        if validation is not None:
            rows = _records(validation, "validation")
            if not rows:
                raise ValueError("validation must not be empty")
        elif fresh:
            seen = set()
            rows = tuple(
                row for row in fresh
                if not (row.run_id in seen or seen.add(row.run_id)))
        else:
            rows = self.engine.store.rows[-k:]
            if not rows:
                raise ValueError("no validation rows available for the cycle")

        if anchors is not None:
            anchor_rows = _records(anchors, "anchors")
            plan = ExperiencePlan(
                validation=rows, anchors=anchor_rows, scores={},
                source="explicit",
                detail=f"{len(rows)} validation row(s), "
                       f"{len(anchor_rows)} explicit anchor(s)")
        else:
            plan = self.plan(rows, k=k, features=features)

        update = self.engine.update(
            validation=plan.validation, anchors=plan.anchors)
        detail = (f"{len(ingested)} ingested, validation={len(plan.validation)}, "
                  f"anchors={len(plan.anchors)}, source={plan.source}, "
                  f"accepted={update.accepted}, rolled_back={update.rolled_back}")
        return AdaptiveExperienceReport(
            ingested=tuple(ingested), plan=plan, update=update, detail=detail)
