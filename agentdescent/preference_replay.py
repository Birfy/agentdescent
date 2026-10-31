"""Use a learned preference model to choose which experience to replay.

`preference.PreferenceModel` learns a scalar from pairwise human preferences;
`experience.ExperienceStore` is a memory of past runs. Nothing used the former
to decide what the latter should replay. This module ranks stored experience by
the preference score and ingests the preferred slice into an `ExperienceEngine`.

    ranker = PreferenceReplay(preference_model, store)
    report = ranker.replay(engine, k=20)
    report.selected, report.ingested

The engine and its replay buffer remain the only update path. The preference
model is the selection policy: a run with features the human prefers is worth
replaying before a run the human merely tolerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .experience import ExperienceEngine, ExperienceRecord, ExperienceStore
from .preference import PreferenceModel

__all__ = ["PreferenceReplayReport", "PreferenceReplay"]


@dataclass(frozen=True)
class PreferenceReplayReport:
    """The selected records, their scores and what the engine ingested."""

    selected: Tuple[ExperienceRecord, ...]
    ingested: Tuple[str, ...]
    scores: Mapping[str, float] = None
    skipped: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", dict(self.scores or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": [record.run_id for record in self.selected],
            "ingested": list(self.ingested),
            "scores": dict(self.scores),
            "skipped": self.skipped,
            "detail": self.detail,
        }


class PreferenceReplay:
    """Rank stored experience by a preference model and replay the top slice."""

    def __init__(self, model: PreferenceModel,
                 store: ExperienceStore, *,
                 featurize: Optional[Callable[[ExperienceRecord], Sequence[float]]] = None) -> None:
        if not isinstance(model, PreferenceModel):
            raise TypeError("model must be a PreferenceModel")
        if not isinstance(store, ExperienceStore):
            raise TypeError("store must be an ExperienceStore")
        if model.report is None:
            raise ValueError("PreferenceReplay needs a fitted PreferenceModel")
        if featurize is not None and not callable(featurize):
            raise TypeError("featurize must be callable or None")
        self.model = model
        self.store = store
        self.featurize = featurize

    def _features(self, record: ExperienceRecord) -> Sequence[float]:
        return (record.features if self.featurize is None
                else self.featurize(record))

    def _ranked(self, k: Optional[int]) -> Tuple[Tuple[ExperienceRecord, ...],
                                                 Dict[str, float], int]:
        if k is not None and (isinstance(k, bool) or not isinstance(k, int)
                              or k < 1):
            raise ValueError("k must be a positive int or None")
        scored = []
        skipped = 0
        for record in self.store.rows:
            try:
                score = float(self.model.score(self._features(record)))
            except ValueError:
                skipped += 1
                continue
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].run_id))
        rows = tuple(record for _, record in scored)
        if k is not None:
            rows = rows[:k]
        scores = {record.run_id: score for score, record in scored
                  if record in rows}
        return rows, scores, skipped

    def rank(self, *, k: Optional[int] = None) -> Tuple[ExperienceRecord, ...]:
        return self._ranked(k)[0]

    def replay(self, engine: ExperienceEngine, *, k: int = 5) -> PreferenceReplayReport:
        if not isinstance(engine, ExperienceEngine):
            raise TypeError("engine must be an ExperienceEngine")
        if engine.store is self.store:
            raise ValueError(
                "PreferenceReplay needs a source store separate from the "
                "engine's replay store; the engine already warms from its own store")
        selected, scores, skipped = self._ranked(k)
        known = {row.run_id for row in engine.store.rows}
        ingested = []
        for record in selected:
            if record.run_id not in known:
                engine.ingest(record)
                known.add(record.run_id)
                ingested.append(record.run_id)
        detail = (f"{len(selected)} selected, {len(ingested)} new, "
                  f"{skipped} skipped, replay={len(engine.store)}")
        return PreferenceReplayReport(
            selected, tuple(ingested), scores, skipped, detail)
