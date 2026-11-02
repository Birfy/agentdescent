"""The ``agentdescent experience`` surface: inspect and update cross-run memory.

`experience.ExperienceStore` is already an append-only JSONL memory, and
`ExperienceEngine` already owns the replay/forget-guard update. This module is
the product surface around that pair: report, ingest, recall, update and
preference replay, with the same envelope for the CLI and the MCP server.

    agentdescent experience report  --store experience.jsonl --json
    agentdescent experience ingest  --store experience.jsonl --records rows.jsonl
    agentdescent experience recall  --store experience.jsonl --features 0.2,0.8 --k 5
    agentdescent experience update  --store experience.jsonl \
        --validation current.jsonl --anchors anchors.jsonl \
        --actions edit,retry --dim 2
    agentdescent experience replay  --source old.jsonl --store replay.jsonl \
        --pairs preferences.jsonl --actions edit,retry --dim 2

The store is a *separate* file from the run store; ``--store`` here means the
experience JSONL. Records are validated with `ExperienceRecord.from_dict`, and
ingest deduplicates by `run_id`.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Tuple

from .continual_meta import ContinualMetaLearner
from .experience import ExperienceEngine, ExperienceRecord, ExperienceStore
from .preference import PreferenceModel, PreferencePair
from .preference_replay import PreferenceReplay
from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, load_jsonl,
    parse_csv, parse_floats, require_mapping, require_text,
)

__all__ = [
    "experience_ingest",
    "experience_recall",
    "experience_replay",
    "experience_report",
    "experience_update",
]


def default_store() -> str:
    home = os.environ.get("AGENTDESCENT_HOME") or "~/.agentdescent"
    return os.path.join(os.path.abspath(os.path.expanduser(home)),
                        "experience.jsonl")


def _store_path(value: Any) -> str:
    if value is None:
        return default_store()
    return os.path.abspath(os.path.expanduser(require_text(value, "store")))


def load_records(value: Any, *, name: str) -> Tuple[ExperienceRecord, ...]:
    rows = []
    for index, row in enumerate(load_jsonl(value, name=name)):
        try:
            rows.append(ExperienceRecord.from_dict(row))
        except (TypeError, ValueError, KeyError) as error:
            raise SurfaceError(
                f"{name}[{index}] is not an ExperienceRecord: {error}") from None
    return tuple(rows)


def _summary(store: ExperienceStore) -> Dict[str, Any]:
    actions: Counter = Counter()
    accepted = 0
    reward_total = 0.0
    reward_n = 0
    for row in store.rows:
        accepted += int(row.accepted)
        for action, reward in row.rewards.items():
            actions[str(action)] += 1
            reward_total += float(reward)
            reward_n += 1
    return {
        "rows": len(store),
        "skipped": store.skipped,
        "accepted": accepted,
        "actions": dict(sorted(actions.items())),
        "mean_action_reward": (reward_total / reward_n) if reward_n else None,
        "last_run": store.rows[-1].run_id if store.rows else None,
        "dimensions": sorted({len(row.features) for row in store.rows}),
    }


def experience_report(*, store: Any = None) -> Dict[str, Any]:
    path = _store_path(store)
    memory = ExperienceStore(path)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "store": path,
        "exists": os.path.isfile(path),
        "summary": _summary(memory),
        "next": ["agentdescent experience ingest --store <store> --records <rows>"],
    }


def experience_ingest(*, store: Any = None, records: Any) -> Dict[str, Any]:
    path = _store_path(store)
    memory = ExperienceStore(path)
    incoming = load_records(records, name="records")
    known = {row.run_id for row in memory.rows}
    ingested: List[str] = []
    for row in incoming:
        if row.run_id in known:
            continue
        memory.add(row)
        known.add(row.run_id)
        ingested.append(row.run_id)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "store": path,
        "ingested": ingested,
        "existing": len(incoming) - len(ingested),
        "summary": _summary(memory),
    }


def experience_recall(*, store: Any = None, features: Any, k: int = 5) -> Dict[str, Any]:
    path = _store_path(store)
    memory = ExperienceStore(path)
    query = parse_floats(features, name="features")
    if not query:
        raise SurfaceError("features must contain at least one number")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SurfaceError("k must be an int >= 1")
    try:
        hits = memory.recall(query, k=k)
    except ValueError as error:
        raise SurfaceError(
            f"cannot recall from {path!r}: {error}; ingest records with "
            f"{len(query)} feature(s) first") from None
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "store": path,
        "features": list(query),
        "k": k,
        "hits": [
            {"distance": hit.distance, "record": hit.record.to_dict()}
            for hit in hits
        ],
    }


def _learner(actions: Any, *, dim: int, learning_rate: float = 0.2,
             l2: float = 1e-3, replay_size: int = 1000,
             min_gain: float = 0.0, seed: int = 0) -> ContinualMetaLearner:
    names = list(parse_csv(actions, name="actions"))
    if not names:
        raise SurfaceError("actions must contain at least one action name")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise SurfaceError("dim must be an int >= 1")
    try:
        return ContinualMetaLearner(
            names, dim=dim, learning_rate=learning_rate, l2=l2,
            replay_size=replay_size, min_gain=min_gain, seed=seed)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"invalid learner settings: {error}") from None


def experience_update(
        *, store: Any = None, validation: Any, anchors: Any,
        actions: Any, dim: int, learning_rate: float = 0.2,
        l2: float = 1e-3, replay_size: int = 1000, min_gain: float = 0.0,
        seed: int = 0, require_anchors: bool = True,
        anchor_tolerance: float = 0.0) -> Dict[str, Any]:
    path = _store_path(store)
    memory = ExperienceStore(path)
    engine = ExperienceEngine(
        _learner(actions, dim=dim, learning_rate=learning_rate, l2=l2,
                 replay_size=replay_size, min_gain=min_gain, seed=seed),
        memory, require_anchors=require_anchors,
        anchor_tolerance=anchor_tolerance)
    val = load_records(validation, name="validation")
    anc = load_records(anchors, name="anchors") if anchors else ()
    if not val:
        raise SurfaceError("validation must not be empty")
    try:
        update = engine.update(validation=val, anchors=anc)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"update refused: {error}") from None
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "store": path,
        "update": {
            "version": update.version,
            "accepted": update.accepted,
            "before": update.before,
            "after": update.after,
            "drift": update.drift,
            "anchor_before": update.anchor_before,
            "anchor_after": update.anchor_after,
            "rolled_back": update.rolled_back,
            "reason": update.reason,
        },
        "summary": _summary(memory),
    }


def _pairs(value: Any) -> Tuple[PreferencePair, ...]:
    rows = []
    for index, row in enumerate(load_jsonl(value, name="pairs")):
        data = require_mapping(row, f"pairs[{index}]")
        try:
            rows.append(PreferencePair(
                tuple(float(x) for x in data["left"]),
                tuple(float(x) for x in data["right"]),
                str(data.get("preferred", "left"))))
        except (KeyError, TypeError, ValueError) as error:
            raise SurfaceError(f"pairs[{index}] is invalid: {error}") from None
    return tuple(rows)


def experience_replay(
        *, source: Any, store: Any = None, pairs: Any,
        actions: Any, dim: int, k: int = 5, learning_rate: float = 0.2,
        l2: float = 1e-3, replay_size: int = 1000, min_gain: float = 0.0,
        seed: int = 0) -> Dict[str, Any]:
    source_path = os.path.abspath(os.path.expanduser(require_text(source, "source")))
    if not os.path.isfile(source_path):
        raise SurfaceError(f"source {source!r} does not exist")
    target = _store_path(store)
    if os.path.abspath(target) == source_path:
        raise SurfaceError(
            "source and store must be different files; replay writes into "
            "the engine's own store")
    preferences = _pairs(pairs)
    if not preferences:
        raise SurfaceError("pairs must not be empty")
    try:
        model = PreferenceModel()
        model.fit(preferences)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"cannot fit preference model: {error}") from None
    engine = ExperienceEngine(
        _learner(actions, dim=dim, learning_rate=learning_rate, l2=l2,
                 replay_size=replay_size, min_gain=min_gain, seed=seed),
        ExperienceStore(target))
    report = PreferenceReplay(
        model, ExperienceStore(source_path)).replay(engine, k=k)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "source": source_path,
        "store": target,
        "replay": report.to_dict(),
        "summary": _summary(engine.store),
    }
