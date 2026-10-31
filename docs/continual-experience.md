# Continual experience — learn across runs without forgetting

`continual_meta` updates a meta-policy with replay and validation.
`ExperienceEngine` adds the cross-run memory around it: an append-only store,
nearest-neighbour recall, and a frozen anchor slice that can roll an update back.

```python
from agentdescent import (
    ContinualMetaLearner, ExperienceEngine, ExperienceRecord, ExperienceStore,
)

learner = ContinualMetaLearner(["direct", "search"], dim=4, min_gain=0.0)
engine = ExperienceEngine(learner, ExperienceStore("experience.jsonl"))

engine.ingest(ExperienceRecord(
    "run-1", features=(0.2, 0.8, 0.1, 0.5),
    rewards={"direct": 0.4, "search": 0.7}, cost=2.0,
    artifact_id="skill-1"))

hits = engine.recall((0.2, 0.8, 0.1, 0.5), k=3)
update = engine.update(validation=validation_rows, anchors=anchor_rows)
update.accepted, update.rolled_back, update.anchor_after, update.reason
```

## The store

`ExperienceRecord` is one run's context features plus the reward of each action.
`ExperienceStore` appends JSONL (optionally on disk), reloads on construction,
and skips corrupt lines while counting them in `store.skipped`. `recall()`
returns nearest records with the same feature dimension; records with another
dimension are skipped, and it raises `ValueError` when no record matches the
query dimension.

## The update and the forgetting guard

`ExperienceEngine.update(validation=..., anchors=...)`:

1. computes the current policy's reward on the frozen anchors;
2. calls `ContinualMetaLearner.step()` on the validation slice;
3. if the candidate improved validation, measures the anchors again;
4. if the anchors regressed by more than `anchor_tolerance`, calls
   `ContinualMetaLearner.rollback()`.

`anchor_after` keeps the **candidate's** anchor score even when the update is
rolled back, so the report shows exactly how far it regressed. The reason quotes
candidate, before and tolerance.

`rollback()` restores the exact policy weights from before the update and keeps
the attempt in the audit trail as a `rejected` / `rolled_back` update; it does
not pretend the update never happened.

## Restart and scope

Constructing an `ExperienceEngine` warms the learner's replay from the existing
store (no row is re-written). The learned policy weights themselves are not
persisted here; serialize them separately if another process must resume the
exact policy.

The store is append-only and bounded only by the caller's retention policy; the
learner's `ReplayBuffer` is the bounded hot set (`replay_size`).


## Preference-ranked replay

`PreferenceReplay` uses a fitted `PreferenceModel` as the source-selection
policy:

```python
replay = PreferenceReplay(preference_model, source_store)
report = replay.replay(engine, k=20)
report.selected, report.scores, report.skipped
```

The model must be fitted on the same feature space as the records it ranks;
`featurize=` can adapt a record. Records with another feature dimension are
skipped (and counted), never crash the ranking. The source store must be
separate from the engine's replay store; the engine already warms from its own
store, so replaying the same store is a silent no-op and is refused.
