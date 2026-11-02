# Experience — inspect and update cross-run memory

`agentdescent experience` is the product surface for an `ExperienceStore`
JSONL and the guarded `ExperienceEngine` update:

```bash
agentdescent experience report  --store experience.jsonl --json
agentdescent experience ingest  --store experience.jsonl --records rows.jsonl
agentdescent experience recall  --store experience.jsonl --features 0.2,0.8 --k 5
agentdescent experience update  --store experience.jsonl \
    --validation current.jsonl --anchors anchors.jsonl \
    --actions edit,retry --dim 2
agentdescent experience replay  --source old.jsonl --store replay.jsonl \
    --pairs preferences.jsonl --actions edit,retry --dim 2 --k 5
```

`--store` here is the experience JSONL, separate from the global run store.
The default is `$AGENTDESCENT_HOME/experience.jsonl`.

## Verbs

| verb | what it does |
|---|---|
| `report` | rows, accepted count, action histogram, mean action reward, last run, dimensions |
| `ingest` | append `ExperienceRecord` rows, deduplicating by `run_id` |
| `recall` | nearest records to a feature vector; different dimensions are skipped |
| `update` | `ExperienceEngine.update` with the same forgetting guard as the library |
| `replay` | fit a Bradley–Terry preference model, rank a source store, replay the preferred slice into a separate engine store |

Records are one JSON object per line:

```json
{"run_id": "r1", "features": [0.2, 0.8],
 "rewards": {"edit": 0.9, "retry": 0.1}, "accepted": true}
```

`update` requires action names and the feature dimension; `--no-require-anchors`
is available for a warm-up. `replay` requires `--source` and `--store` to be
different files, because the engine's own store is the replay destination.
The MCP tools use the same payloads (`experience_report`, `experience_ingest`,
`experience_recall`, `experience_update`, `experience_replay`). The MCP
`store` write is confined to `AGENTDESCENT_SURFACE_WRITE_DIRS`.
