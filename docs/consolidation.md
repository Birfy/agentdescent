# Cross-run consolidation

`ConsolidationStore` archives what a run produced; `seed_from_archive`
retrieves it for the next run — gated. This is the difference between
self-improvement (one run) and *continuing* self-improvement (skills that seed
the next run).

## Why

An evolution run starts from an empty artifact, learns, and then the skill dies
with the process that made it. The trajectory ledger records *what happened*
during a run; the archive records *what it produced* — the artifact that
survived, its measured reward, the configuration that made it, and the
trajectories that justify it — so a later run can adopt it.

## The pieces

| Piece | What it is |
| --- | --- |
| `ConsolidationStore(path)` | A durable archive: one JSON record per run under `<path>/runs/<sha256(run_id)>.json` (the complete id is hashed, so `a/b` and `ab` never collide; the full `run_id` is stored in the record, and a pre-#338 sanitized `<run_id>.json` is still found by `find` when its stored id matches). Atomic, idempotent by `run_id`, thread-safe. |
| `ConsolidationStore.archive(result, *, run_id, artifact_id, strategy, config, trajectory_store)` | Writes a run's product (the artifact's `state`, `rendered`, `final_reward`, rounds) plus the caller's `config` (dataset/scorer/model/params) and a durable `trajectory_store` path — **linked, not copied**. |
| `seed_from_archive(store, tasks, run, reward, *, strategy, config, sample, min_reward, seed)` | Retrieves a candidate and returns `(initial_state, SeedReport)`. Pass `initial_state` to `evolve(initial_state=...)`. |

## The guards

* **Mini-eval gate.** The archived artifact and the empty seed are both scored
  on a seeded sample of the *new* tasks; the archived state is handed over only
  if it does not regress beyond `min_reward` against the empty baseline. A
  harmful cross-domain seed is refused and reported, never installed.
* **Same configuration first.** Archived runs whose `config` fingerprint
  matches the new run's rank first, then by measured reward. `config` is how
  the caller keeps domains apart; a run without a config matches nothing.
* **Same strategy.** A new run adopts the state only through a strategy of the
  same name (its `to_diff`/`render` must understand the keys).
* **Non-finite rewards refused.** A run with a `NaN`/out-of-range
  `final_reward` is not archived.

## Example

```bash
python -m examples.consolidation.consolidation_demo
```

Run 1 evolves a capability-capped skill and archives it; run 2 (new tasks)
retrieves it through the gate and reaches the quality bar ~200× sooner than the
empty seed — measured, deterministic, offline.
