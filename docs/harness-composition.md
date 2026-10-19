# Harness composition

`Harness` names an algorithm port; the consolidation archive is the
leaderboard; `select_harness` routes a new domain to the port the recorded
evidence says wins there. This is the composition layer: the ports themselves
(`evolve()` with different `Policies` / `aggregator_factory` — RRSI,
solver-challenger self-play, population islands, metasearch) never change, and
nothing before this chose between them or remembered which one won where.

## The pieces

| Piece | What it is |
| --- | --- |
| `Harness(name, run)` | A named `(tasks, repo_path) -> EvolutionResult` callable. The caller binds everything the port needs into the closure, so a harness is a black box that turns a task set into a measured result. |
| `record_harness(store, *, harness_name, config, result, run_id)` | Writes the outcome to the archive under `artifact_id="harness:<name>"` with the domain's `config` — one ordinary `ConsolidationRecord`, atomic and idempotent. |
| `harness_score(store, name, config)` | The best measured reward for that harness on that config fingerprint, `None` when there is no evidence. |
| `harness_leaderboard(store, config)` | `name -> {rewards, best}` over the archive, per domain or aggregated. |
| `select_harness(store, harnesses, config, *, default)` | The harness with the highest recorded reward for the config; round-robin (or `default`) when no evidence exists — a harness with no runs on a domain can never beat one that was measured there. |
| `compose_run(store, tasks, harnesses, config, *, mode, repo_dir)` | Runs the selected harness (`mode="best"`) or every harness (`mode="all"` — the leaderboard's training mode), each on a fresh ledger repo, records the outcomes, and returns a `HarnessReport` (per-harness reward / time-to-quality, the winner, and the selection reason). |

## The loop

1. **Measure.** `compose_run(..., mode="all")` runs every harness on a task set
   and records each outcome for the domain's config fingerprint.
2. **Route.** `select_harness` reads the archive: same-config evidence first,
   best measured reward wins. A tie goes to the harness listed first; no
   evidence falls back to round-robin rather than a guess — the reason string
   says which way the choice went, so it can be logged.
3. **Re-measure.** `compose_run(..., mode="best")` runs only the chosen harness,
   records the rerun, and the next route on the same domain sees the updated
   evidence — the composition learns which algorithm wins where, from measured
   runs, without touching any port.

## Domain separation

`config_fingerprint` is the archive's existing domain key, and `select_harness`
uses it exactly as `seed_from_archive` does: a harness that wins on arithmetic
is never mistaken for one that wins on code, and evidence on one domain never
leaks into the choice on another.

## Example

Two deterministic harnesses on the same arithmetic domain, one shallow
(`rounds=4`), one deep (`rounds=8`):

```python
from agentdescent import ConsolidationStore, compose_run, harness

store = ConsolidationStore("~/.agentdescent/archive")
quick = harness("quick", lambda tasks, repo: evolve(tasks, reward, run=run,
              propose=propose, strategy=SingleSlot("cap:0"), rounds=4,
              repo_path=repo))
deep = harness("deep", lambda tasks, repo: evolve(tasks, reward, run=run,
             propose=propose, strategy=SingleSlot("cap:0"), rounds=8,
             repo_path=repo))

report, winner = compose_run(store, tasks, [quick, deep],
                             {"dataset": "cap", "scorer": "exact"}, mode="all")
print(report.winner, report.reason)
# the next compose_run(mode="best") on the same domain runs only the winner
```
