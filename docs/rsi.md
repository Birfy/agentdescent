# Longitudinal RSI report — is the archive still getting better?

Every other read model answers a question about one run or one archive window.
A recursive self-improvement system needs the cross-session question: across the
rules it has adopted, is capability still rising, or has the search plateaued?

`rsi_report(store)` reads the consolidation archive's rule records, groups them by
decision slot, and orders each slot's adoptions by `created_at`. It reports the
first reward, the best reward and the adoption that set it, the gain, and how
many adoptions have happened since the current best. No model calls, no writes.

```python
from agentdescent import rsi_report

report = rsi_report(store)                 # or rsi_report("/path/to/archive")
print(report.verdict, report.reason)       # improving / stagnant / mixed / insufficient
for slot, row in report.slots.items():
    print(slot, row.best_reward, row.gain, row.records_since_best)
```

## Verdicts

- **`improving`** — at least one slot's best adoption is above its first by more
  than `tolerance` (default `0.02`).
- **`stagnant`** — every slot has gone `stagnation_window` adoptions (default
  `3`) without a new best.
- **`mixed`** — some slots improved, others are flat or stagnant.
- **`insufficient`** — fewer than two adopted rules; run more sessions.

The per-slot verdict also distinguishes `flat` (no gain yet, but not stale),
`regressing` (latest below first), and carries the exam counts (`trusted` =
passed, `unexamined`, `flagged`, `demoted`). Demoted and flagged rules are
excluded from the rewarded trend, so a demoted best cannot make a slot look
`improving`. `best_index` / `best_created_at` name the adoption that set the
best.

## CLI and MCP

```bash
agentdescent rsi [--archive PATH] [--slot NAME] [--stagnation-window N] [--json]
```

MCP exposes the same data as the `rsi` tool. The archive is the consolidation
archive (`$AGENTDESCENT_HOME/archive` by default), not a run store.

## What it does not claim

The report measures *adopted* rules on the reward their sessions recorded; it
does not re-run the agent, and it does not compare rules from different slots
against each other. That is deliberate: a trend that costs nothing to read is
useful on every session, and a cross-slot score would need a common task set
that the archive does not carry. Use this report for the direction of travel; a frozen protected task contract
belongs at the acceptance seam, not in the archive read model.
