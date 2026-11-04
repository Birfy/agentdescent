# Ops — publish a campaign to the control room

`agentdescent ops` reads terminal runs from a SQLite run database and writes
them into the console, lineage, telemetry, RSI metrics and alerts:

```bash
agentdescent ops plan    --runs-db platform.sqlite3 --tenant t1 --workspace ws
agentdescent ops publish --runs-db platform.sqlite3 --tenant t1 --workspace ws \
                         --out ops.json --prometheus metrics.txt
```

`plan` only counts the terminal runs and generations it would publish.
`publish` records each run through `OpsFabric` into a fresh in-memory read
model, derives one decision row per generation, and returns the consolidated
`CampaignOpsReport`, metrics and alerts.

- `--min-acceptance` alerts when the campaign's acceptance rate is below it.
- `--max-cost-per-accepted` alerts when calls per accepted decision exceed it.
- A tenant/workspace with no terminal runs is a no-op with
  `published: 0`, not an error.
- A missing database or an unknown tenant/workspace is an input error (exit 2).

## History

Each `publish` builds a fresh in-memory read model, so a single report cannot
answer "how many decisions has this campaign accumulated?". Pass `--history` to
append one record per publish and `history` to summarise them:

```bash
agentdescent ops publish ... --history ops.jsonl
agentdescent ops history --history ops.jsonl [--tenant T] [--workspace W] [--json]
```

`publish` appends and fsyncs one JSON line after the read model is built and
before the `--out`/`--prometheus` files are written, so a failed report write
still leaves the publish recorded; it never rewrites the file, and a no-op
publish (no terminal runs) does not add a line. Each record carries the metrics
for every run it published, not just the campaign total.

`history` reads the JSONL, skips malformed lines and records without per-run
metrics (counting them in `skipped`), filters by tenant/workspace, and
deduplicates **by run id**: the latest snapshot of a run wins, so publishing the
same database twice does not double the totals. Alerts are deduplicated by
message. It aggregates `publishes` (publish calls), `published`/`runs` (unique
runs), `decisions`, `accepted`, `rejected`, `calls`, `prompt_tokens`,
`completion_tokens`, `wallclock`, `alerts` and `categories`, returns the last
record plus `last_recorded_at` as ISO-8601 UTC, and recomputes
`acceptance_rate` / `cost_per_accepted` from the totals. A missing or empty
history is `publishes: 0`, not an error.

The MCP `ops` tool takes the same arguments; `dry_run=true` is `plan`. MCP
`out`/`prometheus` writes are confined to `AGENTDESCENT_SURFACE_WRITE_DIRS`.
It deliberately has no history path: publishing from a host model should not
append to an operator's log without an explicit CLI step.
