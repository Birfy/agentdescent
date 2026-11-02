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

The MCP `ops` tool takes the same arguments; `dry_run=true` is `plan`. MCP
`out`/`prometheus` writes are confined to `AGENTDESCENT_SURFACE_WRITE_DIRS`.
