Fixed
`--budget` / `budget_usd` without `--usd-per-call` / `usd_per_call` is rejected
before a run directory is created, so detach and MCP `start` no longer report
ok/started and then fail only in `log.txt`; `status` also surfaces the real
`error:` line from the log when a child dies without writing a terminal state.
The plugin quickstart now shows both flags (#342).
