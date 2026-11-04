Added
`agentdescent platform status` and the MCP `platform_status` tool: inspect a
campaign's SQLite run database after the process that wrote it has exited. With
no `--run-id`, list the most recent runs (newest first) with status, best
reward, generation count, accepted count and stop reason; with `--run-id`,
return that run's full loop report. The command only reads the database and
needs no compute, foundry or experience callbacks; a missing database or
unknown run is an input error.
