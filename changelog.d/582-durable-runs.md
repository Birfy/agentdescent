Added
`DurableRunService` + `SqliteRunStore`: an event-sourced, reopenable run log
with leases/heartbeats, idempotent event keys, retries, exactly-once completion,
cancel, and expired-lease recovery for long RSI sessions.
