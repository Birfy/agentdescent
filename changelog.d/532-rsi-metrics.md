Added
`metrics_from_records` / `metrics_from_log`, `to_prometheus` and `alerts`: RSI
aggregates from decision records -- acceptance rate, calls/tokens/wall clock,
calls per accepted decision and category frequency -- plus a deterministic
Prometheus text export and pure threshold alerts. The export gives the
per-category counter its own family, escapes label values and rejects malformed
records instead of silently coercing them. No model calls and no second source
of truth.
