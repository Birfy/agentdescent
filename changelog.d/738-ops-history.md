Added
`agentdescent ops publish --history FILE` appends one JSONL record per publish
(tenant, workspace, each run's metrics and alerts) with an fsync, and
`agentdescent ops history` aggregates those records across runs. Totals are
deduplicated by run id, so publishing the same database twice does not double
the counts; alerts are deduplicated by message. Malformed lines are skipped and
counted, tenant/workspace filters and a missing history are handled without
rewriting the file, and the last record is reported with an ISO-8601 timestamp.
