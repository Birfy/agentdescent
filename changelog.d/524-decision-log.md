Added
`DecisionRecord` + `DecisionLog`: an append-only JSONL of immutable adoption and
session decisions with the guard category, evidence, cost and parent link.
Malformed lines (including non-object JSON) are skipped, a torn last line is
closed before the next append, duplicate record ids are refused, non-finite
floats become null, and `children()` exposes the decision tree a dashboard or
audit export needs. Writers in one process share a per-path lock.
