Added
`rsi_report` / `agentdescent rsi` / MCP `rsi`: a read-only longitudinal view of
the consolidation archive. Per decision slot it reports the first and best
adopted reward, the adoption that set the best, the gain, adoptions since that
best, and the exam counts, and returns an overall `improving` / `stagnant` /
`mixed` / `insufficient` verdict with the current portfolio reallocation.
