Added
`diagnose_payload` / `agentdescent diagnose` / MCP `diagnose`: turns `show`'s
samples, rejected candidates and outcome histogram into one actionable verdict
(`solved`, `partial-improvement`, `running`, `cancelled`, `run-failed`,
`reflector-error`, `scoring-mismatch`, `gate-rejected`, `agent-failures`,
`no-improvement`, `unknown`) with bounded evidence and the next step. A
committed-but-partial run is no longer called "no candidate adopted", gate
rejections win over the worst-sample rule, truncated samples do not claim a
scoring mismatch, and running/cancelled states are named. It reuses the same
read model as the CLI/MCP without building a discarded diff.
