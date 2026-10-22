Fixed
`agentdescent show` (CLI and MCP `show`) on a run without a result -- `failed`,
`cancelled` or still `running` -- no longer returns only `note: no result yet`.
It now also reports what is on disk: the round count and a short per-round
`history` (held-out reward, committed/rejected, `propose_error`), the run's
`best_reward` and `propose_errors`/`propose_error`, and the lowest-reward
recorded rollouts as `samples` (output vs gold, same 4 KB cap). The `note` is
kept and there is still no `diff`/`apply_plan`. A run with nothing recorded
is unchanged.
