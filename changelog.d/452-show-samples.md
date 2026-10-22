Added
`show` (CLI and MCP) now explains a low reward: a new `samples` key lists up to
five of the lowest-reward rollouts of the final artifact, each with task id,
the agent's output, the task's gold (both cut to 300 characters) and reward,
under a fixed 4 KB cap; text mode prints a short block. The run store keeps the
engine's rollout records under `ledger/trajectories/` in the run directory. A
new spec field `record_rollouts` (default `true`) turns this off. First part of
#452; rejected-candidate diffs follow separately.
