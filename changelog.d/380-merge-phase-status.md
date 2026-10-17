Fixed
`status` (CLI and MCP) now reports `pending=1 phase=merge` while the merge step
runs (#380). Conflict resolution, the fusion tournament and the held-out gate
were invisible to `pending_calls` / `phase` (#358 only covered the `propose` and
`run` actors), so a reflective merge blocked on one model call for minutes read
as `calls` standing still with nothing in flight, and a host polling `status`
could not tell it from a wedged run. `Meter.timed("merge_gate_seconds")` now marks
its block in flight on both `evolve()` and `async_evolve()`. The shipped
`SKILL.md` tells the host what `phase` means, to judge liveness from `status`
rather than `ps`, and to keep shell waits under a minute.
