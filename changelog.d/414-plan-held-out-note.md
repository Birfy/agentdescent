Fixed
`plan` / MCP `plan` now notes the condition `evolve()` warns about — a held-out
set under 4 tasks — from the same computation (`held_out_size`, `MIN_HELD_OUT`
in `agentdescent.evolution`) instead of a separate `4 <= tasks < 8` rule. 8–11
tasks at the default `held_out_frac=0.3` hold out 2–3 and used to plan clean;
the note names the held-out count, both fixes (more tasks, or a higher
`evolve.held_out_frac`) and the smallest task count that works (12 at 0.3). A
detached / MCP `start` run records the note in `status`'s `notes` from the first
poll (and `start` / `evolve --detach` return it), `agentdescent status` prints
it, and a foreground `agentdescent evolve` shows one plain `warning:` line
instead of a source-line `RuntimeWarning`. The SKILL and quickstart now say
"12 to 20 cases" rather than 8 (#414).
