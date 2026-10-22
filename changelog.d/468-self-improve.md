Added
Closed self-improvement loop (`agentdescent/self_improve.py`, with
`agentdescent/evidence.py` and `agentdescent/admit.py`). `evidence.snapshot`
reconciles the consolidation archive into one deterministic view -- per decision
slot's return-per-cost and exam stamps (from `portfolio.layers_from_archive`),
the harness leaderboard, and the archive's demoted/flagged counters -- and
`rank_slots` turns it into the next targets (falling back to round-robin on a
cold archive). `admit` composes an adoption's trust rings (exam / audit /
tournament / red-team / chain) into one fail-closed `pass` / `flag` / `demote`
verdict -- the generalisation of the joint rung's "isolating untrusted rules" to
every adoption path. `self_improve` runs the loop: read evidence, evolve the
paying slots with `meta_run` (adopt -> `archive_rule`), trust-gate each adoption (a
`demote`/`flag` is stamped on the record with `mark_exam`, quarantining it in the
archive's own paths), and move the next tick's budget with `portfolio_report` --
for a bounded number of ticks. Ticks compound: each target is seeded from the
archive's best rule for its slot, like `continuous_meta_run`. Targets are chosen
by evidence with an exploration share and a per-tick rotation, so an unseen slot
is tried instead of starved.

Based on #465.
