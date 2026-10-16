Fixed
`PopulationAggregator` records `Candidate.parent` and backpropagates
subtree `selected` counts under `evolve()` (#275). Selecting a candidate
remembers its version as the branch tip; the next new state admitted from
that branch is filed as its child, and `selected` increments on the node
and every ancestor (cycle-guarded), so `FlatPuct` / `CostEfficient` /
`MCTS` see a real tree instead of a flat bag. Lineage survives
checkpoint/restore; pre-change checkpoints restore as roots. Archive ids
are unique even when two forks commit different states at the same
ledger version, so one fork's selection never bumps or parents the other;
restoring an older checkpoint that already holds such a collision gives
the later row a fresh id (saved parent references keep the first).
