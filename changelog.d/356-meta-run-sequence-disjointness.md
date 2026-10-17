Fixed
`meta_run` no longer rejects distinct sequence-based train and validation
problems (#356). The disjointness gate compared synthetic `p0`/`p1` labels
assigned independently by `_named_problems`, so every pair of nonempty
`Sequence[Problem]` inputs collided before `meta_evolve`. Isolation now
checks explicit mapping keys when both sides are mappings, and problem
object identity for genuine shared callables (including the sequence API);
seed-overlap rejection is unchanged.
