Added
Persistent-island population engine: `evolve(population_islands=N)` (and
`async_evolve`) widens the population layer into a *persistent* search. With a
declared selection policy (e.g. `Policies(selection=CladeThompson())`), `N`
head-islands survive across steps and each records its own lineage
(`_branch_parent`), so tree-shaped policies score clades -- a lineage that
accumulates, which is what HGM's metaproductivity means. `PopulationAggregator`
creates/reuses the island forks instead of re-forking from `dev` every step,
checkpoints them (`forks`, `island_count`, `branch_parent`, the archive), and
reaps them at `finalize`. `island_count=0` keeps the old per-step-fork
behaviour exactly; a value above `0` without a declared selection policy is
refused ("honour or refuse"). `bench/population_engine.py` A/Bs single-head /
CladeThompson / islands+CladeThompson at equal budget and reports time-to-
quality, final reward and committed counts honestly (on a smooth monotone
landscape single-head is competitive; the population engine targets rugged
landscapes).
