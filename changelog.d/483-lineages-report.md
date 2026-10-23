Added
A persistent-island population run now returns `EvolutionResult.lineages`
(#150, semantics C): a read-only row per lineage (`dev` plus each island) with
its candidate count, best archived version/score, selection count and lineage
tip. It is read from the archive the gate already scored, survives `finalize`,
and is `None` on a single-head run; the official artifact is unchanged.
