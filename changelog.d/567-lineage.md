Added
`LineageGraph` and `lineage_report` for artifact provenance: ancestry,
descendants, shortest derivation path, impact size, roots/leaves, topological
order, external-parent accounting, and cycle detection. Walks are iterative and
child adjacency is indexed, so long self-improvement lineages stay usable.
