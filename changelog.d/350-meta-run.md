Added
Meta-evolution product surface: `meta_run()` evolves one decision slot of the
inner search (`meta_evolve` on the train problems), validates the evolved value
on problems the outer loop never saw (`meta_validate`, with `val_seeds`
disjoint from the training seeds), and returns a `MetaRunReport` — the rendered
value (`report.value`; compile with `spec.compile(...)`), the per-problem
before/after/gain, the transfer ratio (mean val gain / mean train gain), and an
`adopt` / `reject` verdict with its reason. A value is adopted only when its
mean gain on unseen problems clears `min_gain` and the transfer is not below
`transfer_floor`; otherwise it is rejected and the reason says whether that is
"no gain" or "a fit that does not transfer". The report rides the run result
(`EvolutionResult.meta`), so a CLI or plugin shows the decision without
recomputing it. Governance: the caller must keep validation problems and seeds
disjoint from the training set (`meta_run` raises on overlap), so a verdict is
never measured on the problems that shaped the value — the same final-evaluation
isolation every other development set gets. Offline demo:
`python -m examples.meta.meta_evolve_demo`.
