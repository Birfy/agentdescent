Added
Multi-artifact evolution — `extra_artifacts=` on `evolve()` and `async_evolve()`
(#197): register any number of artifacts beside the primary; the worker loop
proposes against each (PP stages name them; `PipelineParallel` lifts once
extras exist), the aggregator merges by `diff.target`, and a contract-breaking
diff lands atomically via `Ledger.commit_atomic` with exact dependent-adapter
coverage (`missing-adapters` when incomplete) and CAS retry that refuses a
stale adapter. `Contract.depends_on` drives dependent cache eviction
(`MemoryCache` / `FileCache.invalidate`). Sync/async rollout records attribute
extras to the extra's artifact id. Default path without extras is unchanged.
