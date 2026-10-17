Added
Cross-run consolidation (`agentdescent/consolidation.py`): what a run learned
survives it. `ConsolidationStore` archives a run's product — the artifact
state, rendered skill, measured reward, rounds and configuration — one JSON
record per run under `<path>/runs/<run_id>.json`, atomic and idempotent, with
a durable trajectory-store path linked rather than copied. `seed_from_archive`
retrieves a candidate for the next run (same configuration fingerprint first,
then measured reward, same strategy only) and hands its state to
`evolve(initial_state=...)` **only if a mini-eval on a sample of the new tasks
shows it does not regress beyond `min_reward` against the empty baseline** —
a harmful cross-domain seed is refused and reported, never installed.
`ConsolidationRecord` / `SeedReport` carry what was decided and why;
non-finite rewards are refused. Offline demo:
`python -m examples.consolidation.consolidation_demo` (seeded run reaches the
quality bar ~200x sooner than the empty seed on the measured domain).
