Added
`AdaptiveExperience`: one cross-cycle composition that ingests counterfactual
credit through `CounterfactualExperience`, selects validation anchors from a
fitted `PreferenceReplay` (or feature recall, recent history, or self-validation
on the first cycle), and delegates the guarded update to `ExperienceEngine`.
Ingestion deduplicates by `run_id`, and missing validation or bridge features
fail closed (#683).
