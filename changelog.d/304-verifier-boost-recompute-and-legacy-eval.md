Fixed
`VerifierBoost` no longer flips between active and inactive on repeated
`Audit.recompute()` calls with no new data, and a wrapped verifier that only
defines the legacy `oracle_eval` no longer raises `AttributeError` (#304).
Activation changes the audit's measurement version but not the scorer's:
`AuditedReward.scorer_version` (stamped on records as an optional
`AuditRecord.scorer_version`, omitted when equal to `verifier_version`) names
the scorer alone, the boost trains on every label of that scorer, and keeps a
validated curve when the unchanged scorer merely lacks labels — a different
scorer still withdraws it. `VerifierBoost.full_eval` / `oracle_eval` delegate
through `full_eval_of`, preserving the inner verifier's independence declaration.
