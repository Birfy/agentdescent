Added
`curriculum_meta_run` gains opt-in verified seeding: `min_exam` (via
`verified_rules`) excludes unexamined, flagged or audit-biased rules when set,
and `oracle` + `audit_store` audit a sampled slice of the
validation pairs and persist the verdict with `audited_n`, `oracle_gain` and
`gain_bias` in the report. The generated-problem map is pruned to the live
frontier pool, and default run_ids get a random suffix so same-second
adoptions no longer collide (#679).
