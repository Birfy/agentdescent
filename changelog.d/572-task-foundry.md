Added
`TaskFoundry`, a failure-seeded task assembly line: generate, validate,
deduplicate (exact + optional near-duplicate), filter contamination and
unlearnable tasks, and split accepted tasks deterministically into train and
holdout. `failure_seeds` mines low-reward trajectory rows as seeds.
