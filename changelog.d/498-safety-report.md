Added
`safety_report` / `agentdescent safety`: a read-only pre-flight check for an
archive that a recursive loop is about to seed from. It combines exam coverage,
whether each slot's best eligible rule is trusted, and the worst oracle-vs-
verifier gain bias from an optional `MetaAuditStore` on the *seedable* set, returning `safe` /
`caution` / `unsafe` with the named checks and reasons. `--min-exam` mirrors
`verified_rules`; CLI exit codes are 0 safe, 1 caution, 3 unsafe, 2 usage. No
model calls, no writes.
