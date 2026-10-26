Added
`evaluate` / `EvaluationReport`: a deterministic, budgeted verifier-vs-oracle
pass. The verifier scores every task, the oracle only a seeded sample capped by
`budget`, and the report carries means, signed/absolute bias, coverage, oracle
calls and per-task rows. Zero coverage is explicitly "no truth evidence", not a
pass.
