Fixed
`agent_society._is_literal` now accepts unary `+` / `-` on numeric constants
(`-1`, `+2`, nested in dict/list), matching `ast.literal_eval`, so the society
gate no longer reports a signed number as "computed code" (#593).
