Fixed
`agentdescent audit` on an unknown run id now prints an unquoted
`error: no archived run ... to audit` and exits 2, like the other unknown-run
errors, instead of a quoted `KeyError` repr with exit 1 (#481).
