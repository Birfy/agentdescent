Added
`RiskPolicy` / `RiskRule` / `RiskContext`: ordered, declarative rules that map a
change's tier, blast radius, diff size, capability domain and tenant to
`allow`/`approve`/`canary`/`rollback`/`deny`. First match wins, no match is
fail-closed (`deny` by default), and invalid actions or duplicate rule names are
refused at construction.
