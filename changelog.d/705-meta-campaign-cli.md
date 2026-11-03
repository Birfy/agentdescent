Added
`agentdescent meta-campaign` and the MCP `meta_campaign` tool: repeated
`continuous_meta_run` sessions over one `ConsolidationStore`, each in a fresh
ledger repo, with `RSIController` target/cost/patience/session stops. `init
--demo` is runnable offline from `agentdescent.meta_demo`; `plan` resolves refs
and configs without calling builders or creating the archive; `run` returns the
per-session verdicts, best session and archive count. A rejected rule is a
normal measured session; an errored session stops the campaign with
`complete=false`. MCP refs and writes use the operator's surface allowlists.
JSON results carry `schema_version: 1`.
