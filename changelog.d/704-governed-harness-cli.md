Added
`agentdescent governed-harness` and the MCP `governed_harness` tool: one
spec-driven surface for the existing safety-gate -> proof -> approval ->
canary/rollback -> promotion chain. `init --demo` is runnable offline; `plan`
validates refs and scores without running the gates; `run` returns the full
`GovernedHarnessReport`. A failed proof or pending/rejected approval never
reaches promotion; a conformal/sequential reject is never waived; MCP refs and
writes use the operator's surface allowlists. JSON results carry
`schema_version: 1`. Direct `govern_harness_change` callers also get the
matching fix: optional conformal `insufficient` / sequential `continue`
evidence is ignored when the rules do not require those tests, while a
conformal/sequential reject still fails the promotion.
