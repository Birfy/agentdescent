Added
`tools.module_audit`, a read-only reachability audit for the package: it
classifies every module as `cli/mcp`, `exported-only` or `isolated` (following
intra-package imports and `agentdescent.module:attribute` string refs), reports
module/LOC totals, the top-level `__all__` length and the CLI/MCP surface
counts, and emits JSON for scripting. It changes nothing by default;
`--fail-on-isolated` is an opt-in local check. This makes "what is actually on
the product path" measurable before any quarantine or merge work.
