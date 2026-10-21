Added
The meta stack on the command line: `agentdescent portfolio`, `exam`, and
`audit`. `portfolio [archive]` reads the consolidation archive (default
`$AGENTDESCENT_HOME/archive`), builds one `LayerBudget` per decision slot via
`layers_from_archive`, runs `portfolio_report`, and prints each layer's
return-per-cost and reallocated budget with its reason (`--json` for
machine-readable output; empty archive exits 1 with a pointer to run a meta
session first). `exam [archive] --domain module:builder` resolves the domain
through a `module:attribute` ref (a zero-arg builder returning `oracle`,
`tasks_for`, `run`, `reward`, `propose`, `strategy`, and optionally `spec`),
runs `exam_archive`, and reports passed / flagged / demoted per rule. `audit
<run_id> --archive PATH [--meta-audit ...] [--inner-audit ...] [--harness
module:callable] [--bias-threshold N]` traces one archived adoption through its
evidence rings via `chain_audit` and prints each ring's verified / flagged /
unverified status; an unknown run or a bad ref fails cleanly (non-zero exit),
never with a traceback. One surface, two transports: the same verbs the MCP
server can expose.
