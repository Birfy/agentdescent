Added
`agentdescent release verify` and the MCP `release_verify` tool: recompute the
HMAC over a signed receipt's embedded manifest, check that its decision is an
accepted `promote` with both hashes present, and optionally require the lineage
node to carry the same manifest hash. A well-formed but edited or wrongly keyed
receipt reports `valid: false` and exits 1, so CI can consume the exit code
without trusting the JSON file.
