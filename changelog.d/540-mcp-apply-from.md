Added
The MCP `apply` tool now accepts optional `from` + `to` (mutually exclusive with
`run_id`, same rules as CLI `apply --from`: `from` requires `to`, symlinks
refused, dry-run / backup via `apply_from_payload`), so an MCP host can install
a campaign `final_target` without shelling out (#540).
