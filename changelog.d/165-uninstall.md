Added
`agentdescent uninstall <dsh|claude-code|codex|opencode>` (`--dry-run`, `--home`)
reverses `install` (#165): it removes the skill files and the plugin directory,
the marker-delimited `cordis.patch.yml` block, the `[mcp_servers.agentdescent]`
table and the `mcp.agentdescent` entry, and nothing else. A config entry that is
no longer exactly what `install` writes (a launcher you edited, one you wrote by
hand) is kept and reported, a commented `opencode.jsonc` is never rewritten, and
the run store is untouched. New `integrations.uninstall()`.
