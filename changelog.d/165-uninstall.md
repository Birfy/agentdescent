Added
`agentdescent uninstall <dsh|claude-code|codex|opencode>` (`--dry-run`, `--home`)
reverses `install` (#165): it removes the skill files and the plugin directory,
the marker-delimited `cordis.patch.yml` block, the `[mcp_servers.agentdescent]`
table and the pinned `mcp.agentdescent` entry, and nothing else. A Codex table
is removed even when Codex has appended other tables after it. Anything that is
no longer exactly what `install` writes (an edited launcher, a bare
`opencode mcp add` entry, a hand-written table) is kept and reported with exit
status 1, a commented `opencode.jsonc` is never rewritten, and the run store is
untouched. New `integrations.uninstall()`.
