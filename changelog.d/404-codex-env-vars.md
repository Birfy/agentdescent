Fixed
Codex scrubs the environment of an MCP server, so the AgentDescent server it
started saw no provider key and no `AGENTDESCENT_HOME`. `install codex` now
writes an `env_vars` line (the provider keys, `ARK_API_KEY` and
`AGENTDESCENT_HOME`) into `[mcp_servers.agentdescent]`, and the plugin's
`.mcp.json` carries the same key for the `codex plugin add` route. Re-running
`install codex` repairs a block written before this, `uninstall codex` still
removes both forms, and `doctor` reads the old block as `stale` (#404).
