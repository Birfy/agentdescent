Added
`agentdescent install grok-bot` / `uninstall grok-bot` scaffold a skill +
`HOST.md` + `mcp.example.json` under `~/.agentdescent/plugins/grok-bot/`
(#165). Grok Bot has no public one-click plugin API for local skill+stdio
MCP yet, so `doctor` reports `mcp n/a` when the scaffold is current rather
than pretending a host config was wired. Docs: `plugins.md`,
`plugin-quickstart.md`.
