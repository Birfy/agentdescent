Fixed
`agentdescent install opencode` no longer leaves OpenCode without its MCP
server when `~/.config/opencode/opencode.jsonc` has comments: the commented file
is still never rewritten, and the `mcp.agentdescent` entry goes into
`opencode.json` beside it, which OpenCode merges. If neither file can be parsed,
install prints a `FAILED:` line and exits 1, and `install` / `uninstall` now
honour `--json`. `doctor` no longer reports such a host as `ok` when the
commented file does not even name `agentdescent` (a commented-out mention no longer counts), and `uninstall opencode`
cleans up the `opencode.json` entry too.
