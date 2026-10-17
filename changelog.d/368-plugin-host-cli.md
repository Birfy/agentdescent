Fixed
`plan`, `evolve` and MCP `start` now check a host plugin's `host` CLI on
`PATH` separately from `agent` and `reflect`, so an OpenCode-only machine no
longer gets `ok: true` (or a launched, doomed run) for a Claude-format plugin
whose `claude` host is missing. `init` warns when the plugin's host CLI is
absent. The nested MCP start stub is unchanged (#368).
