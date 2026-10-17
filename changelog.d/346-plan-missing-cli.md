Fixed
`plan` / MCP `start` / CLI `evolve` fail closed when a named CLI agent
(`claude_code` / `codex` / `dsh` / `opencode`) is missing from `PATH`, instead of
returning `ok: true` and failing later with `AgentError`; the error names the
binary and how to install or switch agents. Isolation/login notes stay soft
warnings when the binary is present (#346).
