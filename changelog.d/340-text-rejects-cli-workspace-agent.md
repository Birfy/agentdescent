Fixed
`compose` / `plan` refuse `kind: "text"` when `agent` resolves to a
`WorkspaceAgent` (`claude_code` / `codex` / `dsh` / `opencode` / `cli_agent`),
with a `SpecError` naming `agent` and pointing at `openai_compatible` /
`host_model` / `claude` instead of returning `ok: true` (#340).
