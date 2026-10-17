Fixed
The `kind=skill_dir` / `agent_dir` "needs a WorkspaceAgent" error (and the
`tree_runner` one) now lists `opencode` alongside `claude_code`, `codex`, `dsh`
and `cli_agent`; the directory-kind and `kind=text` messages share one
`WORKSPACE_AGENT_REFS` phrase so they cannot drift apart (#362).
