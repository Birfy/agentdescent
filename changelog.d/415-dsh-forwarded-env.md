Fixed
`agentdescent install dsh` and the native `dsh-agentdescent` plugin now forward
the same environment list as Codex (`FORWARDED_ENV`: `DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
`ARK_API_KEY`, `AGENTDESCENT_HOME`) instead of four keys, so `ARK_API_KEY`, which
dsh scrubs unless it is declared, reaches the MCP server. There is one tuple
for every host (`DSH_FORWARDED_KEYS` is gone). Re-running `install dsh` rewrites
an older four-key block, `doctor` / `host_status("dsh")` report it as `stale`,
and `install dsh` prints the new list. `docs/plugins.md` lists the names and
notes that a custom `api_key_env` must be added by hand (#415).
