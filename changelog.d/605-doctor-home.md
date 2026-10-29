Fixed
`agentdescent doctor` accepts `--home <dir>` with the same probe semantics as
`install` / `uninstall` (#605): host skill/MCP status is read under that tree,
and an explicit `--home` wins over `CODEX_HOME` / `DSH_HOME` /
`XDG_CONFIG_HOME`. The MCP `doctor` tool takes the same optional `home`
argument.
