Fixed
`agentdescent plan` without `--json` no longer prints a `{"ok": false, ...}`
JSON object when the spec fails to compile (bad `--algorithm` / `--param`, bad
data rows): it prints `error: <message>` on stderr and exits 2, like an
unreadable spec file. `--json` output is unchanged (#551).
