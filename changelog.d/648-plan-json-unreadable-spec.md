Fixed
`agentdescent plan --json` on a missing, unreadable or malformed spec file now
prints the same `{"ok": false, "error": ...}` object on stdout as a spec that
fails to compile, instead of a bare `error:` line on stderr; without `--json`
the stderr line is unchanged (#648).
