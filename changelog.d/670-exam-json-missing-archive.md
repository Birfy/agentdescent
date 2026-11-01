Fixed
`agentdescent exam --json` on a missing archive now prints one
`{"ok": false, "error"}` object on stdout (exit 2) like every other read-only
verb, instead of a bare `error:` line on stderr; plain output is unchanged and
the archive is still not created (#670).
