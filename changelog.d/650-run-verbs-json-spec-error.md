Fixed
`agentdescent evolve`, `campaign` and `tournament` with `--json` now answer a
missing, unreadable, malformed or non-compiling spec with the same
`{"ok": false, "error": ...}` object on stdout that `plan --json` prints,
instead of a bare stderr line; without `--json` they print `spec error: ...`
on stderr as before (`evolve` on an unreadable file now uses that same
prefix) (#650).
