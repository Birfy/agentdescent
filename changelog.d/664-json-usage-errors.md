Fixed
`--json` now also covers argparse usage errors (missing required argument,
invalid choice, unknown option): instead of usage text on stderr, the CLI
prints one `{"ok": false, "error", "usage"}` object on stdout and still
exits 2; plain output is unchanged (#664).
