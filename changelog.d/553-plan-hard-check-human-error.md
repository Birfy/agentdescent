Fixed
`agentdescent plan` without `--json` no longer dumps the `{"ok": false, ...}`
JSON object when the spec compiles but fails a hard check (too few tasks,
missing CLI binary, unfilled placeholder): it prints `error: <message>` on
stderr and exits 2, like a spec that fails to compile. `--json` output is
unchanged (#553).
