Fixed
Under `--json`, run-store errors (`status` / `show` / `apply` / `cancel` /
`resume` / `diagnose` on an unknown run) and archive errors (`portfolio` /
`audit` / `safety`) print `{"ok": false, "error": ...}` on stdout instead of a
bare `error:` line on stderr; exit codes and plain output are unchanged (#658).
