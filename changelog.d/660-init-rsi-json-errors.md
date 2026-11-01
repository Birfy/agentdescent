Fixed
Under `--json`, `init` ValueError / agent-mismatch refusals and `rsi`
archive or parameter errors print `{"ok": false, "error": ...}` on stdout
instead of a bare `error:` line on stderr; exit codes and plain output are
unchanged (#660).
