Fixed
A foreground `agentdescent evolve` whose provider refuses the model or key
(`ProviderConfigError`, or any other exception from the run) no longer dies with
a Python traceback: it prints the status line, one `error: ...` line with the
hint, the run dir and `next: agentdescent diagnose <run_id>`, and exits 1. With
`--json` it prints one `{"ok": false, "run_id", "state", "error", "dir"}`
object. `status`/`diagnose` are unchanged; `AGENTDESCENT_DEBUG=1` keeps the
traceback (#652).
