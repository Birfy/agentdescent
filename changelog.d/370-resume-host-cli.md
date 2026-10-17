Fixed
`resume` (CLI and MCP) now re-checks a stopped `kind="plugin"` run's stored
`host` CLI on `PATH` before relaunching it (#370). A run started where `claude`
was installed and resumed where it is not used to reach `runstore.launch`,
write a fresh `running` status and fail only inside the detached child; it now
refuses with an error naming the `host` role and the missing binary, leaving
the stopped status, PID, spec and ledger untouched. Only the host is a hard
check on resume: `agent` / `reflect` keep their soft warnings, an
already-running run still gets its own rejection, and the host is never
switched to whichever CLI happens to be installed.
