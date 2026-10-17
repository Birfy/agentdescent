Added
`agentdescent doctor` (CLI and the MCP `doctor` tool) now reports whether each
host is wired up and whether it is still current (#382). New read-only
`integrations.host_status(host)` compares the installed `SKILL.md` with the one
this package would write (`ok` / `stale` / `missing`) and checks the MCP entry,
following the same `DSH_HOME` / `CODEX_HOME` / `XDG_CONFIG_HOME` paths as
`install`. `doctor_report()` gains `hosts` and `hints`: a stale or half-written
integration is a problem (the skill a `pip install -U` left behind teaches the
host a procedure the package no longer has), while a host CLI on `PATH` with no
integration is only a hint. Nothing is written.
