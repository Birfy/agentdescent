Fixed
`agentdescent doctor` no longer labels a host CLI it found on `PATH` as
`missing`: an installed but unwired CLI now reads `not installed (cli ...)`
with the `agentdescent install` command, a host without a CLI reads `no cli
on PATH`, and the closing line says `ok: no blocking problems`, naming how
many hosts can be wired. The `--json` report is unchanged (#535).
