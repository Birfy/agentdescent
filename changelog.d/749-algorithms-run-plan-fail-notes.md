Fixed
`agentdescent algorithms run` no longer hides init's notes and warnings when
the follow-up `plan` fails: text mode prints the `note:` / `warning:` lines
(including where an overwritten spec was backed up as `<out>.bak-N`), and the
`--json` / MCP `algorithm_spec` error object now carries `notes` and
`warnings` lists like the success object (#749).
