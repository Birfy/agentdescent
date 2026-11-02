Fixed
`agentdescent exam --domain module:builder` now turns a builder that raises
(e.g. `TypeError` from `--domain json:dumps`) or returns a dict missing
required keys into one `error:` line (exit 2), or one `{"ok": false, "error"}`
object under `--json`, instead of a raw traceback (#672).
