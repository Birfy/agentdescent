Fixed
`--json` now works after the subcommand as well as before it, for every verb:
`agentdescent show --json <id>`, `doctor --json`, `status --json`,
`plan --json <spec>`, `portfolio --json` and `audit <id> --json` used to exit 2
with "unrecognized arguments", even though the docs write them that way.
