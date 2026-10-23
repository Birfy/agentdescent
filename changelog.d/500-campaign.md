Added
`run_campaign` / `agentdescent campaign`: run several evolution sessions on one
spec, each starting from the previous session's artifact. The campaign works on
a copy in a temporary workspace, applies only improving sessions, stops after
`patience` non-improving rounds, and reports the best artifact, workspace and
`stopped_reason`. `--dry-run` prices the sessions first; `policy_slot` is
refused because it has no file to carry.
