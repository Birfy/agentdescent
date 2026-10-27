Added
`run_campaign` / `agentdescent campaign`: run several evolution sessions on one
spec, each starting from the previous session's artifact. The campaign works on
a copy in a temporary workspace, applies only improving sessions, stops after
`patience` non-improving rounds, and reports the best artifact, workspace and
`stopped_reason`. Session 0 is the baseline (not a claimed improvement), the
split seed is fixed across sessions so rewards are comparable, and `final_reward`
is the last session while `best_reward` is the artifact kept. `--dry-run` prices
the sessions first; `policy_slot` and inline `text` targets are refused because
there is no file to carry, and `--workdir` must be empty.
