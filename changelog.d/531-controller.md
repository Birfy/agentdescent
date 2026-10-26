Added
`run_controller` / `ControllerConfig` / `SessionOutcome`: a generic multi-session
loop with max sessions, call budget, kill switch, patience/stagnation and an
optional decision record per session. The caller supplies `run_session(index,
seed)`, so a guarded run, campaign step, population generation or remote job can
sit behind the same scheduling and stop policy.
