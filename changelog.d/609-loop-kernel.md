Added
`LoopKernel`: a resumable RSI loop over `durable_runs` that composes the
existing seams in one order — `step -> RSIController -> apply/rollback` — and
replays completed generations after a crash. Each generation carries its
observation, control action, gain, cost and optional decision-log/telemetry
ids; the final generation is durable instead of being lost in the completion
payload.
