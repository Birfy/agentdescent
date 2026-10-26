# Session controller — when is another self-improvement session justified?

`run_guarded` runs one session. A product needs the loop around it: session
budget, call budget, kill switch, stagnation stop and a decision record per
session. `run_controller` is that loop, and it deliberately does not know what a
session is.

```python
from agentdescent import ControllerConfig, run_controller

report = run_controller(
    run_session=lambda index, seed: my_session(index, seed),  # -> SessionOutcome
    config=ControllerConfig(max_sessions=5, max_calls=500, patience=2,
                            kill_switch=switch, decision_log=log))
print(report.best_reward, report.stopped_reason, report.total_calls)
```

- `run_session(index, seed)` returns `SessionOutcome(reward, calls, error,
  evidence)`. It can wrap a guarded run, a campaign step, a population
  generation or a remote job.
- The first successful session establishes the baseline; later sessions must
  beat the best to reset stagnation. `patience` non-improving sessions stop the
  loop. `best_reward` is `None` when no session succeeded.
- An errored or raised session is recorded, stops the loop, and never updates
  the best -- even if its partial reward is high. `error=""` counts as an error.
- A kill switch is checked before every session (an engagement during a session
  is seen at the next check); a call budget is checked before and after a
  session, so an overshoot stops before another one starts.
- `calls` must be a non-negative integer and `reward` finite; a `run_session`
  that returns anything other than `SessionOutcome` becomes an error record.
- `kill_switch` is any object with `engaged()` and `reason()`; `decision_log`
  any object with `record(kind=..., artifact_id=..., accepted=..., category=...,
  detail=..., evidence=..., cost=...)`. With a log, each session becomes a
  `kind="session"` record with reward, improved flag, error and calls.
- `stopped_reason` is a fixed code (`sessions`/`budget`/`kill-switch`/`error`/
  `patience`); `reason` is the human-readable detail.

This is the scheduling boundary: engine/artifact application stays with the
caller, while "another session or not" has one tested answer.
