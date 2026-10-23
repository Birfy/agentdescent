# Campaign — keep evolving, carrying the artifact forward

One `evolve()` run is a session. A campaign runs several sessions on the same
spec, each starting from the artifact the previous session produced, and stops
when a session fails to improve. That is the smallest honest "keep improving"
loop for a spec: there is no second engine and no hidden state — the next
session's target is the previous session's output.

```bash
agentdescent campaign spec.json --sessions 3 --rounds 2 --algorithm rrsi
```

The campaign works on a **copy** of the target in a temporary workspace, so the
user's file/directory is never touched. The report returns the workspace and
`final_target`; apply that deliberately with `agentdescent apply --to ...` or by
copying it yourself.

- An **improving** session's artifact is written back to the workspace copy.
- A **non-improving** session is discarded and, after `--patience` such sessions
  (default 1), the campaign stops; the workspace keeps the best artifact, not
  the last one.
- Each session's seed is the spec's seed plus the session index, so a campaign
  does not re-run the same search three times.
- `--dry-run` composes one session and prints `calls/session x sessions`.
- `policy_slot` is refused: its target is a slot name, not a file, so there is
  nothing to carry forward.

```python
from agentdescent import run_campaign
from agentdescent.evolvespec import load_spec

report = run_campaign(load_spec("spec.json"), sessions=3, rounds=2)
print(report.best_reward, report.final_target, report.stopped_reason)
```

A campaign is a **loop over ordinary runs**, not a claim that more sessions are
always better. It reports `stopped_reason="patience"` when the measured reward
stops moving; if that happens on session one, the second session bought nothing
and the report says so.
