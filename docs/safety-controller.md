# Deployment safety — kill switch, canary, rollback

An acceptance gate decides whether to commit; a deployment controller decides
whether to *keep* a change live. These are separate operations and a product
needs both.

```python
from agentdescent import KillSwitch, canary, rollback_plan

switch = KillSwitch("~/.agentdescent/STOP")
if switch.engaged():
    raise SystemExit(f"operator stop: {switch.reason()}")

report = canary(incumbent_by_task, candidate_by_task, min_effect=0.01)
if report.verdict == "promote":
    ...
elif report.verdict == "rollback":
    plan = rollback_plan(current_state, previous_state)
```

## Kill switch

A file whose existence is the stop signal; its content is the operator's reason.
`engaged()` is safe to call before every session, `engage(reason)` and
`release()` are operator actions. `engage` writes a temp file and atomically
replaces the target, and a directory at the path still counts as engaged.
No code change or restart is needed.

This module only provides the switch; it does not wire itself into
`self_improve`, the engine or the CLI yet. A caller/controller must check
`engaged()` before starting a session.

## Canary

`canary(incumbent, candidate)` compares per-task scores with a **paired exact
sign test** (ties are not wins). It promotes only when every incumbent task has
a candidate score, the mean gain clears `min_effect`, and the one-sided p-value
is below `alpha`. It rolls back when the mean loss clears `max_regression`
(default 0, so any significant loss) and the reverse sign test is below `alpha`;
otherwise it holds for more evidence. A task the candidate did not score is
listed in `missing` and blocks promotion. Missing scores and non-finite values
are surfaced rather than treated as zero.

`min_effect` is compared with the **mean**, while significance comes from the
**signs**, so one large outlier cannot promote on its own but can block a
rollback.

## Rollback plan

`rollback_plan(current_state, previous_state)` is a pure diff between two
artifact states: `write` restores changed files, `delete` lists paths only in the
current state, and `noop` is true only when both are empty. It does not touch
disk -- the caller applies it with the same materialiser used to write the
artifact, so a rollback goes through the same path as a normal apply.
