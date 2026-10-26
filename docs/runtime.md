# Guarded runtime — compose, inject acceptance, run, record

The pieces exist separately: `compose(spec)` builds the engine call, an
acceptance policy (typically a guard pipeline) composes the checks, a decision
log records the decision and a kill switch is the operator stop.
`run_guarded` wires them into one call so a product does not reimplement the
order.

```python
from agentdescent import run_guarded

guarded = run_guarded(
    spec,
    acceptance=pipeline,          # any AcceptancePolicy, e.g. a GuardPipeline
    decision_log=log,             # any object with .record(**kwargs)
    kill_switch=switch,           # any object with .engaged()/.reason()
    repo_path="runs/current/ledger")
print(guarded.result.final_reward, guarded.decision.category)
print(guarded.notes)
```

- The kill switch is checked **before** composing. This is a pre-flight check:
  an engaged switch raises rather than silently starting a session, but a
  switch engaged mid-run is not polled by this wrapper.
- The supplied acceptance is injected into the composed `Policies`. When the
  spec has no acceptance policy, the guard is chained **before the shipped
  `DefaultAcceptance`** rather than replacing it. When the spec has its own
  policy, that policy is the second link. Guards can only make acceptance
  stricter, never remove the statistical gate.
- `bind` / `configure` and the stateful `checkpoint_state` /
  `restore_state` / `on_screen_reject` hooks are forwarded through the chain.
- A `decision_log` receives one `kind="run"` record with final reward, outcomes,
  rounds, error, acceptance names, calls/tokens/wallclock. `accepted` means a
  candidate was actually committed; a run that made no commits is not recorded
  as an acceptance. A failing log does not hide the completed run: the note
  says the log failed and `GuardedRun.result` is still returned.
- An algorithm `aggregator_factory` owns its merge path; the injected
  acceptance is then not consulted, and `GuardedRun.notes` says so.

The runtime is deliberately duck-typed: it does not import a specific guard,
log or switch, so those can evolve independently and a third-party
implementation is first-class.
