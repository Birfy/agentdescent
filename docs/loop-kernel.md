# The RSI loop kernel — one durable loop over the existing seams

`run_guarded` runs one session; `RSIController` turns one observation into
accept / reject / defer / rollback / stop; `durable_runs` keeps a leased event
log. `LoopKernel` is the small state machine that puts them in one order and
survives a crash.

```python
from agentdescent import LoopConfig, LoopKernel, LoopObservation
from agentdescent.durable_runs import SqliteRunStore

store = SqliteRunStore("runs.sqlite3")
kernel = LoopKernel(
    store,
    LoopConfig("tenant-a", "ws-1", max_generations=4, target=0.9),
)
run_id = kernel.submit()

def step(ctx):
    # generate + evaluate one candidate; PRM / planner / verifier all fit here
    return LoopObservation(reward=0.7, cost=1.0, artifact_id="skill-1")

# apply is called once per accepted generation; make it idempotent for
# (run_id, generation), exactly like step.
def apply_artifact(context, observation):
    materialise(observation.artifact_id)

report = kernel.run(run_id, "worker-1", step, apply=apply_artifact)
report.best_reward, report.stopped_reason, report.generations
```

## What one generation is

`step(context) -> LoopObservation` is the caller's session: generate a
candidate, score it, and return the reward, cost, risk, artifact id and any
evidence. The kernel never runs a strategy itself.

`LoopContext` carries the generation, deterministic seed, previous
`budget_scale`, previous best and the reconstructed history. A caller that
wants PRM ranking, MCTS or an ensemble chooses it inside `step` (or delegates
the choice to `ComputeController`).

## What the kernel owns

- **Order.** `step -> RSIController -> apply/rollback -> record`.
- **Durability.** Every completed generation is a `step` event carrying the
  observation *and the decision that was actually taken*. A restart replays
  those decisions through `RSIController.restore()`; it never re-calls a gate.
- **Completion.** The final generation is a `step` event too, so the run's last
  observation is not hidden inside the `completed` payload.
- **Audit.** Optional `DecisionLog` records `kind="loop"` rows with a
  deterministic per-generation id; optional `RSITelemetry` emits one span per
  generation.
- **Control.** Target, cost budget, patience, risk limit, gates and an optional
  `cancel()`. A missing `LoopConfig` value or a step returning the wrong type is
  a recorded failure, never a silent zero.

## At-least-once, not exactly-once

`durable_runs` is at-least-once. `step` and `apply` must therefore be
idempotent for `(run_id, generation)`: a crash after the external effect but
before the event replays that generation. The kernel records a generation only
after both return. The event key is `generation:{n}`, and the decision log's
record id is `{run_id}:generation:{n}`, so the replay is a no-op for the log.

If retryable failures exhaust the step budget, `run()` can return with
`status == "running"`; call `run()` again until `report.terminal`.

## Reporting and recovery

```python
report = kernel.report(run_id)          # replay + reconstruct, no gates
snapshot = kernel.snapshot(run_id)      # durable_runs snapshot
kernel.cancel(run_id, reason="operator stop")
```

`LoopGeneration` exposes the action, reason, reward, gain, cost, artifact,
stop/applied flags and span id for each generation. `LoopReport.to_dict()` is
JSON-serialisable for a console or a daily report.

The kernel is not a distributed scheduler: one SQLite lease, one worker at a
time. The event schema is what a Postgres/Temporal adapter keeps.
