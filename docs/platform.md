# The RSI platform cycle — five layers, one hand-off

The loop kernel, compute allocator, promotion decision, capability foundry and
continual experience are separately tested. `RSIPlatform` is the narrow system
seam that makes them one cycle:

```text
compute -> loop -> promote -> admit capability -> learn experience
```

```python
from agentdescent import PlatformConfig, RSIPlatform
from agentdescent.durable_runs import SqliteRunStore

platform = RSIPlatform(
    SqliteRunStore("runs.sqlite3"),
    PlatformConfig("tenant-a", "ws-1", max_generations=3, target=0.9),
    compute=controller,          # ComputeController
    foundry=foundry,             # CapabilityFoundry (optional)
    experience=engine,           # ExperienceEngine (optional)
    decision_log=log, telemetry=telemetry)

run_id = platform.submit()
report = platform.run_cycle(
    run_id, "worker-1",
    step=my_step,                 # (LoopContext, ComputeDecision) -> LoopObservation
    evidence=my_evidence,         # (LoopReport, best) -> PromotionRequest
    capability=my_capability,     # (LoopReport, PromotionDecision) -> CapabilityRequest
)
report.complete, report.to_dict()
```

## What the platform owns

- **Compute before the step.** It calls `ComputeController.choose()` with the
  cycle's `ComputeContext`, passes the decision to `step`, observes the
  reward/cost, and writes `compute_action` / `compute_option` / `compute_model`
  / `features` into the observation evidence. A caller's own `action` key is
  preserved. Compute observation is at-least-once: a retried generation calls
  `observe` again, so callers that retry a failed step should account for that.
- **The durable loop.** `LoopKernel` runs the generations; `apply` and
  `rollback` are passed straight through.
- **Promotion.** The best accepted generation is handed to the `evidence`
  callback; `promote()` makes the decision with the platform's rules.
- **Foundry.** The `capability` callback receives the loop report and promotion
  decision; a returned `CapabilityRequest` is admitted by the foundry.
- **Experience.** Every generation becomes an `ExperienceRecord` (reward map
  from `evidence["rewards"]`, action from `evidence["compute_action"]`), is
  ingested once, and the engine updates with the supplied validation/anchor
  rows. If no validation rows are supplied, the cycle's own records are used and
  the report marks `self_validated=True` — that is a warm-up, not held-out
  evidence. Anchors default to the most recent **pre-cycle** record; on the
  first cycle they bootstrap from the cycle itself. A campaign reuses the same
  rows every cycle unless the caller supplies fresh ones.
- **Completeness.** A configured layer that does not produce a decision makes
  `complete` false. A layer skipped by an upstream decision (for example the
  foundry after a rejected promotion) is named in `skipped` and `detail`, never
  silently skipped.
- **Gate / risk gate.** The constructor's optional `gate` / `risk_gate` are
  passed straight to `LoopKernel`; they act on the loop's control decision, not
  on promotion.

## What it deliberately does not do

It adds no decision logic. It does not invent evidence, rerun a gate, or decide
whether a promotion is good — it only hands the tested outputs to the next
layer. A cycle with no `evidence`, `capability` or `experience` runs just the
loop and is still `complete`.

