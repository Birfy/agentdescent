# Marginal-gain stopping — stop when calls stop paying

`evolve` already stops on a target, patience, rounds, wall clock, calls, tokens
and diminishing returns. A campaign or long RSI run often wants a different
question: *is the next call still buying reward?* `MarginalGainStop` turns the
recent reward/call curve into a `stop_when` callable.

```python
from agentdescent import MarginalGainStop

policy = MarginalGainStop(min_gain_per_call=1e-4, patience=3, window=3)
evolve(..., stop_when=policy)
```

- The policy keeps the last `window + 1` `(calls, held_out_reward)` points from
  the `RoundInfo` the engine already passes.
- A measurement is **low** when the gain per additional call is below
  `min_gain_per_call`. `min_gain` is an optional absolute-gain floor combined by
  OR: if the recent gain is at most `min_gain`, the measurement is also low.
  With the default `min_gain=0`, the per-call bar is what decides — including
  the case this policy exists for, a run that is still improving but too slowly
  to pay for its calls.
- `patience` consecutive low measurements stop the run with the engine's
  `stop_reason="stop_when"`; `reason` names the gain, call delta and streak for
  the caller's report.
- A round that spent no additional calls does not establish a slope and never
  counts as low.
- `held_out_reward` is this round's measurement, not the best so far, so a noisy
  dip reads as a negative gain.
- The policy is stateful. Call `reset()` before reusing one instance for a new
  run (a campaign or a meta loop should not carry the previous run's points).

This is an economic stop, not a safety one: it says the search has stopped
paying for itself, not that the artifact is good. Pair it with a target or a
round budget when both matter.


## Budgeted search — stop when another expansion stops paying

`MarginalGainStop` stops a run by calls; `BudgetedSearch` applies the same idea
to UCT. It runs increasing budgets through `SearchFabric`, computes the value
gain **per expansion**, and stops when the marginal return falls below
`min_gain_per_expansion` (or when a `MarginalGainStop` fires).

```python
from agentdescent import BudgetedSearch

search = BudgetedSearch(fabric, patience=1, min_gain_per_expansion=1e-3)
report = search.run(state, budgets=(5, 20, 60))
report.best.value, report.stopped_reason, report.rounds
```

Each round records its budget, cumulative expansions, value, gain and
gain-per-expansion, so the compute spend is auditable. The fabric's planner
budget and RNG state are restored after `run()`.
