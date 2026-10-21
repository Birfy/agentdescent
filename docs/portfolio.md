# Portfolio governance

The framework spends across layers — inner `evolve()` runs, meta sessions,
audit sampling, harness reruns, population islands — and nothing decided how
the next window's budget is split. `portfolio.py` is that rung: a *layer* is
one spending line, the portfolio view ranks layers by return-per-cost, and the
reallocation rule shifts the next budget to what pays.

## The pieces

| Piece | What it is |
| --- | --- |
| `LayerBudget(name, budget, spent, return_, samples, return_sd, diminishing, note)` | One spending line: the window's limit, what it actually spent, the measured return that spend bought, and how many measurements it rests on. |
| `portfolio_report(layers, *, floor, min_budget, min_samples, grow, shrink, cap)` | Ranks by `return_ / spent` and returns a `PortfolioReport` with the new budgets and why each line moved. Deterministic. |
| `layers_from_archive(store)` | One layer per decision slot, aggregated from the archive's rule records (return = mean measured reward, spend = mean rounds per rule, budget = total rounds, samples = rule count). |

## The reallocation rule

* **Grows** — a layer whose return-per-cost is at or above `floor` and whose
  return rests on at least `min_samples` measurements: `budget * (1 + grow)`,
  up to `cap`.
* **Shrinks** — a layer below the floor (it is not paying for its cost) or
  flagged `diminishing`: `budget * (1 - shrink)`.
* **Holds** — a layer with too few measurements: nothing is concluded from a
  sample of one, and the budget must not move on a guess.
* **Never zeroes** — a shrunk layer keeps at least `min_budget` budget (default
  5% of the layer's current budget), so a temporarily quiet layer is not
  killed by a busy one. (`floor` is only ever the return-per-cost threshold.)

The rule is deliberately a fixed, honest policy first — the product the
framework can deploy today. Evolving the reallocation policy itself via
`meta_run` is the next rung, and the fixed rule is its baseline.

## Composing the layers

```python
from agentdescent import ConsolidationStore, layers_from_archive, portfolio_report

layers = layers_from_archive(store)                    # one layer per decision slot
layers.append(LayerBudget(                             # audit bias as a layer
    name="audit", budget=200, spent=120, return_=-0.3, samples=40))
report = portfolio_report(layers, floor=0.05, min_samples=3)
print(report.reallocated, report.reasons)
```

The consolidation archive is the honest "which decision's evolution pays";
audit bias (a negative layer), the harness leaderboard, and population islands
are caller rows in the same table. Whatever the units of `budget` / `spent`
(tokens, calls, seconds), the rule only ever uses their ratio.
