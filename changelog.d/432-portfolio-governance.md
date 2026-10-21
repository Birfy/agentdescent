Added
Portfolio governance (`agentdescent/portfolio.py`): watch every layer's
measured return, shift the budget to what pays. A `LayerBudget` is one spending
line (a spend limit, what it spent, the measured return that spend bought, and
how many measurements it rests on). `portfolio_report` ranks layers by
return-per-cost and reallocates the next window deterministically and
conservatively: a layer at or above `floor` with at least `min_samples`
measurements grows (`budget * (1 + grow)`, capped), one below the floor or
flagged `diminishing` shrinks (`budget * (1 - shrink)`), one with too few
measurements is held (no budget moves on a sample of one), and no layer is ever
zeroed. `layers_from_archive` turns the consolidation archive's rule records
into one layer per decision slot (return = mean measured reward, spend = mean
rounds per rule, budget = total rounds, samples = rule count) — the honest view
of "which decision's evolution
pays" — and other layers (audit bias, harness leaderboard, population islands)
are caller rows in the same table. Non-finite inputs are refused. The rule is
the fixed, deployable baseline for the next rung: evolving the reallocation
policy itself via `meta_run`.
