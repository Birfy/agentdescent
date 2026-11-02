Added
`BudgetedStep`: composes `SearchFabric` and `BudgetedSearch` into a durable
loop step that scales search budgets with the controller's `budget_scale`,
stops on marginal gain, and returns a `LoopObservation` with round-by-round
evidence, risk and an artifact id (#678).
