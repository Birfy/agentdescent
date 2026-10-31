Added
`BudgetedSearch`: runs `SearchFabric` at increasing UCT budgets and stops when
the value gain per expansion falls below a bar, using the existing
`MarginalGainStop` or a patience rule. The report keeps every round, the gain
it bought and why the search stopped, so compute spend is auditable rather than
hidden in one fixed budget.
