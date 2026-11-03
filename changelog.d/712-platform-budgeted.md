Changed
A `platform` spec may now set `step: {"budgeted": {...}}` to use the existing
`BudgetedStep` (PRM/verifier search with marginal-gain stopping) as the durable
loop's step. The block takes a `SearchFabric` builder/instance, a state
callable, `BudgetedStepConfig` and optional `risk`/`evidence_of` callables;
`plan` validates refs and config without calling the builders. This gives the
previously isolated `budgeted_loop` a product consumer without changing the
platform's compute or promotion behavior.
