Added
Portfolio evolution (`agentdescent/portfolio_evolve.py`): the framework evolves
the rule that governs its own spend. `ReallocationPolicy` is the reallocation
rule as four numbers (`floor`, `grow`, `shrink`, `min_samples`); its `apply`
delegates to the shipped `portfolio_report`, so the evolved rule is the fixed
policy with different numbers (the default is the baseline, no second
implementation to drift). `reallocation_spec` compiles the numbers with
`ParamSlot`; a `LayerBudget` window is the problem, scored by the projected
return the policy's reallocation would buy; `portfolio_evolve` runs the ordinary
`evolve()` over the archive's (or the caller's) windows and reports the evolved
policy's validation gain over the fixed baseline with an adopt/reject verdict.
