Added
`fit_power_law` / `plan_budget` fit a deterministic saturating compute curve
(`a - b * budget^-c`) with grid + local refinement, report R² and RMSE, and
invert it to the smallest budget that reaches a target. Unreliable fits are
refused by default and targets already observed are not extrapolated.
