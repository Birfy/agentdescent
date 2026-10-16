Fixed
`CladeThompson.tau()` reads `budget_remaining` like `CostEfficient` (#280):
`+inf` (no ceiling) and NaN mean a full budget (`tau = 1`) instead of an
exhausted one (`max_tau`); the power is taken in log space and capped, so a
small fraction with a steep `cool_down_beta` no longer raises
`OverflowError` mid-`select()`. Non-finite `cool_down_beta` / `max_tau` are
rejected with `ValueError`.
