Fixed
`audit.ppi` refuses what it used to turn into a NaN or zero-width interval:
`Stratum` raises `PPIError` for a non-finite `weight`, any non-finite score in
`f_lab`/`y_lab`/`f_unlab`, non-finite `mean_unlab`/`var_unlab`, and a non-integral
or non-finite `n_unlab`; `ppi_mean_stratified` raises `PPIError` unless
`0 < alpha < 1` and `k_folds >= 2`, and its weight-sum check now fails on NaN.
`Calibrator` rejects an invalid `alpha` with `ValueError` at construction, and
`Calibrator.recompute` turns any unexpected estimator exception into a stale
rectification, keeping `current()`'s "never raises" contract (#314).
