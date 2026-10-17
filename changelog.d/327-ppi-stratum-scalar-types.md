Fixed
`audit.ppi.Stratum` now refuses a numeric string or a `bool` in `weight`,
`n_unlab`, `mean_unlab` or `var_unlab` with a `PPIError` that names the field,
and stores those fields as plain `float` (`int` for `n_unlab`). Previously
`weight="0.5"` escaped as an unrelated `TypeError`, `mean_unlab="0.3"` was
stored as a string and broke `ppi_mean_stratified` later, and `True` was
silently read as 1 and reported back as `True` in `per_stratum`. numpy scalars
and integral-float counts are still accepted (#327).
