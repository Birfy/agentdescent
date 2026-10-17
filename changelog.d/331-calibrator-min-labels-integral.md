Fixed
`audit.calibrator.Calibrator` now refuses `min_labels` / `min_per_stratum` that
are not non-negative integers (`numbers.Integral`, not `bool`) at construction,
matching `AuditPolicy` (#325). Floats, numeric strings and negatives used to be
stored and later poison folding or the stale message as `TypeError` (#331).
