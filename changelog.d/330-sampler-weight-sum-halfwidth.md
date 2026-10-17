Fixed
`audit.sampler.plan` now refuses a stratum-weight total that is not finite and
positive (two finite `1e308` weights used to overflow the sum to `inf` and raise
`ZeroDivisionError`), and requires `AuditPolicy.target_halfwidth` to be finite
and `> 0` (0 / negative used to silently fall back to `max_labels`; ±inf
collapsed the Neyman total so only the floors decided) (#330).
