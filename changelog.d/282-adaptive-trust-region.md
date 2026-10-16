Fixed
`AdaptiveTrustRegion` moves each cap by at least one unit per widen or
tighten (#282). With plain `round(cap * factor)`, the default `widen=1.25`
mapped an op cap of 2 to `round(2.5) == 2` (and 1 to 1), so two reversals
from the default 6 left the op cap stuck for the rest of the run while
`chars` and `history` kept moving. A tighten no longer clamps a cap that sits
below `minimum` *up* to it, so a rollback can never loosen the region.
`maximum` stays a hard ceiling. `window < 1` (which raised
`ZeroDivisionError` on the first `observe`), `widen < 1`, `tighten` outside
`(0, 1]`, non-finite factors, `accept_rate_to_widen` outside `[0, 1]`, and
non-positive bounds are rejected with `ValueError` at construction.
