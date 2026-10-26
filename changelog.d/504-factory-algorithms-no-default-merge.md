Fixed
`gepa`, `skillopt` and `dgm` specs no longer get the default reflective
merge pair their own aggregator ignores, so `evolve` stops emitting a
`RuntimeWarning` about `conflict`/`fusion` policies nobody wrote; a spec that
names those policies explicitly gets the warning from `plan` instead (#504).
