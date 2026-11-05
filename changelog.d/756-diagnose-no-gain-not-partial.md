Fixed
`diagnose` no longer calls a run `partial-improvement` when its held-out reward
never rose above the seed's (for example `0.000 -> 0.000` with only
`dgm-select` parent-pick commits); such a run falls through to
`agent-failures` / `no-improvement`, and `partial-improvement` now names
`initial -> final` (#756).
