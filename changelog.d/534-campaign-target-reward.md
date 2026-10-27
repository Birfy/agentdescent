Fixed
`campaign` now stops as soon as a session reaches the spec's `target_reward`
and reports `stopped_reason: "target_reward"`, instead of spending another
session and then reporting `patience` (#534).
