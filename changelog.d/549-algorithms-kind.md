Fixed
`agentdescent algorithms --kind K` now filters like `algorithms list --kind K`
(with or without `--json`) instead of exiting 2 with a misleading
`invalid choice` error, and `--kind` with `algorithms show` is refused with a
clear message rather than ignored (#549).
