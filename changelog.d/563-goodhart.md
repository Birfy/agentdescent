Added
`detect_goodhart` and `GoodhartMonitor` measure proxy-oracle inflation, rank
tracking, and concrete shortcut candidates, so a self-improving loop can alarm
when the reward it optimises stops tracking the reward it cares about.
