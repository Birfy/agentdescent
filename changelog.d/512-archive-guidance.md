Fixed
Missing-archive errors from `safety` / `portfolio` / `audit` (CLI and MCP) now
name `continuous_meta_run` and say that `evolve` / plugin runs do not write an
archive, so a host agent does not invent a nonexistent "consolidation" step.
Those read-only verbs also leave an existing empty archive untouched (no
`runs/` directory as a side effect) (#512).
