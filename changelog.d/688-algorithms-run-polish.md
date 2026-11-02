Fixed
`agentdescent algorithms run` now prints `wrote <spec>` before the error when
the plan step fails after writing the spec, rejects an invalid group-level
`--kind` with the list of kinds before writing anything, and accepts
`--agent REF` like `init` (a mismatched ref is refused before writing) (#688).
