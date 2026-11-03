Added
`agentdescent demo --algorithm NAME` runs the offline demo with a named
algorithm, so one can be watched end to end before wiring a model and data.
Only algorithms verified against the offline agent are accepted
(`archive_selection`); others are refused with the `algorithms run` command
that runs them for real (#719).
