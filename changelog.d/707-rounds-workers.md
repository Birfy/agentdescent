Added
`agentdescent algorithms run` and `agentdescent init` take `--rounds N` /
`--workers N` (each >= 1), written into the spec's `evolve` block, so a cheap
trial of a named algorithm (`--rounds 1 --workers 1`) needs no hand-edited JSON.
Without the flags the plain `algorithms run` output names that quick-trial path
when the default plan is large, and the host skill and `/algorithms` commands
mention it (#707).
