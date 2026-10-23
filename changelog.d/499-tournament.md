Added
`run_tournament` / `agentdescent tournament`: run the same spec under several
registered algorithms (and seeds) with the same evolve budget, then report the
measured ranking and winner. The winner is aggregated per algorithm over seeds
(mean reward, fewer mean calls on a tie); a tie is reported as a tie, and
`--dry-run` prices every arm before a run. `default` is the spec's own recipe; each named arm
gets a deep copy, so the spec file is untouched. Ordinary
`EvolveSpec -> compose() -> evolve()` per arm, not a second engine or a claim to
reproduce the ports' published numbers.
