Added
Algorithm registry (#384, slice 1): `EvolveSpec.algorithm` (`"rrsi"` or
`{"name": ..., "params": {...}}`) picks a named recipe whose `Policies` slots,
`agg_config` and `evolve` defaults the spec can still override; `agentdescent
algorithms list [--kind K] [--json]` and `algorithms show <name>` describe each
one (kinds, typed parameters with defaults, extra cost, fidelity label) from the
wheel. Unknown names, unknown or mistyped parameters and a kind the algorithm
does not apply to fail in `plan` naming the field and the valid choices, and
`plan` echoes the resolved algorithm and which of its defaults the spec
overrides. Two entries ship: `rrsi` (text; the RRSI screen / proposal /
acceptance policies) and `archive_selection` (DGM-style archive parent sampling,
any kind). The RRSI mechanisms moved from `examples/rrsi/` into
`agentdescent/algorithms/rrsi/` so the wheel carries them; the `examples.rrsi.*`
modules remain as aliases.
