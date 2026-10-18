Fixed
`plan` now also refuses `evolve.allocator` (it is annotated `Any` but is an
object) and `evolve.extra_artifacts` (values must be `EvolvingArtifact`
instances), and `kind: policy_slot` converts `evolve.agg_config` through
`build_agg_config` the same way every other kind does -- without that, `plan`
said ok and the run died on `'dict' object has no attribute
'audit_drain_per_step'` (#385 review / #384).
