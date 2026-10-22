Added
Verified seeding (`agentdescent/verified_seed.py`): trust becomes a gate, not a
report. `verified_rules` starts from `_archived_rules` and returns only the rule
records that clear the caller's bar -- `min_exam` (default `"passed"`, so an
unexamined or once-flagged rule does not seed) and, when a `MetaAuditStore` is
supplied, a gain bias within `bias_threshold` -- and explains, per excluded run,
why it did not qualify. `continuous_meta_run`, `meta_population_run` and
`async_meta_population_run` gain a `min_exam` knob that filters the seeding
rules before the session starts; `None` keeps the existing behaviour unchanged.
