# MethodPolicy porting checklist

Use this for mechanism microports and analogues — declarative
[`MethodPolicy`](policies.md) definitions measured in the
[runtime matrix](matrix-overview.md). Benchmark-faithful ports follow the
[main checklist](porting-checklist.md).

- [ ] Declare the fidelity class up front (`mechanism_microport`,
  `environment_analogue`, `inference_analogue`, `self_edit_analogue`) and pin
  the upstream revision you traced.
- [ ] Define the method as a `MethodPolicy` in `examples/<name>/` — frozen
  datasets, pure `solve`/`propose`/`reward`, a shared or local
  [strategy](strategies.md), and a `main()` via
  `examples._method_runner.standard_main`.
- [ ] Put the mechanism at the standard seams: `engine=Policies(selection=…,
  task_sampler=…, acceptance=…)`; a local ~15-line policy subclass where the
  upstream rule differs from a shipped one. Reach for
  [`aggregator_factory`](aggregator-factory.md) only when the mechanism needs
  state the pipeline does not keep.
- [ ] Validation lives once, in the strategy's `to_diff`: an unparseable
  proposal costs its candidate, is counted, and produces no diff — **no
  fallback substitution anywhere**, and never a gold answer in a prompt.
- [ ] Set `reflective=False` when artifact values are code or strict JSON
  (synthesised merges bypass the validator).
- [ ] Freeze evaluation splits per seed; self-generated curricula must never
  shape their own test set.
- [ ] Add offline tests to `tests/test_candidate_methods.py` (the matrix test
  runs every method in every scheduler without an API key) and register the
  builder in `bench.candidate_methods`.
- [ ] Add `docs/algo-<name>.md`: mechanism, seam wiring table, boundaries, and
  the matrix-results placeholder.
- [ ] Record the port author and the upstream trace, exactly as Path A does.
