# The runtime matrix

The runtime matrix asks one question of the eleven
[`MethodPolicy` ports](self-evolution-examples.md#the-eleven-microports-and-analogues):
**what does changing only the scheduler do?** Same method, same candidate and
proposal-call budget, three runtimes:

- `serial` — `evolve(max_concurrency=1)`
- `sync_parallel` — `evolve(max_concurrency=workers)`, with its round barrier
- `async_pipeline` — `async_evolve(...)`, completion-order merge sweeps

Nothing else varies, which is what makes the timing columns attributable to the
scheduler rather than to the algorithm.

```bash
python -m bench.candidate_methods --provider openai --model glm-5.2 \
  --workers 2 --candidates 2 --repeats 1 --seed 0 \
  --modes serial sync_parallel async_pipeline --yes
python -m bench.candidate_methods_report --input bench/results/<run>.json
```

The report writes [`docs/matrix-report.md`](matrix-report.md) — that page is
generated, so edit the generator rather than the page.

!!! warning "The current report predates the restructuring"
    [Runtime matrix — report](matrix-report.md) was produced against source
    fingerprint `381b663…`, before the ports moved to their declarative
    `MethodPolicy` form and before the hand-written arithmetic fixture was
    replaced by GSM8K, GSM-Hard and the widened generated domains. **Its timing
    columns still describe a real experiment** — 1.36× end-to-end and 1.89×
    inside the engine window, paired over n=33, with 11/11 methods showing a
    median sync-over-serial win — and are retained for provenance. **Its quality
    columns do not**: they were measured on the fixture whose baseline was 0.000
    by construction, which is why every cell reads `0.000 -> …`. Current quality
    numbers are the three-seed async rows on each port's own page, summarised in
    [measured results — all eighteen](self-evolution-examples.md#measured-results-all-eighteen).

## What the rerun has to report

The comparison is equal-*candidate*, not equal-*work*, so three columns have to
sit beside any headline:

* **Cost.** The barrier-free loop performs more rollouts and more merge sweeps
  for the same candidate budget. `observed_rollouts` and `observed_fusion_calls`
  are recorded per run for this reason.
* **Target-reach rates.** The time-to-quality ratio drops any pair where either
  side missed the target, so the reach rates are the denominator a reader needs
  before quoting a TTQ figure; the summary JSON carries
  `per_mode_target_reach`.
* **Merge behaviour.** Batches are sized to the worker count, so concurrent
  proposals now actually meet in one merge — union on library and memory
  artifacts, reflective merge on contested text. Whether merged unions change
  the speedup picture is an open measurement.

Self-verify methods ([Voyager](algo-voyager.md), [SkillWeaver](algo-skillweaver.md))
pay one extra rollout per candidate in every mode, so the comparison stays
equal-budget across them.
