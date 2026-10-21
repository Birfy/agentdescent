Added
Async meta-population (`async_meta_population_run`): the islands of
`meta_population_run` run concurrently. Each island starts from its own
archived rule, trains on the shared problems, validates the shared isolated
set, and runs in its own fresh ledger repo — so they are independent sessions
run through a `ThreadPoolExecutor` with `workers` islands at a time (default
all of them). The archive is only read while the islands run
(`ConsolidationStore` is thread-safe) and written once at promotion, so the
decision and the report are identical to the sequential call's, only faster to
reach. An errored island is reported and never promoted, exactly as sequential;
an empty archive is refused with the same pointer to run `continuous_meta_run`
first.
