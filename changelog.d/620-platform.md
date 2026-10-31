Added
`RSIPlatform` + `PlatformConfig` / `PlatformReport`: one bounded cycle that hands
compute allocation, the durable loop, trusted promotion, capability admission and
continual experience to each other. `complete` is false when a configured layer
does not produce a decision; a layer skipped by an upstream decision (for example
the foundry after a rejected promotion) is named in `skipped` and `detail`.
Post-loop decisions are cached per `run_id`, so re-running a terminal run returns
them instead of re-admitting or re-learning; a terminal run without a cache is
refused. A cycle that validates experience on its own rows marks
`self_validated=True`.

Changed
`LoopGeneration` now carries the observation's evidence, so the platform (and a
caller's own report) can see the compute action/option/features that produced a
generation without re-reading the event payload.
