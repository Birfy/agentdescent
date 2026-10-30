Added
`TaskPipeline` + `TaskPipelineConfig` / `TaskPipelineReport`: production
`TraceSample`s are mined by `DataFlywheel`, accepted failures become seeds, and
`TaskFoundry` validates, deduplicates, checks contamination and splits them into
train and holdout. The pipeline respects the foundry's own verdict (`insufficient`,
`filtered`, `no-train`) instead of turning it into `ready`; only an empty seed
set maps to `empty`, and `no-holdout` is explicit.
