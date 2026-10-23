Added
GEPA's Pareto illumination is now a named algorithm (`"algorithm": "gepa"`,
`kind: text`) in the in-package registry (#384, slice 5a). The Pareto pool,
per-instance scoring and win-frequency parent selection moved from
`examples/gepa` into `agentdescent/algorithms/gepa.py`; the example keeps its
HotpotQA domain, prompts and CLI and imports the one implementation, and the
generated catalog lists its `seed` / `eval_concurrency` parameters.
