Added
Rationale distillation (`agentdescent/rationale_distill.py`): the artifact
learns to carry its own reasoning. `extract_rationale` turns one rollout into a
rule — a success says how it answered, a near-miss says what it needed
(`need cap >= d+1`) — and `distill` dedupes the rollouts' rationales into the
skill the artifact carries; `rationale_evolve` runs the ordinary `evolve()` and
then distills the run's own successes back into the skill, returning a
`RationaleReport` with the distilled skill, the rationale count and the success
count. The STaR (Self-Taught Reasoner) idea: the artifact boots itself by
carrying what its own runs taught it, near-misses included.
