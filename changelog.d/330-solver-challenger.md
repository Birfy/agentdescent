Added
Solver-challenger self-play (`agentdescent/selfplay.py`): `curriculum_evolve()`
co-evolves the solver and its task distribution. A `Challenger` owns a
task-generation spec; each round it generates a validated frontier batch, the
`FrontierSampler` mixes those into the solver's train pool, and the challenger
tunes its spec on the measured learnability (GoDoors-style `r0` = a frontier
task's last solver score minus its first). The engine runs once -- one
aggregator, one ledger, held-out fixed from the seed tasks -- so the challenger
never touches the reported evaluation. A validity gate is mandatory
(`Challenger.validate` raises on malformed tasks; failures are counted), the
frontier is self-regulating (tasks the solver already passes leave instantly,
tasks it never learns are dropped as stuck), and the sampler may return a Task
object directly, which the engine resolves without a pool lookup (both the
synchronous and barrier-free paths). `EvolutionResult.curriculum` carries the
run's `r0_mean` / learned / stuck / invalid / generated stats. Offline demo:
`python -m examples.selfplay.selfplay_demo --dry-run`.
