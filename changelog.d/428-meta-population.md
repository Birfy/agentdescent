Added
Meta-population (`agentdescent/meta_population.py`): the archive seeds a
population, the population promotes. `meta_population_run` ranks the archive's
rule records for a slot/spec/config as `continuous_meta_run` does, seeds the
top `islands` distinct rules each into its own session, and every island trains
on the same problems and validates the same `val_problems` at `val_seeds`
disjoint from training — so the islands differ only in where they started. Each
island runs in its own fresh ledger repo (`repo_dir=` names the base directory;
a shared `repo_path` would resume the ledger and silently ignore the distinct
`initial_state`), the best island whose verdict is `adopt` is written back via
`archive_rule`, and `MetaPopulationReport` / `PopulationIsland` report every
starting rule's outcome, the promotion and the archive count. An island that
errors is reported and never promoted without stopping the others; a population
needs at least one archived rule to diversify from (an empty archive is
refused, with a pointer to run `continuous_meta_run` once first). Offline demo:
`python -m examples.meta.population_meta_demo`.
