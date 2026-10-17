Fixed
`PopulationAggregator.restore` drops island fork names whose ledger branches
were already deleted by `finalize`, and `step` recreates a missing island
head before committing to it — a clean-stop checkpoint no longer routes workers
to nonexistent branches on resume (#339).
