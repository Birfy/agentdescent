Added
End-to-end self-evolution bench (`bench/self_evolution.py`): one command runs
the whole chain — seed the archive (`continuous_meta_run`), diversify
(`meta_population_run`), re-judge with truth (`exam_archive`), reallocate spend
(`layers_from_archive` + `portfolio_report`), compose competing harnesses
(`compose_run` / `harness_leaderboard` / `select_harness`) and trace one
adoption (`chain_audit`) — on a deterministic offline arithmetic domain, and
reports every rung's measured number. Exits non-zero only when a rung fails to
run or produces no measurement; a raising rung is recorded as a failed stage,
not a crash. `--json` prints the structured report.
