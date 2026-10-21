Added
Self red-teaming (`agentdescent/red_team.py`): the framework attacks its own
value to find the OOD gap. `red_team` mutates the training tasks outward and
scores the current value on each mutation, collecting the failures — the
out-of-distribution region where the value does not generalise (deterministic,
reproducible); and `red_team_evolve` runs the ordinary `evolve()`, attacks its
own result, and retrains on the union of the problems and the failures it
found, closing the OOD gap. `RedTeamReport` shows the failures found and the
coverage (fraction of the attack's mutations solved) before and after the
retraining, so "the framework found its own blind spot and fixed it" is
measured, not claimed.
