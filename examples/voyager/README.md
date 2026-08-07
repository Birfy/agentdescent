# Voyager — AgentDescent candidate port

Add-only per-goal skill library; repair from the deterministic world's first-failure feedback; critic as the self-verify rollout.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.voyager.voyager_skill_library --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
