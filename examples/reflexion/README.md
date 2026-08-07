# Reflexion — AgentDescent candidate port

Bounded append-only episodic memory (last 3 entries), reflect on external feedback, retry via the gate rerun.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.reflexion.reflexion_episodic_memory --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
