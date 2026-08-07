# SICA — AgentDescent candidate port

AST-gated real source self-edits with Archive (performance) base selection.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.sica.sica_self_edit --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
