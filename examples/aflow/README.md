# AFlow — AgentDescent candidate port

Soft mixed-probability workflow selection (the pinned revision's rule, not UCT) with per-parent experience in the expansion prompt.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.aflow.aflow_workflow_search --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
