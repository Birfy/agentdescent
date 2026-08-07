# Agent0 — AgentDescent candidate port

Curriculum/Executor co-evolution with sandboxed calculator stop-and-go rollouts.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.agent0.agent0_tool_curriculum --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
