# Godel Agent — AgentDescent candidate port

Artifact-owned recursive self-improvement prompt; upstream gatelessness available as --gateless.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.godel_agent.godel_agent_self_modify --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
