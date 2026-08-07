# Self-Refine — AgentDescent candidate port

Separate FEEDBACK and REFINE calls with the upstream 'it is correct' early stop.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.self_refine.self_refine_feedback_loop --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
