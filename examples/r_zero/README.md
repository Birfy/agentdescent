# R-Zero — AgentDescent candidate port

Challenger/Solver role memories with min(p,1-p) uncertainty from repeated solver samples; GRPO shape at the acceptance seam.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.r_zero.r_zero_challenger_solver --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
