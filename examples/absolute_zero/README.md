# Absolute Zero — AgentDescent candidate port

Proposer/solver self-play with a grounded verifier; frozen evaluation carts; the 1-r propose-reward shape surfaced verbatim.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.absolute_zero.absolute_zero_selfplay --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
