# PromptBreeder — AgentDescent candidate port

Binary-tournament population over a task/mutation-prompt genome; zero-order, first-order, and hyper-mutation operators.

The definition is a declarative `MethodPolicy` (see `examples/_method_policy.py`);
scheduling, budgets, and merging live in `examples/_method_runner.py`. Fidelity
class and boundaries are recorded in the module's docstring and `notes`.

```bash
python -m examples.promptbreeder.promptbreeder_genetic_prompts --dry-run
```

Part of the candidate-method runtime matrix: `python -m bench.candidate_methods`.
