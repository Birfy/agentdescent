# Evaluation service — verifier vs oracle under a hard budget

An adoption decision often has two signals: a cheap verifier on every task and an
expensive oracle on a sample. `evaluate` makes that one explicit operation:
deterministic sampling under a hard oracle-call budget, per-task rows, coverage
and signed/absolute bias.

```python
from agentdescent import evaluate

report = evaluate(artifact, tasks, verifier=verifier_score,
                  oracle=oracle_score, sample_rate=0.2, budget=50, seed=0)
print(report.bias, report.abs_bias, report.coverage, report.oracle_calls)
```

- The verifier runs on **every** task; the oracle only on the sample.
- The sample size is `min(budget, ceil(len(tasks) * sample_rate))` (at least
  one when both are positive). `budget=0` audits nothing; `oracle_mean`, `bias`
  and `abs_bias` are then `None`, not zero. Sampling is seeded: same `seed` →
  same rows.
- `EvaluationReport` carries tasks, audited count, verifier/oracle means, the
  audited-subset verifier mean, signed/absolute bias, coverage, oracle calls and
  the rows. `bias` is measured on the audited subset, so
  `verifier_mean - oracle_mean` is not generally it.
- Non-finite verifier/oracle scores raise a `ValueError` naming the task; a
  missing `task.id` falls back to its position (visible in the row).
- If the oracle raises partway through, the exception propagates and the
  already-paid calls are not reported. Use a small budget or catch at the
  caller.

A zero-coverage report means **no truth evidence**, not a pass. Pair it with
`stats_eval` and the decision log; the service does not decide.

## Relationship to `audit`

The `audit` package is the stratified/PPI machinery for a production verifier;
`evaluate` is a small, explicit pass for a spec/session that just needs "score
these tasks with the verifier and this many with the oracle". Use `audit` when
you need boundary stratification and rectified estimates; use `evaluate` for a
one-shot decision/evidence row.
