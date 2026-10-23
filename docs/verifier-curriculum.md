# Verifier–curriculum co-evolution

A verifier is a proxy. A curriculum that only samples tasks the verifier already
scores well measures the proxy against itself, so the two can agree while both
drift from the oracle. This is the Goodhart failure mode of a self-improving
loop, and it is not fixed by auditing the verifier on a random sample: the tasks
where the proxy is wrong are the ones a random sample is least likely to pick.

`VerifierCurriculum` makes the task distribution target that disagreement:

```python
from agentdescent import VerifierCurriculum

cur = VerifierCurriculum(pool, verifier=verifier_score, oracle=oracle_score)
rows = cur.probe(artifact)                 # (task, verifier, oracle, gap) per task
hard = cur.select(artifact, n=8)           # largest |verifier - oracle| first
report = cur.coevolve(artifact, rounds=3, n=8, update=fix_verifier)
print(report.initial_abs_gap, report.final_abs_gap, report.improved)
```

`select(mode=...)` chooses the direction: `abs` (default) targets any
disagreement, `overestimate` targets a verifier that is too generous, and
`underestimate` one that is too harsh. `coevolve` probes the pool once per round, ranks those rows, and feeds each
selected task's already-paid oracle label to `update(task, oracle_score)`; the
caller mutates its verifier (a correction table, a fitted judge, a prompt), and
the next round's verifier score is re-measured without buying the oracle again.
The oracle cost is therefore one label per pool task per round -- a full-truth
probe, stated honestly: this finds disagreement by paying for truth, it does
not infer it from the verifier alone. The report is the gap before and after,
so "did the curriculum help the verifier?" is a measurement rather than a hope.

## At the acceptance seam

`VerifierAgreement` is an `AcceptancePolicy` that audits a fixed probe set on a
random fraction of commits and refuses a gain the verifier claims and the oracle
denies:

```python
Policies(acceptance=VerifierAgreement(
    audit_tasks, verifier=verifier_score, oracle=oracle_score,
    inner=default_acceptance, sample_rate=0.2))
```

It wraps the inner policy (it can only make acceptance stricter), and an
`inner` rejection is never overridden. `inner=None` is the shipped
`DefaultAcceptance`, configured through the forwarded `configure` hook. An
oracle regression on the audit set is refused whatever the verifier says; a
verifier-only gain is `oracle-disagrees`. The audit draw is seeded from the diff
id, so it does not depend on merge order.

Both pieces take `(artifact, task) -> float` callables, so they work with the
shipped verifier, a model judge, or a deterministic fixture. They are deliberately
separate from `meta_audit`: that store records sampled truth for a *run*, while
this module selects the tasks on which truth is most informative and lets the
verifier learn from them.
