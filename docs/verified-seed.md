# Verified seeding — trust becomes a gate, not a report

*Module:* [`agentdescent.verified_seed`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/verified_seed.py)

The self-exam stamps adoptions `passed` / `flagged` / `demoted`, and the
meta-audit measures the oracle-vs-verifier gain bias. `_archived_rules` already
retires a `demoted` rule. This module turns the rest of that trust into a
**seeding gate**: only rules that clear the caller's bar may seed the next
search — a rule that failed truth is not merely reported, it is never used
again.

```python
from agentdescent import verified_rules

seed = verified_rules(store, slot="task_sampler", spec=spec, config=CONFIG,
                      min_exam="passed", meta_audit=audit, bias_threshold=0.1)
for rule in seed.rules:        # only certified, unbiased rules seed
    ...
print(seed.excluded)           # run_id -> why it did not qualify
```

## The bar

- **`min_exam`** — when set, only rules whose `exam` status is **at or above**
  it qualify, ordered `None < flagged < passed`. `"passed"` (the default) means
  a rule the oracle has **not examined** (`exam is None`) or that **failed
  once** (`flagged`) does not seed; `"flagged"` keeps flagged and passed rules.
  `min_exam=None` reproduces `_archived_rules` exactly (backwards compatible).
  Anything else raises rather than silently emptying the seed set.
- **`meta_audit` + `bias_threshold`** — when a `MetaAuditStore` is supplied, a
  rule whose recorded gain bias for its run exceeds the threshold is excluded
  **even if it passed the exam**.

`excluded` carries the reason per run, so "this rule did not seed" is
answerable.

## One knob that reaches the loop

`continuous_meta_run(..., min_exam="passed")`, `meta_population_run(...,
min_exam=...)` and `async_meta_population_run(..., min_exam=...)` filter the
archive's rules before they seed, so the certification bar is applied at the
seeding decision itself — not by a caller remembering to filter. The default
`None` keeps today's behaviour unchanged.
