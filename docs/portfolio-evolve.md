# Portfolio evolution — evolve the rule that governs the spend

*Module:* [`agentdescent.portfolio_evolve`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/portfolio_evolve.py)

`portfolio_report` is a fixed, honest reallocation policy: grow the layers that
pay, shrink the ones that do not, hold the ones with too little evidence. The
framework's thesis is recursion — the rule that decides how the framework
spends should itself be improved by the framework. This module makes the policy
**evolvable**.

```python
from agentdescent import portfolio_evolve

report, result = portfolio_evolve(
    store,                       # or windows=[...], val_windows=[...]
    propose=offline_propose,     # rewrites the four numbers
    seeds=(0, 1, 2), val_seeds=(3, 4))
print(report["verdict"], report["val_gain"])   # adopt, +0.05 over the baseline
print(result.rendered)                          # the evolved parameter block
```

## The rule is four numbers — the shipped rule with different numbers

`ReallocationPolicy(floor, grow, shrink, min_samples)` is the evolvable rule,
and `apply(rows)` **delegates to `portfolio_report`** rather than
re-implementing it. The default values are `portfolio_report`'s own defaults, so
`ReallocationPolicy()` *is* the fixed policy the framework ships — the baseline
and the evolved rule cannot drift apart. `reallocation_spec()` is the
`ParamSlot` that compiles the four numbers.

## A window is the portfolio's own unit

`LayerBudget` rows are exactly what `portfolio_report` consumes; a window is a
set of them. A problem's reward is the **projected return** the policy's
reallocation would buy: each layer's measured return scaled by its budget ratio
under the policy, averaged over the window. The projection is not clipped (the
default `auc` reward still normalises); a custom `meta_reward` scores it in the
search **and** in validation, so the knob drives what gets adopted rather than
only the scorecard. `windows_from_archive(store)` cuts the archive's rule
records into contiguous windows (aggregated per slot, records without a slot
skipped) and splits them positionally into train and validation, so the evolved
rule is judged on windows it never trained on. Callers with their own history
can pass `windows=` / `val_windows=` directly.

## What the report says

The evolved policy is measured against the **fixed baseline**
(`ReallocationPolicy()`) on both splits. The report carries the per-window
validation, the mean train/val **gain** over the baseline, and an
`adopt`/`reject` verdict (adopt only when the validation gain is strictly
positive; a refused no-op is `reject`, not an adoption). An out-of-bounds
proposal (outside the `ParamSlot` bounds) is refused by the gate, so the
artifact stays the fixed baseline rather than committing a broken rule.
