# Risk policy — turn a change's context into one safety action

An acceptance gate answers "is this better?", not "what should we do about the
risk?". `RiskPolicy` is the ordered, declarative mapping from a change's context
to `allow` / `approve` / `canary` / `rollback` / `deny`.

```python
from agentdescent import RiskContext, RiskPolicy, RiskRule

policy = RiskPolicy([
    RiskRule("never-touch-verifier", "deny", domain="verifier"),
    RiskRule("l1-approval", "approve", tiers=(1,), min_blast_radius=0.5),
    RiskRule("large-canary", "canary", min_ops=6),
])
decision = policy.decide(RiskContext(tier=2, blast_radius=0.2, ops=8))
print(decision.action, decision.rule, decision.reason)
```

- Rules are evaluated in order; the **first match wins**. Every supplied
  condition must match (`tiers`, `min_blast_radius`, `min_ops`, `domain`,
  `tenant` are conjunctive).
- **A missing fact is `None`, not `0`.** `RiskContext(tier=None, blast_radius=None,
  ops=None)` matches any rule that tests the missing field: unknown risk is
  worst-case, not the lowest risk. `blast_radius` must be finite and in
  `[0, 1]`; `tier` and `ops` non-negative integers.
- No match uses the policy `default` (`deny` unless set). Invalid actions,
  duplicate/empty rule names and bad rule fields are refused at construction, so
  a typo cannot silently allow a change.
- A rule with no conditions matches everything; place a catch-all `allow` only
  at the end, where it cannot shadow stricter rules.
- `min_*` are lower bounds; "only small changes" is expressed by ordering a
  stricter rule first. The actions are alternatives, not a total order.
- `describe()` returns the rules in evaluation order for a plan or an audit
  export.

This policy is pure: it does not call a guard, an approval store or a canary. It
produces the decision those mechanisms then execute, which keeps the policy
auditable and testable on its own.
