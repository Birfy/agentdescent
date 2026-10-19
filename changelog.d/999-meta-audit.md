Added
Meta-audit — the trust layer for the loop that evolves the rules
(`agentdescent/meta_audit.py`). `audited_meta_validate` scores every
`(problem, seed)` validation pair under the meta reward for the decision, and
with probability `sample_rate` scores the pair again under an independent
oracle and persists it to a `MetaAuditStore` — a probability sample (every
record carries its `inclusion_prob`) with a durable, append-only JSONL trail
that reconciles by last occurrence, so truth arriving later from another
process can be attached without a rewrite. `meta_run_audited` is `meta_run`
with this validation: the verdict is persisted beside the sampled evidence,
and the report carries `audited_n` / `oracle_gain` / `gain_bias`. Over the
resolved sample, `gain_bias` is the Hajek estimate of `verifier_gain -
oracle_gain` — the number that says "your meta reward is 0.3 optimistic"
instead of letting that optimism adopt rules in silence. The decision path
never sees the oracle (auditing can be turned on mid-fleet without changing
which rules are adopted), and the same `_require_disjoint` isolation `meta_run`
enforces applies.
