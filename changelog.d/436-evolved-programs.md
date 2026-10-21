Added
Evolved decision code that beats the shipped implementation on quality-per-cost
(`agentdescent/evolved_programs.py`). `source_ops` counts the arithmetic/call
nodes of a candidate rule's source — a deterministic, attributable cost proxy;
`priority_grid_problem` scores a compiled `priority` rule's ordering agreement
with the shipped seed's own ordering on a fixed grid (the seed scores 1.0 at
the baseline cost); `cost_adjusted_reward` is `quality - lam * (cost / baseline)`,
so a candidate must be cheaper at equal quality (or better at equal cost) to
outscore the seed; and `evolve_fast_rule` evolves the `selection` priority rule
on that reward and reports the evolved rule against the seed on **both axes**
(`FastRuleReport.seed_ops` / `evolved_ops` / quality / the cost-adjusted gain).
The AlphaEvolve / FunSearch idea applied to the decision plane: the search
finds a rule that is as good *and cheaper* than the hand-written default, and
the report measures it.
