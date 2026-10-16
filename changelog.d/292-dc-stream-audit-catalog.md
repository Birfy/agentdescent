Fixed
Dynamic Cheatsheet online mode now matches its claim (#292, follow-up to #317):
`MethodPolicy` gains opt-in `cheap_eval_tasks` (default `None`, engine default)
so a cheap/full held-out disagreement can no longer lower the audit's trust
below 0.75 and veto a rewrite the accept-everything policy meant to keep, and
opt-in `test_time_learning` (default `False`) so the runner scores the test
split as a sequential Gen -> Cur stream. No other port changes. The
self-evolution catalog counts (twenty-two ports, fourteen microports) are
corrected across the docs, README and install guide.
