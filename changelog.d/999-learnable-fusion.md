Added
LearnableFusionPolicy (`agentdescent/fusion.py`, exported from the package
root): a `FusionPolicy` that learns when merging wins. An online logistic over
diff-only features (candidate count, key overlap, diff sizes) is trained on the
tournament's own `FusionTrial`s, and once confident that the union is the right
answer for a merge shape it returns the union **without buying the cheap
ranking sweep**; uncertain merges still run the tournament and observe a fresh
label. Counterfactual exploration (`epsilon`) re-verifies confident skips, and
the rollback guard retires skipping permanently after a confident-and-wrong
counterfactual (default `max_wrong_skips=1`) -- a rule that learned to be wrong
is disabled, not patched. With no trials the policy is
`DefaultFusion(tournament=True)` verbatim; `tournament=False` (the budget
governor's flip) degrades to the plain union. Commits are unaffected: a skip
returns the same union the non-tournament default would have committed, and the
acceptance gate still scores it on the full held-out set.
