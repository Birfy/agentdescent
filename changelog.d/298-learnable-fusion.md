Added
LearnableFusionPolicy (#298): a `FusionPolicy` that learns when merging wins.
An online logistic over diff-only features (candidate count, key overlap, diff
sizes) is trained on the tournament's own `FusionTrial`s, and once confident
that the union is the right answer for a merge shape it returns the union
**without buying the cheap ranking sweep**; uncertain merges still run the
tournament and observe a fresh label. The rollback guard scores the **pre-
update** prediction -- the rule actually in force -- and retires skipping
permanently after a confident-and-wrong counterfactual (default
`max_wrong_skips=1`). The budget-governor's `tournament=False` degrades to the
plain union and keeps recording unranked trials on the list `fusion_stats`
reads. With no trials the policy is `DefaultFusion(tournament=True)` verbatim.
Commits are unaffected: a skip returns the same union the non-tournament
default would have committed, and the acceptance gate still scores it on the
full held-out set.
