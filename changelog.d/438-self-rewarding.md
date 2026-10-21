Added
Self-rewarding meta-evolution (`agentdescent/self_rewarding.py`): the judge
improves as the run goes. `outcome_features` turns a `MetaOutcome` into
curve-shape features (mean best-so-far, final, sweeps, rollouts — never the
answer); `LearnableMetaJudge` is a small pure-Python linear regressor from those
features to oracle quality, fit by least squares (deterministic, no numpy),
trained **only on labels the oracle actually produced**; and
`self_rewarding_meta_run` alternates — evolve the slot with the plain `auc`,
score `judge_sample` outcomes with the oracle and fit the judge, then re-evolve
with the judge as the meta reward and refit on the newly oracle-labelled
outcomes — so each round the search moves against a closer approximation of
truth. The report carries the judge's measured agreement with the oracle on
validation outcomes it never trained on (`SelfRewardingReport.judge_agreement`),
how many labels it learned from, and the evolved value's oracle gain over the
seed. The Self-Rewarding Language Models (Meta 2024) idea applied to the meta
level.
