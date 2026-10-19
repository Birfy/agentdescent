Added
Meta-curriculum (`agentdescent/meta_curriculum.py`): the meta loop generates
the problems it is judged on. `curriculum_meta_run` is `continuous_meta_run`
plus a `MetaProblemChallenger` — the solver-challenger of `curriculum_evolve`
(#329) applied one level up, where the challenger generates frontier *problems*
instead of frontier *tasks*. Between outer rounds the challenger draws a
candidate problem configuration, runs it through a mandatory validity gate
(`validate(config) -> Problem`; a payload that does not build a runnable inner
problem is counted, never crashed over), and installs the validated problems
into a `FrontierSampler` that mixes them with the seed problems. Learnability
is two-phase and measured online (`record` sees the meta reward; a problem the
rule already solves is dropped at once, one it never learns is dropped as
stuck), and the challenger's `propose(descriptor, report)` pushes the
generation spec outward. The challenger never touches the final evaluation —
the validation set is fixed from `val_problems` and the held-out split is
computed from the seed tasks alone. The archive still drives continuity: the
search starts from the best archived rule in a fresh ledger repo, and on
`adopt` the rule is written back with its problem descriptor. The report
carries the decision fields plus the frontier statistics (`generated_total`,
`invalid_total`, `learned`, `stuck`, `r0_mean`). Offline demo:
`python -m examples.meta.curriculum_meta_demo`.
