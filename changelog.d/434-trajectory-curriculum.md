Added
Trajectory-driven curriculum (`agentdescent/trajectory_curriculum.py`): the
problems come from the run's own failures. `failed_rollouts` reads a rule's
linked `TrajectoryStore` (via `ConsolidationRecord.trajectory_store`, #334) and
returns the rollouts whose reward fell below `failure_threshold` — the memory
of #248 turned into evidence. `trajectory_challenger` is a
`MetaProblemChallenger` whose generation spec is an archived run_id: `generate`
mines that rule's recorded failures, `validate` builds a frontier problem over
the failed tasks via `problem_from_tasks` (raising — and so counted invalid —
when there is nothing to mine), and `propose` advances across the archive's
rules that recorded trajectories. Plug it into `curriculum_meta_run` (#429) and
the meta loop's frontier is generated from what its own runs got wrong, with
the #429 guarantee inherited unchanged: the validation set stays fixed and the
held-out split is computed from the seed tasks alone, so a mined failure can
only ever appear in the training frontier.
