Added
`SearchFabric`: a process reward model (and optional verifier ensemble) as the
value function for `MCTSPlanner`, plus PRM ranking/selection for candidates
outside the planner. `MCTSPlanner` now returns the chosen child state in
`PlanResult.state`, so the fabric reports that state without stepping a
possibly side-effecting simulator twice. An incomplete verifier panel is
exposed as `verifier_complete=False` and can never report a pass.
