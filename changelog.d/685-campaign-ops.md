Added
`CampaignOps` + `CampaignOpsReport`: publish a sequence of finished
`LoopReport`s into the control room in one call, recording each into the
console, lineage, telemetry and QD and returning the campaign metrics, alerts
and snapshot. Unknown statuses, duplicate run ids, malformed costs and invalid
thresholds fail closed before anything is recorded; fractional per-generation
costs are summed per loop and rounded up to integer calls (#685).
