Added
`CampaignOps` + `CampaignOpsReport`: publish a sequence of finished
`LoopReport`s into the control room in one call, recording each into the
console, lineage, telemetry and QD and returning the campaign metrics, alerts
and snapshot. Thresholds and malformed loops fail closed before anything is
recorded (#685).
