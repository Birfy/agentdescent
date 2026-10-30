Added
`ExperienceStore` + `ExperienceEngine`: append-only cross-run experience
memory with nearest-neighbour recall and a continual meta-policy update guarded
by a frozen anchor slice. The engine warms its learner replay from an existing
store on restart; a candidate that improves validation but regresses the
anchors is rolled back, and `ContinualMetaLearner.rollback()` restores the
weights while keeping the attempt in history as a rejected update.
