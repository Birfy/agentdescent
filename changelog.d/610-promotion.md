Added
`promote()` + `PromotionRules`: one fail-closed promotion decision over
evaluation, contamination, conformal/sequential evidence, canary, proof,
admission, risk and human approval. Missing evidence returns `defer`, a failed
gate returns `reject`, a canary regression returns `rollback`, and only a fully
green bundle returns `promote` with a stable evidence hash for manifest/lineage.
