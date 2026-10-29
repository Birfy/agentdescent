Added
`CapabilityFoundry`: the admission loop for a synthesized tool. It binds
validation score, debate/review agreement, `safe_selfmod` capability/canary
checks, proof certificate verification, holdout gain and artifact-graph
registration into one fail-closed outcome (`promoted` / `deferred` / `rejected`
/ `rolled_back`). Missing evidence defers; failed gates reject; a measured
canary regression rolls back.
