Added
`agentdescent experience cycle` and the MCP `experience_cycle` tool: one
guarded adaptive-experience cycle that ingests new `ExperienceRecord`s, chooses
validation/anchors through the preference/recall/history cascade, and delegates
the update and forgetting guard to `ExperienceEngine`. It makes the previously
isolated `AdaptiveExperience` reachable from the product surface; the result
carries the plan, ingested ids and update decision, with `schema_version: 1`.
The CLI exits 1 when the update is not accepted; MCP `store` writes use
`AGENTDESCENT_SURFACE_WRITE_DIRS`.
