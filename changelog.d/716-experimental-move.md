Changed
Moved the research prototypes `society_runtime`, `openended`, `scaling`,
`calibrated_deferral` and `deferral` under `agentdescent/experimental/`, so the
top-level package no longer presents them as part of the main engine. Their old
import paths remain compatibility shims (`from agentdescent.experimental.x
import *`), so existing code and tests keep working. No CLI/MCP verb or
top-level export changed.
