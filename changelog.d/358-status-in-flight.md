Fixed
`status` (CLI and MCP) now shows `pending_calls` and `phase` while an actor
(`propose` / `run`) is in flight, so a multi-minute first openai_compatible
propose no longer looks hung at `calls=0` (#358).
