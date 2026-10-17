Fixed
`status` (CLI and MCP) now shows `pending_calls` and `phase` while an actor
(`propose` / `run`) is in flight, so a multi-minute first openai_compatible
propose no longer looks hung at `calls=0` (#358). In-flight state is paired
by token (`Usage.begin` / `Usage.end`), so a provider's own retry records on
a shared `Usage` do not end the actor call, `phase` reflects the calls still
pending, and concurrent status rewrites are serialized per run.
