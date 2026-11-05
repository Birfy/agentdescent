Added
`agentdescent traces digest|show` and the MCP `traces_digest` tool: turn raw
agent trajectory JSONL/JSON into a layered, sourced evidence digest (per-run and
per-task counts, normalised failure signatures, and a token-bounded prompt
block), then drill back into the exact source events by signature id. Failure
messages are normalised so repeated causes group, secret-shaped values are
redacted, malformed input is an input error, and no model or network call is
made.
