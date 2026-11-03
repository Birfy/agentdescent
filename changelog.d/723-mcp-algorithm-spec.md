Added
MCP `algorithm_spec` tool: the MCP form of `agentdescent algorithms run NAME
PATH [cases]`. It runs the same checks, writes the starter spec beside the
target (never inside a directory target), plans it, and returns the estimate
plus the absolute `spec` dict to pass to `start`, so an MCP-only host no longer
hand-assembles a spec to try a named algorithm. `algorithms run` and the tool
share one code path (#723).
