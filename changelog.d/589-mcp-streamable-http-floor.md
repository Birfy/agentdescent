Fixed
`agentdescent mcp --transport streamable-http` on mcp 1.2–1.7 now exits with a
one-line "needs mcp>=1.8" message instead of the SDK's opaque Unknown transport
error; CLI host/port defaults stay locked to `mcp.DEFAULT_HTTP_*`, and the
`.mcp` import stays lazy so a shadowed cwd still prints one line (#589).
