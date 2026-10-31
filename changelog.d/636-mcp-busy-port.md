Fixed
`agentdescent mcp --transport streamable-http` now checks that its host:port
can be bound before announcing the URL: a busy port prints one line naming the
port and suggesting another `--port`, and exits 4 instead of failing inside
uvicorn's logs with the same exit code 3 as a missing MCP SDK. The Grok Bot
`HOST.md` scaffold and `docs/plugins.md` mention it (#636).
