Added
`agentdescent mcp --transport streamable-http [--host 127.0.0.1] [--port 8000]`
serves the MCP tools over HTTP at `/mcp`, so URL-only hosts such as Grok Bot can
connect through an HTTPS tunnel; stdio stays the default. A non-loopback
`--host` is refused unless `--allow-remote` is passed. The Grok Bot `HOST.md`
and plugin docs now give the copy-paste commands (#560).
