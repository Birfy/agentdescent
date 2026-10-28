Fixed
`agentdescent mcp --transport streamable-http --allow-remote` now serves
non-loopback clients on mcp 1.x too: the SDK pins allowed Host headers to
loopback when the server is built, so a remote bind answered every request 421;
the pin is now dropped for a non-loopback host (#587).
