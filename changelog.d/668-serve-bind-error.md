Fixed
`agentdescent serve` on a busy, privileged or out-of-range `--port` now prints
one `cannot listen on HOST:PORT -- <reason>` line and exits 4, like
`agentdescent mcp --transport streamable-http`, instead of announcing the URL
and then crashing with a traceback; under `--json` it is one
`{"ok": false, "error"}` object on stdout. The banner now names the port
actually bound, so `--port 0` reports the real one (#668).
