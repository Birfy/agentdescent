Fixed
`init`, `algorithms run` and the MCP `algorithm_spec` tool now back up a
replaced spec to numbered `<out>.bak-N` slots (like `apply`), so a second
re-run no longer overwrites the first backup; the OpenCode provider-only config
written next to the spec is backed up and named in a note the same way when it
held different content (#727).
