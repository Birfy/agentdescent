Fixed
The MCP `algorithm_spec` tool now passes its `params` object straight through
instead of re-encoding it as `--param KEY=VALUE` strings, so a key containing
`=` is no longer split into a different key and value, and an empty or
repeated key is reported as a `params` key rather than a CLI flag (#729).
