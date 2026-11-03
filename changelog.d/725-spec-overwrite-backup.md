Fixed
`init`, `algorithms run` and the MCP `algorithm_spec` tool no longer replace an
existing, different spec at the output path without a word: the old copy is
saved as a `<out>.bak-N` backup and a note names it. An identical re-run writes no backup
(#725).
