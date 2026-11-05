Fixed
`kind: text` now refuses an existing directory as its target: `init`,
`algorithms run`, `plan`, `evolve` and MCP `start` stop with "is a directory"
and point at a file inside it or `--kind skill_dir`, instead of evolving the
path string as the prompt; `apply` / `apply --dry-run` onto a directory give
the same error instead of claiming a write and then tracing back on
`IsADirectoryError` (#755).
