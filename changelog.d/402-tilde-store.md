Fixed
A literal `~` in `--store` or `$AGENTDESCENT_HOME` is now expanded everywhere
(#402). It arrives unexpanded from an MCP host config's `args`/`env` or a quoted
`--store '~/st'`; `create` already landed the run under `$HOME` (via `RunDir`,
#400) but `status`/`list_runs` looked in `./~/st`, said "no runs", and left a
stray `~` directory in the cwd, and `AGENTDESCENT_HOME=~/x` put every run under
`./~/x`. The store base is now expanded and made absolute once
(`runstore.store_path`), so the CLI, MCP `status`'s `store`, and `doctor`'s
`run_store` all report the same absolute path.
