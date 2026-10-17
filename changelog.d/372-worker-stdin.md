Fixed
A worker CLI no longer inherits the caller's stdin (#372). `cli_agent()` ran a
worker with `input=None` unless `via_stdin` was set, and `runners._sh` started
gate commands and plugin hosts with no `stdin=` at all, so under an open pipe
that never reaches EOF (a CI step, `… | agentdescent evolve spec.json`, a
notebook kernel, a tool runner) `opencode run "<prompt>"` waited on the pipe and
every call burned the whole agent `timeout` before ending as `AgentError:
... exceeded timeout`. A worker that gets its prompt as an argument now gets
`stdin=DEVNULL`, as do `runners._sh` commands; `via_stdin=True` still delivers
the prompt on stdin.
