Fixed
`init` / `algorithms run`: an unknown `--agent` (for example the install host
name `claude-code`) is refused before the spec is written, listing only the
agent refs and suggesting `claude_code`, instead of writing a spec `plan`
rejects with a list of policy classes (#745).
