Fixed
`agentdescent init` without `--agent` now writes the worker CLI that is on
PATH (first of `claude_code`, `codex`, `dsh`, `opencode`, as `doctor` sees
them) instead of always `claude_code`, so the quickstart `init` -> `plan` path
works on an OpenCode-only machine. With no worker CLI on PATH it keeps the
`claude_code` placeholder and prints a warning pointing at `doctor` /
`--agent`; `--agent` still overrides, and only `claude_code` gets
`--permission-mode acceptEdits` (#361).
