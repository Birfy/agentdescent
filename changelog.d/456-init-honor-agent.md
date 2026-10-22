Fixed
`agentdescent init --agent <ref>` is no longer silently discarded for
`kind: text` and `kind: policy_slot`: an explicit `--agent host_model` (or any
other plain model ref) is written as given instead of being replaced by
`openai_compatible` + `deepseek-v4-flash`. When the ref cannot work for the kind,
`init` now refuses with exit 2 and writes nothing, using the same rules `plan`
enforces: `kind: text` refuses file-editing agents (`claude_code`, `codex`,
`dsh`, `opencode`, `cli_agent`), and `skill_dir` / `agent_dir` refuse plain model
calls (`host_model`, `openai_compatible`, `claude`, `echo`). Omitting `--agent`
keeps the old defaults.
