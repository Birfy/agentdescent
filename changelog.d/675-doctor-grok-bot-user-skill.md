Fixed
`agentdescent doctor` now checks the skill Grok Bot actually loads: a saved user
skill named `agentdescent` under `$AGENTDESCENT_GROK_BOT_SKILLS` (default
`<home>/agent-data/workflows`) is reported as `user skill ok|stale|missing`, a
stale one is a problem with a rewrite hint, and `--json` gains
`hosts.grok-bot.user_skill` / `user_skill_path`. The `host grok-bot` line no
longer says `cli not on PATH` (#675).
