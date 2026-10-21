Fixed
`agentdescent demo` now writes a valid Agent Skill: the generated
`csv-total/SKILL.md` carries `name` / `description` frontmatter, so a real run on
the demo skill keeps the SKILL.md structural gate on (no more `no-frontmatter`
note from `plan` / `evolve --detach` / `status`) and hosts such as OpenCode,
Claude Code and Codex actually load the skill. The offline agent and reflector
are unchanged.
