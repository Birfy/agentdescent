---
description: Evolve a skill, agent, prompt, codebase or plugin against examples with AgentDescent
argument-hint: <path> [cases.jsonl]
---

Use the agentdescent skill. The target is `$ARGUMENTS` (a path, optionally
followed by a cases file). Run `doctor`, build the spec, `plan` it, show me the
spec and the estimate, and wait for my yes before `start`.

If I named a method that is in `agentdescent algorithms list` (for example RRSI
or archive_selection), set `algorithm` in the spec (or pass `--algorithm` to
`init`) with its parameters and show me the resolved card and cost in the plan;
if I did not, use the kind's default recipe.
