---
description: List, show, or run a named AgentDescent algorithm (same cards as algorithms list/show)
argument-hint: "[list [--kind K] | show <name> | run <name> <path> [cases.jsonl] [--param k=v...]]"
---

Use the agentdescent skill / MCP tools. `$ARGUMENTS` chooses the verb; the data
is the same as `agentdescent algorithms list` / `show` and the MCP tools
`list_algorithms` / `describe_algorithm`. Never invent a name.

Parse `$ARGUMENTS` (trim; empty means `list`):

- **`list`** [optional `--kind K`]: call `list_algorithms` (or
  `agentdescent algorithms list [--kind K] [--json]`). Show each name, one-line
  summary, kinds, and fidelity. If the user named a kind, filter to it.
- **`show <name>`**: call `describe_algorithm` (or
  `agentdescent algorithms show <name>`). Show the full card (params, defaults,
  extra cost, what it changes). Stop here unless they ask to run it.
- **`run <name> <path>`** [optional `cases.jsonl`] [optional `--param k=v`]:
  build a spec for the path with `algorithm` set to that name (and only the
  params they gave — validate with `describe_algorithm` / `show` first). Run
  `doctor`, `plan`, show the resolved algorithm card and estimate, and wait for
  my yes before `start`. Same apply/cancel/resume rules as `/evolve`.

If they only named a method with no path, `show` it and ask whether to `run`.
