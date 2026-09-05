---
name: agentdescent
description: Evolve a skill, agent definition, prompt, small codebase or host plugin against examples with AgentDescent's parallel, merge-based optimiser. Use when the user wants to improve, tune, optimise or "train" a SKILL.md, an agent folder, a system prompt, agent code with a test, or a plugin, and has (or can write) examples with expected answers.
---

# AgentDescent

You have tools (MCP server `agentdescent`) or, without MCP, the `agentdescent`
command with the same verbs. A run is an evolution: N workers propose edits in
parallel, a merger keeps the ones that improve held-out reward, and nothing is
written back until the user says so.

## The procedure

1. **`doctor` first.** Report what is missing (worker agent CLI, provider key,
   container engine). Stop if there is no worker agent for a directory kind.
2. **Establish the four things a spec needs**: `target`, `data`, `score`, `agent`.
   - `kind`: `text` (a prompt or instruction), `skill_dir` (a SKILL.md folder),
     `agent_dir` (subagent definitions), `agent_code` (a tree that runs behind
     tests), `plugin` (a host plugin; needs `host`).
   - No data? Offer to draft 8 to 20 cases into `eval/cases.jsonl`
     (`{"prompt": ..., "gold": ...}` per line) and have the user check them.
     Never evolve against data the user has not seen.
   - No obvious score? Prefer `"contains"` or `"exact"`; offer
     `{"cmd": "./grade.sh"}` when the answer is a file, code, or a format check
     (task JSON on stdin, `$ANSWER` in the env, a number in [0, 1] on stdout).
   - Leave `policies` empty unless the user asks for a mechanism by name; the
     empty bundle is the shipped run.
3. **`plan`** with the spec. Show the user the spec and the estimate (agent calls
   per round and in total; dollars only if a per-call price is known). Get a
   yes. Fix any error it names; it names the field.
4. **`start`**. Then poll **`status`** about once per round, not more. Summarise
   round deltas (reward, commits, refusal reasons), not raw JSON.
5. When done, **`show`** with `diff=true`. Explain what changed and why using
   the `outcomes` histogram (`committed`, `below-threshold`, `oracle-rejected`
   ...). Do not paste the whole tree.
6. **Ask before `apply`.** It overwrites the target; it backs up first. Tell the
   user the backup path afterwards.

If the user wants to stop a run, or one is going badly (cost climbing, reward
flat for several rounds), use **`cancel`** — it stops the run and every worker
it started, and keeps the ledger. **`resume`** continues a cancelled, failed or
stopped run from where it left off. Say what a cancel will cost them (the
rounds already committed are kept).

## A spec

```json
{
  "kind": "skill_dir",
  "target": "~/.claude/skills/pdf-audit",
  "data": {"path": "eval/cases.jsonl", "prompt": "prompt", "gold": "gold"},
  "score": "contains",
  "agent": {"ref": "claude_code", "extra_args": ["--permission-mode", "acceptEdits"]},
  "reflect": {"ref": "openai_compatible", "model": "deepseek-v4-flash"},
  "evolve": {"rounds": 6, "n_workers": 4}
}
```

Agents by short name: `claude_code`, `codex`, `dsh`, `openai_compatible`,
`claude`. A cheap `reflect` model behind an expensive `agent` is the usual
trade. For `kind: plugin`, set `host` to `dsh`, `claude_code` or `codex`.

## Guardrails

- Never edit the target directory yourself while a run is in progress.
- Never raise `budget`, `rounds` or `n_workers` without asking.
- Cost scales as rounds x n_workers x tasks agent calls; say so when the host
  is itself the worker.
- If `start` returns `nested: true`, this session is a worker inside another
  run: report that and do not retry.

## Without MCP

```
agentdescent doctor
agentdescent plan   spec.json
agentdescent evolve spec.json --detach
agentdescent status <run_id>
agentdescent show   <run_id>
agentdescent apply  <run_id> --dry-run
```
