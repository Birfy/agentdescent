# Design: AgentDescent as a plugin for Claude Code, Codex and other agents

> Status: **design proposal**, nothing here is implemented yet. It is written
> against the code as of `0.4.6` and names the exact module each piece lands in.

## 0. The one-paragraph version

Today AgentDescent is a **Python library**: you write a script, call
`evolve_skill_dir(...)`, wait, then `result.write_to(...)`. The goal is that a
person sitting *inside* Claude Code, Codex, Gemini CLI, OpenCode, Cursor, or any
future agent can say *"evolve this skill against these examples"* and the host
agent does it for them, with no script and no Python knowledge.

Every one of those hosts speaks two things: **MCP** (tools) and the
**Agent Skills standard** (`SKILL.md`). So the design is one shared core and thin
shells:

```
                 ┌──────────────────────────────────────────────┐
                 │  agentdescent (library, unchanged engine)    │
                 │  evolve / evolve_skill_dir / evolve_agent_*  │
                 └────────────────────┬─────────────────────────┘
                                      │ Recipe (JSON)  +  Run store (~/.agentdescent/runs)
                 ┌────────────────────┴─────────────────────────┐
                 │  agentdescent.cli        agentdescent.mcp     │   <- NEW, in this package
                 │  `agentdescent evolve …` `agentdescent-mcp`   │
                 └───┬──────────────┬──────────────┬────────────┘
                     │              │              │
        Claude Code plugin     Codex skill +    Gemini ext / OpenCode /
        (.claude-plugin)       config.toml      Cursor rules (+MCP)
        skill+cmd+mcp+hook     mcp_servers      all reuse the same SKILL.md
```

The **Recipe** is the new central abstraction: a declarative JSON/YAML
description of *what to evolve, against what data, scored how, by which agent*.
The host agent's job shrinks to *"write a recipe from the user's words, run it,
report the outcome, offer to apply"*. That is what makes the module able to
"evolve anything": the recipe is a serialisable view of the `evolve()` call, and
the engine already accepts every one of its fields.

## 1. Constraints that shape the design

These come from the code and from how the hosts behave; each one rules out an
obvious alternative.

| Constraint | Consequence |
|---|---|
| A run takes minutes to hours and costs real money. MCP tool calls time out in seconds to a few minutes. | Runs are **background jobs**. The MCP tool that starts a run returns a `run_id` immediately; separate tools poll and fetch. Never block a tool call on `evolve()`. |
| The host agent (e.g. Claude Code) is usually *also* the worker the run spawns (`claude -p`). | Nested agents must work. `runners._child_env` already strips the environment; the worker launcher must also drop host session markers (`CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, `CODEX_*`) so the child does not think it is inside a session, and must never inherit the host's MCP config, or the worker will call `agentdescent` recursively. |
| The engine's callables are closures and cannot be pickled (see `workspec.py`). | The recipe uses `Ref` semantics: named scorers (`SCORERS`), named agents (`claude_code`, `codex`, `openai_compatible`), dotted paths for user code. This is why `RolloutSpec`/`Ref` exist and the recipe should reuse them rather than invent a second scheme. |
| `write_to` overwrites the user's real skill directory. | The plugin **never** auto-applies. `evolve` produces a result under the run store; `apply` is a separate, explicit tool/command that backs up first (already `write_to(backup=True)`). The host agent must ask before calling it. |
| The core has zero dependencies and must stay that way. | `agentdescent.mcp` imports the MCP SDK lazily and is installed via an extra: `pip install "agentdescent[mcp]"`. The CLI uses only `argparse` and stays in the core. |
| Every host has its own plugin manifest format, and they change often. | Put **all logic** in the package; keep per-host directories to manifests plus a shared `SKILL.md`. A new host is a new manifest, not new code. |

## 2. The Recipe

A recipe is what a host agent writes from the user's request. It must be
small enough for an LLM to author correctly and complete enough to reproduce a
run.

```jsonc
{
  "version": 1,
  "kind": "skill_dir",                 // text | skill_dir | agent_dir | agent_code
  "target": "~/.claude/skills/pdf-audit",
  "name": "pdf-audit",                 // optional; defaults to basename

  "data": {                            // one of:
    "path": "./eval/cases.jsonl",      //   local json / jsonl / csv
    // "hf": {"dataset": "hotpotqa/hotpot_qa", "split": "validation", "config": "distractor", "limit": 40},
    // "inline": [{"prompt": "...", "gold": "..."}],
    "prompt": "question", "gold": "answer"
  },

  "score": "contains",                 // SCORERS name | {"cmd": "./grade.sh"} | {"ref": "pkg.mod:fn"}

  "agent":   {"ref": "claude_code", "extra_args": ["--permission-mode", "acceptEdits"]},
  "reflect": {"ref": "openai_compatible", "model": "deepseek-v4-flash"},   // optional

  "layout": "claude_skill",            // where the tree lands in each workspace (runners.LAYOUTS)
  "editable": ["**"], "frozen": ["references/policy.md"],

  "evolve": {                          // passes straight through to evolve()
    "rounds": 6, "n_workers": 4, "asynchronous": true,
    "target_reward": 0.98, "budget_usd": 5.0
  }
}
```

Design points:

* **`kind` selects the wrapper**, nothing else: `text → evolve_skill`,
  `skill_dir → evolve_skill_dir`, `agent_dir → evolve_agent_dir`,
  `agent_code → evolve_agent_code`. Governance (L1 vs L2 blast radius) follows
  from `kind` exactly as it does today, so a user cannot accidentally evolve a
  harness under skill rules.
* **`score` has three forms** because users have three kinds of graders: a
  built-in name, an arbitrary shell command (`{"cmd": ...}` gets the task as
  JSON on stdin and the answer as `$ANSWER`, prints a float), and a Python
  reference. The shell form is what lets someone evolve *anything* without
  touching Python: a linter, a compiler, a diff against a golden file, a
  web-check script are all one command.
* **`agent` / `reflect` are `Ref`s** resolved through the existing allowlist in
  `workspec.py`. Adding `agentdescent.agents.*` to the default allowed prefixes
  is the only change the resolver needs.
* **Secrets never enter the recipe.** Providers read keys from the environment on
  the worker side, as `SandboxSpec` does today. The recipe is written to the run
  store and shown to the user, so this matters.
* **A recipe is also a file** (`.agentdescent/recipes/<name>.json`), so a run is
  reproducible from the CLI and can be checked into the repo alongside the skill
  it evolves. The plugin encourages that: "evolve this again with more data" is
  then a one-line edit.

Implementation: `agentdescent/recipe.py` with `Recipe.from_dict / to_dict /
validate()` and `run_recipe(recipe, *, run_dir, on_round) -> EvolutionResult`.
This module is the **only** place the mapping from JSON to `evolve*()` kwargs
lives; CLI and MCP both call it.

## 3. The run store and background execution

```
~/.agentdescent/runs/<run_id>/
    recipe.json        what was asked
    status.json        {"state": "running|done|failed|cancelled", "round": 3, "rounds": 6,
                        "best_reward": 0.81, "usd": 1.72, "pid": 41022, "started": ..., "updated": ...}
    rounds.jsonl       one RoundInfo per line, appended by the on_round hook
    result.json        EvolutionResult.save() on completion
    tree/              the evolved directory, materialised (for skill_dir/agent_*)
    ledger/            the git ledger (repo_path=), which is also what makes resume work
    log.txt            stderr of the run
```

* The run is a **detached subprocess**: `python -m agentdescent.cli run
  --run-dir <dir>`. Detaching (not a thread) is what survives the MCP server
  being restarted by the host, which Claude Code does on `/mcp` reconnects and
  on session end. `status.json` is written atomically (`os.replace`) after every
  round from the `on_round` callback.
* **Resume for free**: `evolve(repo_path=<run_dir>/ledger)` already resumes a
  ledger, so `agentdescent resume <run_id>` re-launches the same recipe on the
  same ledger after a crash or a cancel.
* **Cancel** sends SIGTERM to the process group; `runners._sh` already uses
  `start_new_session`, so the worker agents die with it and no `claude -p`
  orphans are left behind.
* **Budget** is enforced inside the run: `budget_usd` maps to the existing
  `Usage` accounting, and the run stops itself with `stop_reason="budget"`.
  This is the safety rail the host agent points to when it asks "how much may I
  spend?".

## 4. The CLI (`agentdescent`)

A console script (`[project.scripts] agentdescent = "agentdescent.cli:main"`)
that exposes exactly the verbs the MCP server exposes, so a user can do by hand
anything the agent can do, and the SKILL.md can fall back to shell when a host
has no MCP.

```
agentdescent init      <path> [--kind skill_dir] [--data cases.jsonl]  # write a starter recipe
agentdescent evolve    <recipe.json> [--detach] [--budget 5]           # start a run
agentdescent status    [<run_id>]                                     # one run or all
agentdescent watch     <run_id>                                       # tail rounds.jsonl
agentdescent show      <run_id> [--diff]                              # evolved tree, diff vs original
agentdescent apply     <run_id> [--to <path>] [--no-backup]           # write_to, backs up first
agentdescent cancel    <run_id>
agentdescent resume    <run_id>
agentdescent doctor                                                   # which agents/keys are available
agentdescent mcp                                                      # run the MCP server (stdio)
```

`doctor` matters more than it looks: the most common failure will be "the
worker agent is not on PATH" or "no API key for the reflector", and the host
agent should run this first and tell the user what is missing rather than start
a run that fails on round one.

## 5. The MCP server (`agentdescent.mcp`)

Stdio server, launched as `agentdescent mcp`. Tools mirror the CLI one-to-one;
descriptions are written for the *calling model*, since that is who reads them.

| Tool | Returns | Notes |
|---|---|---|
| `agentdescent_doctor()` | available agents, providers, missing keys | call first |
| `agentdescent_plan(kind, target, data, score, agent?, ...)` | a validated recipe + cost estimate (`rollouts × rounds`, `Usage` rates) | **does not run**; the host shows it to the user |
| `agentdescent_start(recipe, budget_usd)` | `run_id` | detaches; returns in < 1 s |
| `agentdescent_status(run_id?)` | `status.json` (+ last 3 rounds) | cheap; safe to poll |
| `agentdescent_show(run_id, diff=true)` | evolved tree, unified diff vs original, `outcomes()` | what the user reads before deciding |
| `agentdescent_apply(run_id, to?)` | files written, backup path | **destructive**; host must confirm with the user |
| `agentdescent_cancel(run_id)` / `agentdescent_resume(run_id)` | new status | |

Two resources, for hosts that support them: `agentdescent://runs` (list) and
`agentdescent://runs/{id}/rounds` (progress), so a host can render progress
without the model spending tokens on polling.

Why `plan` is separate from `start`: it forces the "show the recipe and the
price, then run" beat into the protocol itself. A skill can *ask* the model to
do that; a two-step tool surface *makes* it do that.

## 6. The shared skill (`SKILL.md`)

The same file ships in every host package. It is what teaches the host model
*when* and *how* to use the tools. Its shape:

```
---
name: agentdescent
description: Evolve a skill, agent definition, prompt or small codebase against
  examples using AgentDescent's parallel merge-based optimiser. Use when the user
  wants to improve, tune, optimise or "train" a SKILL.md, an agent folder, a
  system prompt, or code that has a test/score, and has (or can write) examples
  with expected answers.
---
1. Run agentdescent_doctor. Report anything missing; stop if no worker agent.
2. Establish the four things a recipe needs: target, data, score, agent.
   - No data? Offer to draft 8-20 cases into eval/cases.jsonl and have the user
     check them. Never evolve against data the user has not seen.
   - No obvious score? Prefer `contains`/`exact`; offer a `cmd:` grader when
     the answer is a file, code, or a format check.
3. Call agentdescent_plan; show the recipe and the cost estimate; get a yes.
4. agentdescent_start; then poll agentdescent_status about once per round,
   not more. Summarise round deltas, not raw JSON.
5. When done: agentdescent_show with diff=true. Explain *what changed and why*
   using result.outcomes(). Do not paste the whole tree.
6. Ask before agentdescent_apply. Mention the backup path afterwards.
Guardrails: never edit the target directory yourself during a run; never raise
budget without asking; if the host is itself the worker, warn about cost
scaling (rounds × n_workers × tasks agent calls).
```

The skill is host-agnostic on purpose. If a host has no MCP, the same steps run
through `agentdescent <verb>` shell commands; the SKILL.md carries both forms.

## 7. Per-host packaging

All per-host material lives under `integrations/` in this repo and is generated
from one source of truth (the `SKILL.md`, the tool list, the server command) by a
small script so they cannot drift.

### 7.1 Claude Code plugin (`integrations/claude-code/`)

```
.claude-plugin/plugin.json      name, version, description
.claude-plugin/marketplace.json in the repo root so `/plugin marketplace add Birfy/agentdescent` works
skills/agentdescent/SKILL.md    the shared skill
commands/evolve.md              /agentdescent:evolve <path> — a slash-command shortcut that
                                invokes the skill with the argument prefilled
.mcp.json                       {"mcpServers": {"agentdescent": {"command": "agentdescent", "args": ["mcp"]}}}
hooks/hooks.json                SessionStart: `agentdescent status --brief` so an in-progress run
                                is surfaced when the user comes back
agents/evolution-reviewer.md    optional subagent that reads `show --diff` output and writes
                                the plain-language "what changed" summary
```

The worker inside the run is `claude -p` with `--permission-mode acceptEdits`
and `--strict-mcp-config` plus an empty MCP config, so the worker does *not*
see the agentdescent server (no recursion) or the user's other servers (no
surprise side effects).

### 7.2 Codex (`integrations/codex/`)

Codex reads skills from `~/.codex/skills/<name>/SKILL.md` (and project
`.agents/skills/`) and MCP servers from `~/.codex/config.toml`:

```toml
[mcp_servers.agentdescent]
command = "agentdescent"
args = ["mcp"]
```

Install is `agentdescent install codex`, which copies the skill and appends
the server block if absent. The worker is `codex exec --full-auto` in the
workspace. Codex has no hooks; the "surface in-progress runs" behaviour comes
from a line in `AGENTS.md` telling the model to run `agentdescent status --brief`
on start.

### 7.3 Other hosts

| Host | Skill location | MCP config | Worker command |
|---|---|---|---|
| Gemini CLI | `gemini-extension.json` + `GEMINI.md` + `skills/` | in the extension manifest | `gemini -p` |
| OpenCode | `.opencode/skills/` | `opencode.json → mcp` | `opencode run` |
| Cursor | `.cursor/rules/agentdescent.mdc` (rule wraps the skill text) | `.cursor/mcp.json` | not a worker; use `claude_code`/`codex`/API |

`agentdescent install <host>` is the single entry point for all of them; each
host is ~30 lines in `agentdescent/integrations/<host>.py` that knows the paths
and the manifest shape. Adding a host does not touch the core.

**Note on "DSH":** the request named DSH alongside Codex and Claude Code. I
could not identify which product this is (OpenCode? Gemini CLI? a typo?). The
table above covers the common hosts; if DSH is another MCP-speaking agent, it
slots in as one more `install` target with the same `SKILL.md`.

## 8. Worker adapters: what the run spawns

`agents.claude_code()` and `agents.codex()` exist. Two additions:

* **Structured output.** `claude -p --output-format json` yields the answer and
  the exact token/cost figures; parsing it feeds `Usage` with real numbers
  instead of wall-clock estimates. Same for `codex exec --json`. Add
  `claude_code(structured=True)` and read `total_cost_usd`; this is what makes
  `budget_usd` honest.
* **Session hygiene.** A `_worker_env()` in `agents.py` that drops
  `CLAUDECODE`, `CLAUDE_CODE_*`, `CODEX_*`, `OPENAI_AGENT_*`, and points
  `CLAUDE_CONFIG_DIR`/`CODEX_HOME` at an empty directory inside the workspace,
  so the worker starts clean and cannot read the user's real config, memory or
  MCP servers.

## 9. Security and cost posture

* **Two trust boundaries, kept apart.** The host session (trusted, has the
  user's keys and config) starts a run. Each worker (semi-trusted, model-driven,
  edits files) runs in a leased workspace with a stripped environment and its own
  config dir. Candidate *code* (`agent_code` kind) runs with `runners._child_env`
  as today. Nothing here weakens what the library already does.
* **Applying is explicit and backed up.** Same as today; the plugin just makes
  it a separate verb with a confirmation step in the skill.
* **Cost is visible before, during, after.** `plan` estimates, `status` reports
  spend so far, `budget_usd` stops the run. The estimate formula is honest about
  what it does not know (per-call token counts for a tool-using agent), and says
  so.
* **The recipe is the audit log.** Everything a run did is derivable from
  `recipe.json` + the ledger; nothing hides in a closure.

## 10. Delivery plan

Ordered so that each step is usable on its own and testable offline (the
`echo()` agent and the router domain make every step runnable without a key).

| Step | Lands in | Tests |
|---|---|---|
| 1. `recipe.py`: schema, validation, `run_recipe` mapping to the four wrappers | core | round-trip every field; `kind` picks the right blast radius |
| 2. Run store + detached runner + `status.json` from `on_round` | `agentdescent/runstore.py`, `cli.py` | start with `echo()` agent, poll, cancel, resume |
| 3. CLI verbs (`init/evolve/status/show/apply/cancel/resume/doctor`) | `cli.py`, `[project.scripts]` | `subprocess` tests against the offline domain |
| 4. MCP server (`[mcp]` extra) mirroring the CLI | `agentdescent/mcp.py` | tool schema snapshot; start→status→show with `echo()` |
| 5. Shared `SKILL.md` + `integrations/claude-code` plugin + marketplace.json | `integrations/` | plugin manifest validates; `/agentdescent:evolve` reaches the skill |
| 6. `install codex` (+ others); structured-output worker adapters | `agentdescent/integrations/`, `agents.py` | parse fixtures of `claude -p --output-format json` / `codex exec --json` |
| 7. Docs page `docs/plugins.md`; README section "Use it from your agent" | docs | `test_docs_links` |

Steps 1 to 4 are the product; 5 to 7 are packaging. Nothing in the aggregator,
ledger, async runtime or governance changes.

## 11. Alternatives considered

* **Python-only, "just import it".** That is the status quo, and it is why the
  question is being asked: the audience inside an agent session does not want to
  write Python.
* **MCP only, no CLI.** Loses reproducibility and debuggability. The detached
  run process needs an entry point anyway; making it the CLI costs nothing.
* **A per-host plugin with its own logic.** Fastest to demo, slowest to keep
  alive: every host revs its manifest format yearly. Logic in the package,
  manifests in `integrations/`.
* **Blocking `evolve` tool call.** Fails on the first real run: hosts time tool
  calls out and restart servers between sessions. Background jobs are not
  optional.
* **Auto-apply on success.** Tempting for a one-shot feel, but the artifact is
  the user's own skill directory and the reward is a proxy. Show the diff, ask.
