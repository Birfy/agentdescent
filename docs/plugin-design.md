# Design: AgentDescent as a plugin for DeepSeek Harness, Claude Code, Codex and other agents

> Status: **design record; steps 1 to 9 of the delivery plan are implemented**
> (`evolvespec.py`, `runstore.py`, `cli.py`, `mcp.py`, `integrations/`,
> `plugin_runner`, `stop_when`). The user-facing guide is
> [Use it from your agent](plugins.md); this page keeps the reasoning. Written
> against `main` after the one-call wrappers were removed (`evolve()` is the
> only entry point) and the policy install hooks landed. Section 14 lists what
> changed since the first draft and why.

## 0. The one-paragraph version

Today AgentDescent is a **Python library**: you write a dozen lines that hand
public building blocks to `evolve()`, wait, then `result.write_to(...)`. The goal
is that a person sitting *inside* DeepSeek Harness (DSH), Claude Code, Codex,
Gemini CLI, OpenCode, Cursor, or any future agent can say *"evolve this skill
against these examples"* and the host agent does it for them, with no script and
no Python knowledge.

Every one of those hosts speaks two things: **MCP** (tools) and the
**Agent Skills standard** (`SKILL.md`). So the design is one shared core and thin
shells:

```
                 ┌──────────────────────────────────────────────┐
                 │  agentdescent (library, unchanged engine)    │
                 │  evolve() + the public building blocks       │
                 └────────────────────┬─────────────────────────┘
                                      │ EvolveSpec (JSON)  +  Run store (~/.agentdescent/runs)
                 ┌────────────────────┴─────────────────────────┐
                 │  agentdescent.cli        agentdescent.mcp     │   <- NEW, in this package
                 │  `agentdescent evolve …` `agentdescent mcp`   │
                 └───┬──────────────┬──────────────┬────────────┘
                     │              │              │
        DSH: ~/.dsh/skills +   Claude Code      Codex skill +   Gemini / OpenCode /
        cordis.patch.yml       plugin           config.toml     Cursor (+MCP)
        (mcp-client, hooks)    (.claude-plugin) mcp_servers     all reuse the same SKILL.md
```

The **EvolveSpec** is the new central abstraction: a declarative JSON description
of *what to evolve, against what data, scored how, by which agent, under which
policies*. It is an `evolve()` call as data, in the same spirit as
`RolloutSpec` in `workspec.py` is a rollout as data. The host agent's job shrinks
to *"write a spec from the user's words, run it, report the outcome, offer to
apply"*. That is what makes the module able to "evolve anything": every field
of the spec is an ordinary `evolve()` argument or a public building block, and
the spec is the only place the two are joined.

## 1. Constraints that shape the design

These come from the code and from how the hosts behave; each one rules out an
obvious alternative.

| Constraint | Consequence |
|---|---|
| **Upstream removed the one-call wrappers.** `evolve_skill`, `evolve_skill_dir`, `evolve_agent_dir`, `evolve_agent_code` are gone; `evolve()` is the only entry point and the quickstarts show the full call. | The plugin must not reintroduce them as Python. The spec is a **serialisation boundary for non-Python callers**, and `evolvespec.py` is the one place that composes the public blocks into an `evolve()` call. Python users never see it; the quickstarts stay the template and the spec's composition table (section 2) must match them line for line. |
| A run takes minutes to hours and costs real money. MCP tool calls time out in seconds to a few minutes. | Runs are **background jobs**. The MCP tool that starts a run returns a `run_id` immediately; separate tools poll and fetch. Never block a tool call on `evolve()`. |
| The host agent (e.g. Claude Code, DSH) is usually *also* the worker the run spawns (`claude -p`, `dsh --profile headless`). | Nested agents must work. `runners._child_env` already strips the environment for candidate code; the worker launcher must also drop host session markers (`CLAUDECODE`, `CLAUDE_CODE_*`, `CODEX_*`, `DSH_*`) so the child does not think it is inside a session, and must never inherit the host's MCP config, or the worker will call `agentdescent` recursively. |
| The engine's callables are closures and cannot be pickled (see `workspec.py`). | The spec uses `Ref` semantics: named scorers (`rewards.SCORERS`), named agents (`claude_code`, `codex`, `dsh`, `openai_compatible`), named policies (`Beam`, `AdvantageAcceptance`, ...), dotted paths for user code. `RolloutSpec`/`Ref` exist for exactly this; the spec reuses them rather than inventing a second scheme. |
| Merge-side policies now take the verifier and thresholds through `bind`/`configure` when installed, and wrappers default their `inner`. | Policies are **JSON-configurable with no glue**: `{"ref": "AdvantageAcceptance", "strength": 1.0}` resolves to an object the aggregator installs itself. Without the hooks the spec would have had to construct verifiers it cannot reach. The composition rules (`Policies.require_supported`, the one-pair rule, factory exclusivity) are enforced by the engine, so the spec validates by building the `Policies` bundle and letting the engine refuse. |
| `write_to` overwrites the user's real skill directory. | The plugin **never** auto-applies. A run leaves its result in the run store; `apply` is a separate, explicit tool that backs up first (`write_to(backup=True)`), and `write_to(dry_run=True)` gives the plan to show before asking. |
| The core has zero dependencies and must stay that way. | `agentdescent.mcp` imports the MCP SDK lazily and is installed via an extra: `pip install "agentdescent[mcp]"`. The CLI uses only `argparse` and stays in the core. |
| Every host has its own plugin manifest format, and they change often (DSH says so explicitly). | Put **all logic** in the package; keep per-host directories to manifests plus a shared `SKILL.md`. A new host is a new manifest, not new code. |

## 2. The EvolveSpec

A spec is what a host agent writes from the user's request. It must be small
enough for an LLM to author correctly and complete enough to reproduce a run.

```jsonc
{
  "version": 1,
  "kind": "skill_dir",                 // text | skill_dir | agent_dir | agent_code | plugin
  // "host": "dsh",                    // plugin kind only: dsh | claude_code | codex
  "target": "~/.claude/skills/pdf-audit",
  "name": "pdf-audit",                 // artifact_id; defaults to basename

  "data": {                            // one of:
    "path": "./eval/cases.jsonl",      //   local json / jsonl / csv
    // "hf": {"dataset": "hotpotqa/hotpot_qa", "split": "validation", "config": "distractor", "limit": 40},
    // "inline": [{"prompt": "...", "gold": "..."}],
    "prompt": "question", "gold": "answer"          // -> tasks_from(rows, prompt=, gold=)
  },

  "score": "contains",                 // SCORERS name | {"cmd": "./grade.sh"} | {"ref": "pkg.mod:fn"}

  "agent":   {"ref": "claude_code", "extra_args": ["--permission-mode", "acceptEdits"]},
  "reflect": {"ref": "openai_compatible", "model": "deepseek-v4-flash"},   // optional; defaults to agent

  "layout": "claude_skill",            // runners.LAYOUTS key or literal prefix
  "editable": ["**"], "frozen": ["references/policy.md"],
  "max_files_per_diff": 2,

  "policies": {                        // optional; each slot is a Ref, installed by the engine
    "selection":    {"ref": "Beam", "k": 4},
    "task_sampler": {"ref": "DifficultyWeighted"},
    "acceptance":   {"ref": "AdvantageAcceptance", "strength": 1.0},
    "conflict":     {"ref": "AdvantageConflict", "margin": 0.5},
    "staleness":    "guarded"
  },
  "agg_config": {"base_delta": 0.5},   // thresholds are config, not policies (policy-guide §7)

  "evolve": {                          // passes straight through to evolve()
    "rounds": 6, "n_workers": 4, "asynchronous": true,
    "target_reward": 0.98, "patience": 3, "held_out_frac": 0.3,
    "max_seconds": 3600, "max_calls": 400
  }
}
```

### 2.1 What `kind` composes

`kind` is the only field that is not a plain `evolve()` argument. It selects one
row of this table, and each row is exactly the block the corresponding
quickstart shows, so `tests/test_dataset_to_skill.py` and
`tests/test_dir_evolution.py` are the reference the composition is tested against.

| `kind` | strategy | `run=` | `propose=` | `reward` | `blast_radius` | extra defaults |
|---|---|---|---|---|---|---|
| `text` | `SingleSlot(initial_value=<target file or string>)` | `lambda skill, task: model(template.format(skill, prompt))` | `reflector(reflect)` | `scorer(score)` | `SKILL_BLAST_RADIUS` | none |
| `skill_dir` | `FileTree(load_tree(target), editable, frozen, max_files_per_diff)` | `tree_runner(agent, layout, name, overlay=frozen_files)` | `tree_reflector(reflect, strategy)` | `scorer(score)` | `SKILL_BLAST_RADIUS` | `self_verify=False`, `cheap_eval_tasks=4` |
| `agent_dir` | same | `tree_runner(..., layout="claude_agent")` | same | same | `HARNESS_BLAST_RADIUS` | same |
| `agent_code` | `FileTree(..., frozen=["tests/**","conftest.py"] + frozen)` | `code_runner(entrypoint, test_cmd, setup_cmd, overlay=frozen_files)` | `tree_reflector(..., context_files=("**/*.py",))` | `gated_reward(scorer(score))` | `HARNESS_BLAST_RADIUS` | same |
| `plugin` | `FileTree(..., frozen=PLUGIN_FROZEN[host] + frozen)` | `plugin_runner(host, agent_args, overlay=frozen_files)` (section 8) | `tree_reflector(..., context_files=PLUGIN_CONTEXT[host])` | `gated_reward(scorer(score))` | `HARNESS_BLAST_RADIUS` | same, plus a container sandbox by default |

The two defaults in the last column are the ones the removed wrappers used to
set and the quickstarts now pass explicitly: a rollout is a real agent call, so
re-running each proposal's trajectory (`self_verify`) doubles the cost and
ranking on the whole held-out set is the dominant expense. The spec sets them
for the three directory kinds because the person writing the spec is a model
that has not read the cost model; they remain overridable in `evolve`.

Governance follows from `kind` exactly as it does in the quickstarts, so a user
cannot accidentally evolve a harness under skill rules. `agent_code` additionally
requires `entrypoint` and defaults `test_cmd` to `python -m pytest -q`; `plugin`
requires `host` and is the subject of section 8.

### 2.2 Field semantics

* **`score` has three forms** because users have three kinds of graders: a name
  from `rewards.SCORERS`, an arbitrary shell command (`{"cmd": ...}` gets the
  task as JSON on stdin and the answer in `$ANSWER`, prints a float in `[0, 1]`),
  and a Python reference. The shell form is what lets someone evolve *anything*
  without touching Python: a linter, a compiler, a diff against a golden file, a
  web check are all one command. `cmd` graders run through the same
  `_child_env` trimming as candidate code.
* **`agent` / `reflect` / `policies.*` are `Ref`s** resolved through the
  allowlist in `workspec.py`. Adding `agentdescent.agents`,
  `agentdescent.selection`, `agentdescent.sampling`, `agentdescent.advantage`,
  `agentdescent.fusion` and `agentdescent.staleness` to the default allowed
  prefixes is the only change the resolver needs. `"staleness": "guarded"` is
  sugar for `get_policy("guarded")`. `reflective_merge` is exposed as a single
  ref that fills both `conflict` and `fusion`, so the half-installed pair the
  policy guide warns about cannot be written.
* **`agg_config` holds numbers, `policies` holds rules.** This is the policy
  guide's own line, and the spec keeps it: a threshold is an
  `AggregatorConfig` field, a decision is a policy object.
* **Secrets never enter the spec.** Providers read keys from the environment on
  the worker side, as `SandboxSpec` does today. The spec is written to the run
  store and shown to the user, so this matters.
* **A spec is also a file** (`.agentdescent/<name>.evolve.json`), so a run is
  reproducible from the CLI and can be checked into the repo alongside the skill
  it evolves. "Evolve this again with more data" is then a one-line edit.

Implementation: `agentdescent/evolvespec.py` beside `workspec.py`, with
`EvolveSpec.from_dict / to_dict / validate()` and
`run_spec(spec, *, run_dir, on_round) -> EvolutionResult`. `validate()` builds
the `Policies` bundle and resolves every `Ref` without running anything, so the
engine's own composition checks fire at plan time rather than round one. This
module is the **only** place the mapping from JSON to `evolve()` lives; CLI and
MCP both call it.

## 3. The run store and background execution

```
~/.agentdescent/runs/<run_id>/
    spec.json          what was asked
    status.json        {"state": "running|done|failed|cancelled", "round": 3, "rounds": 6,
                        "best_reward": 0.81, "calls": 212, "usd": 1.72, "pid": 41022, ...}
    rounds.jsonl       one RoundInfo per line, appended by the on_round hook
    result.json        EvolutionResult.save() on completion
    tree/              the evolved directory, materialised (directory kinds)
    ledger/            the git ledger (repo_path=), which is also what makes resume work
    log.txt            stderr of the run
```

* The run is a **detached subprocess**: `python -m agentdescent.cli run
  --run-dir <dir>`. Detaching (not a thread) is what survives the MCP server
  being restarted by the host, which Claude Code does on `/mcp` reconnects and
  DSH does on profile reboot. `status.json` is written atomically
  (`os.replace`) after every round from the `on_round` callback.
* **Resume for free**: `evolve(repo_path=<run_dir>/ledger)` already resumes a
  ledger, so `agentdescent resume <run_id>` re-launches the same spec on the
  same ledger after a crash or a cancel.
* **Cancel** sends SIGTERM to the process group; `runners._sh` already uses
  `start_new_session`, so the worker agents die with it and no `claude -p` or
  `dsh` orphans are left behind.
* **Budget.** `evolve()` already has `max_seconds`, `max_rollouts` and
  `max_calls`, and they pass through. A dollar budget needs one small engine
  addition, flagged here as the only core change in the plan: a
  `stop_when(info: RoundInfo) -> bool` hook next to `on_round`, so the runner
  can end the run with `stop_reason="budget"` when `Usage.estimated_cost`
  crosses the line. Until it lands, `plan` converts `budget_usd` into
  `max_calls` from its per-call estimate.

## 4. The CLI (`agentdescent`)

A console script (`[project.scripts] agentdescent = "agentdescent.cli:main"`)
that exposes exactly the verbs the MCP server exposes, so a user can do by hand
anything the agent can do, and the SKILL.md can fall back to shell when a host
has no MCP.

```
agentdescent init      <path> [--kind skill_dir] [--data cases.jsonl]  # write a starter spec
agentdescent plan      <spec.json>                                    # validate + cost estimate, no run
agentdescent evolve    <spec.json> [--detach] [--budget 5]            # start a run
agentdescent status    [<run_id>]                                     # one run or all
agentdescent watch     <run_id>                                       # tail rounds.jsonl
agentdescent show      <run_id> [--diff]                              # evolved tree, diff vs original
agentdescent apply     <run_id> [--to <path>] [--dry-run] [--no-backup]
agentdescent cancel    <run_id>
agentdescent resume    <run_id>
agentdescent doctor                                                   # which agents/keys are available
agentdescent install   <dsh|claude-code|codex|gemini|opencode|cursor>
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
| `doctor()` | available agents, providers, missing keys | call first |
| `plan(spec)` | validated spec + cost estimate | **does not run**; `validate()` resolves refs and builds `Policies`, so composition errors surface here |
| `start(spec, budget_usd?)` | `run_id` | detaches; returns in < 1 s |
| `status(run_id?)` | `status.json` (+ last 3 rounds) | cheap; safe to poll |
| `show(run_id, diff=true)` | evolved tree, `write_to(dry_run=True)` plan, unified diff, `outcomes()` | what the user reads before deciding |
| `apply(run_id, to?)` | files written, backup path | **destructive**; host must confirm with the user |
| `cancel(run_id)` / `resume(run_id)` | new status | |

Tool names carry no prefix because every host namespaces them itself
(the host decides the prefix: `mcp__agentdescent__plan` under a dsh mcp-client
entry or a plain `.mcp.json`, `mcp__plugin_agentdescent_agentdescent__plan` when
Claude Code loads it as a *plugin* -- measured, see plugins.md). Two resources, for
hosts that support them: `agentdescent://runs` (list) and
`agentdescent://runs/{id}/rounds` (progress), so a host can render progress
without the model spending tokens on polling.

The cost estimate in `plan` is `rounds × n_workers × train_tasks` agent calls
for rollouts plus `rounds × cheap_eval_tasks` for ranking plus one held-out
sweep per committed candidate, priced from the provider's `Usage` rates when
known. It says which of those numbers it does not know (per-call tokens for a
tool-using agent) rather than hiding the gap.

Why `plan` is separate from `start`: it forces the "show the spec and the
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
1. Run doctor. Report anything missing; stop if no worker agent.
2. Establish the four things a spec needs: target, data, score, agent.
   - No data? Offer to draft 8-20 cases into eval/cases.jsonl and have the user
     check them. Never evolve against data the user has not seen.
   - No obvious score? Prefer `contains`/`exact`; offer a `cmd:` grader when
     the answer is a file, code, or a format check.
   - Leave `policies` empty unless the user asks for a mechanism by name; the
     empty bundle is the shipped run.
3. Call plan; show the spec and the cost estimate; get a yes.
4. start; then poll status about once per round, not more. Summarise round
   deltas, not raw JSON.
5. When done: show with diff=true. Explain *what changed and why* using
   outcomes(). Do not paste the whole tree.
6. Ask before apply. Mention the backup path afterwards.
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

### 7.2 DeepSeek Harness (`integrations/dsh/`)

DSH is the host that fits this design most naturally, because it *is* a plugin
system: models, tools, skills, hooks, sessions and the UI are all Cordis plugins
composed by a profile. Verified against the current developer preview:

| DSH mechanism | What it is | How AgentDescent uses it |
|---|---|---|
| Skills (`skill-filesystem`) | `SKILL.md` bundles, kebab-case names, discovered from `<project>/.dsh/skills`, `<project>/.agents/skills`, `~/.dsh/skills`, `~/.agents/skills`, plus `customSkillDirs` | the shared `SKILL.md` is copied to `~/.dsh/skills/agentdescent/SKILL.md` (or `.agents/skills`, which Codex also reads) |
| MCP (`@deepseek-ai/dsh-mcp-client`) | one plugin instance per server in `cordis.patch.yml`; tools appear as `mcp__<serverName>__<tool>` | one entry pointing at `agentdescent mcp` |
| Hooks (`hooks-claude-code`) | consumes **Claude Code's `hooks.json` format** (`SessionStart`, `PreToolUse`, `Stop`, ...) | the same `hooks/hooks.json` the Claude Code plugin ships, unchanged |
| Headless profile | `dsh --profile headless "task"` runs one session and prints the last assistant text to stdout | the **worker**: `cli_agent(["dsh", "--profile", "headless"])` |
| Profiles + `$DSH_HOME/cordis.patch.yml` | layered YAML config; `dsh plugin --profile <p> add github:owner/repo#main` installs a plugin via pnpm | tier A edits the patch file; tier B is an installable plugin |

**Tier A, config only (ships first).** `agentdescent install dsh` writes:

```
~/.dsh/skills/agentdescent/SKILL.md          the shared skill
~/.dsh/cordis.patch.yml                      appended (if absent):
```

```yaml
- id: mcp-agentdescent
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: agentdescent
    transport: stdio
    command: agentdescent
    args: ['mcp']
    toolCallTimeoutMs: 120000
    env:
      # dsh scrubs any ambient variable matching KEY|PASSWORD|SECRET|TOKEN
      # before starting an MCP server, so provider keys must be forwarded here.
      DEEPSEEK_API_KEY: !!js process.env.DEEPSEEK_API_KEY
      OPENAI_API_KEY:   !!js process.env.OPENAI_API_KEY
- id: hooks-agentdescent
  name: '@deepseek-ai/dsh-hooks-claude-code'
  config:
    configPath: ~/.dsh/skills/agentdescent/hooks.json
```

The env block is not optional and is the one DSH-specific trap: without it the
MCP server starts with no provider credentials, `doctor` reports every reflector
as unavailable, and nothing else explains why. `install dsh` therefore prints the
variables it forwarded and `doctor` checks for the scrubbed set explicitly.

**Tier B, native Cordis plugin `dsh-agentdescent` (later, optional).** A small
TypeScript package installable with
`dsh plugin --profile web add github:Birfy/dsh-agentdescent#main` that does what
config alone cannot:

* registers the skill through `ctx.skills.registerProvider()` and the MCP
  server through the same `mcp-client` plugin, so one `add` installs everything;
* a **Web UI panel** listing runs from `~/.agentdescent/runs` with live round
  progress and the diff, reading the `agentdescent://runs` MCP resources. DSH is
  web-first, and a run that takes an hour deserves a progress view rather than
  a polling model;
* an optional **loop plugin**: DSH's loop/scheduling plugins already run
  auto-research style cycles, and "re-evolve this skill nightly against the
  cases that failed this week" is exactly a scheduled spec. The spec file
  makes this a one-line job definition.

Tier B contains **no optimisation logic**; it calls the same CLI/MCP surface as
everything else, which keeps the promise that adding a host never touches the
core.

**DSH as the worker.** Add `agents.dsh()` beside `claude_code()` and `codex()`:

```python
def dsh(*, workspace=None, extra_args=(), **kwargs) -> Completion:
    """DeepSeek Harness headless profile as a Completion."""
    return cli_agent(["dsh", "--profile", "headless", *extra_args],
                     workspace=workspace, **kwargs)
```

and two layouts in `runners.LAYOUTS` so an evolved skill is materialised where
the worker actually looks for it:

```python
"dsh_skill":    ".dsh/skills/{name}",     # DeepSeek Harness project skill
"agents_skill": ".agents/skills/{name}",  # Agent Skills standard; DSH and Codex both read it
```

A DSH-hosted run whose spec says `agent: {"ref": "dsh"}` therefore evolves a
skill with DSH grading DSH, no other vendor in the loop; the reflector can still
be `openai_compatible(model="deepseek-v4-...")` for cost. Session hygiene for the
worker drops `DSH_*` from the environment and points `DSH_HOME` at an empty
directory inside the workspace, so the worker does not load the user's profile,
plugins, or this very MCP server.

### 7.3 Codex (`integrations/codex/`)

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

### 7.4 Other hosts

| Host | Skill location | MCP config | Worker command |
|---|---|---|---|
| Gemini CLI | `gemini-extension.json` + `GEMINI.md` + `skills/` | in the extension manifest | `gemini -p` |
| ~~OpenCode~~ | **implemented and verified** -- `~/.config/opencode/skill/`, `mcp` in `opencode.jsonc`, `opencode run`; see plugins.md | | |
| Cursor | `.cursor/rules/agentdescent.mdc` (rule wraps the skill text) | `.cursor/mcp.json` | not a worker; use `dsh`/`claude_code`/`codex`/API |

`agentdescent install <host>` is the single entry point for all of them; each
host is ~30 lines in `agentdescent/integrations/<host>.py` that knows the paths
and the manifest shape. Adding a host does not touch the core.

## 8. Evolving the plugins themselves

Everything above evolves *what a host runs*: a skill, an agent folder, a prompt,
a codebase. A host **plugin** is one level up: the DSH Cordis package that adds
tools, skills and hooks to `dsh`; the Claude Code plugin directory with its
skills, commands, agents, hooks and MCP servers; a Codex skills-plus-config
bundle. Those are the things a team actually maintains, and the
`dsh-agentdescent` and Claude Code plugins from section 7 are themselves
examples. So `plugin` is a fifth `kind`, and the question it has to answer is
the same as for every other kind: what is the state, how does one rollout run,
what guards it.

### 8.1 What a plugin is, per host

| Host | The tree (`FileTree` keys) | How a rollout loads the *candidate* plugin | Validate gate | Worker command |
|---|---|---|---|---|
| DSH | `package.json`, `src/**/*.ts`, `cordis.patch.yml`, `skills/**/SKILL.md`, `hooks.json`, `tests/**` | `HOME=<ws>` already makes `$DSH_HOME` resolve to `<ws>/.dsh`; setup writes `<ws>/.dsh/cordis.patch.yml` referencing the plugin by path (or runs `dsh plugin --profile headless add link:<ws>/plugin`) | `pnpm install --offline`, `pnpm build`, `pnpm test`; then `dsh --profile headless --dump-config` must list the plugin, or the gate fails | `dsh --profile headless "<task>"` |
| Claude Code | `.claude-plugin/plugin.json`, `skills/**`, `commands/**`, `agents/**`, `hooks/hooks.json`, `.mcp.json` | `claude -p --plugin-dir <ws>/plugin`; `CLAUDE_CONFIG_DIR=<ws>/.claude` so the user's real plugins and settings are not loaded | `claude plugin validate <ws>/plugin` (structure, JSON, referenced files exist) | `claude -p --plugin-dir <ws>/plugin --strict-mcp-config --mcp-config <ws>/.claude/mcp.json "<task>"` |
| Codex | `.agents/skills/**`, a `config.toml` fragment (MCP servers) | `HOME=<ws>` puts the fragment at `<ws>/.codex/config.toml` | TOML parses; every `mcp_servers.*.command` resolves on PATH | `codex exec "<task>"` |

Two facts make this cheap to build. First, `code_runner` already does
*materialise, setup, gate, run entrypoint with the task prompt*, and its
`_child_env` sets `HOME` to the workspace, so the host's home directory **is**
the workspace for free: `~/.dsh`, `~/.claude`, `~/.codex` all land inside the
sandbox and the user's real profile is never touched. Second, the Claude Code
`--plugin-dir` flag and DSH's `link:` install exist precisely for loading an
uninstalled plugin from a path. So `plugin_runner(host, ...)` is a thin
per-host table over `code_runner`: which `setup_cmd`, which `test_cmd`, which
`entrypoint`, which environment variables to forward. Nothing new in the
engine.

```python
def plugin_runner(host: str, *, name: str, agent_args=(), env_passthrough=(),
                  overlay=None, fixtures=None, timeout=900.0, sandbox_pool=None):
    spec = PLUGIN_HOSTS[host]                  # setup / validate / entrypoint / env per host
    return code_runner(
        [*spec.entrypoint, *agent_args], layout=f"plugin/{name}", name=name,
        setup_cmd=spec.setup, test_cmd=spec.validate, overlay=overlay,
        fixtures=fixtures, timeout=timeout, sandbox_pool=sandbox_pool,
        env={**spec.env, **{k: os.environ[k] for k in env_passthrough if k in os.environ}})
```

`env_passthrough` is necessary because `_child_env` deliberately strips every
variable but a short allowlist, and a host agent needs its provider key. The
spec names the variables, never the values, exactly as `SandboxSpec` does; the
value is read on the worker side at launch.

### 8.2 What "better plugin" means: tasks and reward

A plugin's job is to make the host better at something, so its tasks are
**end-to-end host tasks run with the plugin loaded**, scored like any other
run (`scorer`, or a `cmd` grader). Three kinds of task are worth having in
every plugin's dataset:

* **Capability tasks** exercise what the plugin adds: "total the `amount`
  column of `data.csv`" for a CSV skill; "list open runs" for the
  agentdescent plugin. Gold is the answer, `contains`/`exact` scores it.
* **Regression probes** check the plugin is *present and wired*: a task whose
  prompt is "which tools do you have?" and whose `cmd` grader greps the answer
  for the host's own spelling of the tool name (see plugins.md -- a plugin adds
  a `plugin_<name>_` segment). A refactor that breaks registration fails
  these before it fails anything subtle. On DSH, `--dump-config` in the gate
  covers most of this for free.
* **Cost probes** read structured output, `claude -p --output-format json`
  gives `total_cost_usd` and turn count, and a `cmd` grader turns "under N
  turns and under $X" into a `[0, 1]` score. This is how "the skill works but
  makes the agent take twelve turns" becomes a gradient instead of an
  anecdote. Reward stays scalar; a multi-objective plugin run is a task *mix*,
  and `Beam` or `ParetoFrontier` selection can keep more than one head alive
  while it settles.

A failed gate (`TEST_FAILURE_MARKER`) scores 0 through `gated_reward`, and
the failure text is what the reflector reads, so "you broke the build" and
"the plugin no longer registers its tool" are learning signals rather than
crashed rounds.

### 8.3 Governance and isolation

A plugin is a harness: it decides what tools the host has and, through hooks,
what the host may do. So the layer is `HARNESS_BLAST_RADIUS` (L1, every merge
through the oracle) and four things are **frozen by default**, enforced twice
as in `agent_code` (the strategy refuses proposals that touch them and the
runner overlays pristine copies after materialisation):

```python
PLUGIN_FROZEN = {
    "dsh":         ["tests/**", "pnpm-lock.yaml", "hooks.json", "**/permission*"],
    "claude_code": ["hooks/**", ".claude-plugin/marketplace.json", "**/permission*"],
    "codex":       ["**/permission*"],
}
```

`hooks` are frozen because a hook that blocks a tool call is the plugin's own
L0: an optimizer that could loosen it to score better would be optimising the
guard away. Lockfiles are frozen because they are regenerated by tooling in
`setup_cmd`, not authored by a model. Both lists are extendable and, with a
deliberate `frozen=[]`, overridable.

Isolation is stricter than for the other kinds. Candidate plugin **code runs
inside the host process**, with the host's tool access, so the `plugin` kind
defaults to the container provider (`sandbox_container.ContainerProvider`,
network on because the host needs its API) and `doctor` warns, rather than
silently downgrading, when no container engine is available. The trimmed
environment is still applied inside the container.

Cost has one new term: a per-rollout `pnpm install` for DSH plugins. Two
mitigations, both in the setup step and neither in the engine: a shared pnpm
store (`npm_config_store_dir` under `workspace_root`) so installs are
hardlinks after the first, and a build cache keyed by the hash of
`package.json` plus the lockfile, so a candidate that only touched `SKILL.md`
reuses the previous `node_modules` and `dist/`. `SharedSandboxPool` is the
place the cache lives when rollouts span processes.

### 8.4 The plugin evolving itself

The `dsh-agentdescent` Cordis plugin and the Claude Code plugin from section 7
are `plugin`-kind targets like any other, and the interesting dataset for them
is **the user requests the skill is supposed to handle**: "evolve
`~/.claude/skills/pdf-audit` against `eval/cases.jsonl`", with the skill
directory and cases staged as fixtures. The grader reads the run store and the
transcript, and all of its checks are mechanical:

* `plan` was called before `start` (the tool sequence in the transcript);
* the spec `plan` received validates (`agentdescent plan <spec>` exits 0);
* the host stopped before `apply` and asked (no `apply` call; the last
  message is a question);
* cost stayed under the probe's ceiling.

That is a reward for "the skill teaches the host to use the tool correctly",
which is what a plugin's `SKILL.md` is for, and the same loop then evolves the
tool descriptions, the `commands/evolve.md` shortcut and the hooks-adjacent
prose.

Two guards keep this from folding in on itself:

* **A recursion guard.** The worker environment carries
  `AGENTDESCENT_NESTED=1`. The MCP server's `start` refuses under it and
  returns a stub run id, so the transcript still shows *that* the host called
  `start` (which the grader wants) without a nested evolution actually
  running; `doctor` reports nested mode explicitly.
* **The controller never hot-swaps itself.** A self-evolved plugin reaches the
  user's real installation only through `apply`, which is a human decision on
  a diff, after an L1 oracle gate, with a backup. The loop can improve its own
  skill text and tool descriptions; it cannot change what is running while it
  runs.

## 9. Worker adapters: what the run spawns

`agents.claude_code()` and `agents.codex()` exist; `agents.dsh()` is added in
section 7.2. Two further additions:

* **Structured output.** `claude -p --output-format json` yields the answer and
  the exact token/cost figures; parsing it feeds `Usage` with real numbers
  instead of wall-clock estimates. Same for `codex exec --json`. Add
  `claude_code(structured=True)` and read `total_cost_usd`; this is what makes
  a dollar budget honest. DSH's headless profile prints text only; its `sdk`
  profile (JSON-RPC over stdio) is the route to token counts there, and is
  deferred until the preview's protocol settles.
* **Session hygiene.** A `_worker_env()` in `agents.py` that drops
  `CLAUDECODE`, `CLAUDE_CODE_*`, `CODEX_*`, `OPENAI_AGENT_*`, `DSH_*`, and points
  `CLAUDE_CONFIG_DIR`/`CODEX_HOME`/`DSH_HOME` at an empty directory inside the workspace,
  so the worker starts clean and cannot read the user's real config, memory or
  MCP servers.

## 10. Security and cost posture

* **Two trust boundaries, kept apart.** The host session (trusted, has the
  user's keys and config) starts a run. Each worker (semi-trusted, model-driven,
  edits files) runs in a leased workspace with a stripped environment and its own
  config dir. Candidate *code* (`agent_code` kind) and `cmd` graders run with
  `runners._child_env` as today. Nothing here weakens what the library already
  does.
* **Applying is explicit and backed up.** Same as today; the plugin just makes
  it a separate verb with a `dry_run` plan and a confirmation step in the skill.
* **Cost is visible before, during, after.** `plan` estimates, `status` reports
  calls and spend so far, `max_calls`/`max_seconds` (and later `stop_when`)
  stop the run.
* **The spec is the audit log.** Everything a run did is derivable from
  `spec.json` + the ledger; nothing hides in a closure. This is the same reason
  `workspec.py` refuses `cloudpickle`.

## 11. Delivery plan

Ordered so that each step is usable on its own and testable offline (the
`echo()` agent, `cli_agent(["python", "-c", ...])` as in
`examples/skill_dir_evolution.py`, and the router domain make every step
runnable without a key).

| Step | Lands in | Tests |
|---|---|---|
| 1. `evolvespec.py`: schema, `validate()` (refs + `Policies` bundle), `run_spec` composing the four `kind` rows | `agentdescent/evolvespec.py`; allowlist prefixes in `workspec.py` | every `kind` produces the same `evolve()` kwargs as the matching quickstart block; `policies` round-trips through `install_policy`; bad pairs are refused at `validate()` |
| 2. Run store + detached runner + `status.json` from `on_round` | `agentdescent/runstore.py`, `cli.py` | start with `echo()`, poll, cancel, resume on the same ledger |
| 3. CLI verbs | `cli.py`, `[project.scripts]` | `subprocess` tests against the offline domain |
| 4. MCP server (`[mcp]` extra) mirroring the CLI | `agentdescent/mcp.py` | tool schema snapshot; plan → start → status → show with `echo()` |
| 5. Shared `SKILL.md` + `install dsh` (tier A) + `integrations/claude-code` plugin + marketplace.json | `integrations/`, `agentdescent/integrations/dsh.py` | patch YAML round-trips; skill lands in a discovery root; plugin manifest validates |
| 6. `install codex` (+ others); `agents.dsh()` + `dsh_skill`/`agents_skill` layouts; structured-output worker adapters | `agentdescent/integrations/`, `agents.py`, `runners.py` | parse fixtures of `claude -p --output-format json` / `codex exec --json`; `tree_runner(layout="dsh_skill")` materialises under `.dsh/skills/` |
| 7. `plugin` kind: `plugin_runner` host table, `PLUGIN_FROZEN`, container default, `AGENTDESCENT_NESTED` guard, pnpm store/build cache | `runners.py`, `evolvespec.py`, `mcp.py` | offline: a stub host CLI that echoes its `--plugin-dir` / `$DSH_HOME`; frozen hooks survive a proposal that edits them; nested `start` returns a stub |
| 8. `stop_when` hook in `evolve()` for a dollar budget | `evolution.py` (the one core change) | a run ends with `stop_reason="budget"`; `on_round` semantics unchanged |
| 9. Docs page `docs/plugins.md`; README section "Use it from your agent" | docs | `test_docs_links`, `test_docs_examples` |
| 10. (optional) `dsh-agentdescent` Cordis plugin: one-command install, Web UI runs panel, scheduled re-evolution | separate repo | smoke test against `dsh --dump-config` |

Steps 1 to 4 are the product; 5 to 10 are packaging, the `plugin` kind, and one small engine hook.
Nothing in the aggregator, ledger, async runtime or governance changes.

## 12. Alternatives considered

* **Python-only, "just import it".** That is the status quo, and it is why the
  question is being asked: the audience inside an agent session does not want to
  write Python.
* **Bring the one-call wrappers back as the plugin's API.** Upstream removed them
  deliberately: each was a second signature to keep in step with `evolve()`.
  The spec is not a signature Python callers see; it is a wire format, and it
  has to exist anyway for MCP. One composer in `evolvespec.py`, tested against
  the quickstarts, is the smallest thing that satisfies both.
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

## 13. DSH facts this design relies on

Checked on 2026-09-05 against the developer preview; DSH says breaking changes
are expected, so `install dsh` should verify with `dsh --dump-config` rather
than assume.

* Headless one-off run: `dsh --profile headless "task"`, final assistant text on
  stdout (the old `dsh run` subcommand was removed). `$DSH_HOME` defaults to
  `~/.dsh`; profiles live under `$DSH_HOME/profiles/<name>`.
* Skill discovery roots and ranks: `.dsh/skills` (100), `.agents/skills` (200),
  `customSkillDirs` (300), `~/.dsh/skills` (400), `~/.agents/skills` (500),
  bundled (600). Names must match `^[a-z0-9]+(?:-[a-z0-9]+)*$`.
* MCP: one `@deepseek-ai/dsh-mcp-client` entry per server; stdio or
  streamable-http; tools named `mcp__<serverName>__<tool>`; ambient env matching
  `KEY|PASSWORD|SECRET|TOKEN` and `DSH_*` is scrubbed before the server starts.
* Hooks: `hooks-claude-code` plugin reads Claude Code's `hooks.json`
  (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`;
  exit code 2 denies).
* Plugins: `dsh plugin --profile <p> add "github:owner/repo#main"` (or
  `link:/path`), forwarded to pnpm; a plugin is a Cordis package with a
  `cordis.patch.yml`; the community topic is `dsh-plugin`.

Sources: [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness),
its `docs/config-catalog.md`, `docs/subsystems/skills.md`, `apps/cli/README.md`
and `packages/mcp/mcp-client/README.md`;
[DSH CLI user guide](https://deepseekdocs.com/en/docs/user-guide/cli);
[awesome-deepseek-harness](https://github.com/Dominic789654/awesome-deepseek-harness).

## 14. What changed since the first draft

Two upstream commits landed between the first draft and this one, and both
moved the design.

**`evolve()` is the only entry point** (the wrappers `evolve_skill`,
`evolve_skill_dir`, `evolve_agent_dir`, `evolve_agent_code` and the modules
`skill.py` / `skilldir.py` were removed; `scorer` / `SCORERS` moved to
`rewards`, `gated_reward` was added to `runners`, the blast-radius constants
moved to `governance`). The first draft said "`kind` selects the wrapper".
Now `kind` selects a row of the composition table in section 2.1, the spec
composer is the single place that wiring lives, and it is tested against the
quickstarts so it cannot drift from what a Python user would write. The two
cost defaults the wrappers used to set (`self_verify=False`,
`cheap_eval_tasks=4`) are set by the spec for directory kinds, because the spec
author is a model that has not read the cost model.

**Policies install through `bind` / `configure`** (`install_policy`, wrappers
default their `inner`, `PolicyUnboundError`, and `docs/policy-guide.md`). This
made a `policies` block in the spec possible: every shipped policy is now
constructible from JSON scalars alone and the aggregator hands it the verifier
and thresholds itself. The spec follows the guide's rule that thresholds are
`agg_config` and decisions are `policies`, exposes `reflective_merge` only as
the pair, and validates by building the bundle so the engine's composition
rules fire at `plan` time.

Smaller: the abstraction is renamed from "Recipe" to **EvolveSpec** to sit
beside `RolloutSpec` and to avoid colliding with the policy guide's own
"Recipes" table; `plan` is now also a CLI verb; `show` uses
`write_to(dry_run=True)`; the dollar budget is stated honestly as needing one
small engine hook rather than pretending `evolve()` has it.

**A fifth kind, `plugin`** (section 8), added on request: the host plugins
themselves, the `dsh-agentdescent` and Claude Code plugins included, are
evolvable. It reuses `code_runner` end to end, because `_child_env` already
makes the workspace the host's home and Claude Code's `--plugin-dir` / DSH's
`link:` install load an uninstalled plugin from a path. What is new is a
per-host table, a frozen-by-default list with hooks in it, a container
default, and a recursion guard for the plugin that evolves itself.
