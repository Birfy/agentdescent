# Use it from your agent — DeepSeek Harness, Claude Code, Codex, OpenCode

Everything else in these docs is Python: you write a dozen lines and hand them
to `evolve()`. This page is for the other audience — a person sitting *inside*
an agent (DeepSeek Harness, Claude Code, Codex, OpenCode) who wants to say *"evolve this
skill against these examples"* and have the agent do it. It is also for the
agent: the same surface is what the agent calls.

The design record — why it is shaped this way and what was considered — is
[Plugin for DSH, Claude Code, Codex and other agents](plugin-design.md).

## Install

```bash
pip install "agentdescent[mcp]"        # the CLI works without [mcp]; the MCP server needs it
agentdescent doctor                    # which agent CLIs, keys and optional pieces are here
agentdescent install dsh               # or: claude-code, codex, opencode
```

`install` writes the shared skill and the host's manifest, and nothing else:

| host | what it writes | how to load it |
|---|---|---|
| `dsh` | `~/.dsh/skills/agentdescent/SKILL.md` + `hooks.json`; an `@deepseek-ai/dsh-mcp-client` entry and the `hooks-claude-code` bridge appended to `~/.dsh/cordis.patch.yml` | restart `dsh`; check with `dsh --profile web --dump-config \| grep agentdescent` |
| `claude-code` | a plugin directory at `~/.agentdescent/plugins/claude-code/` (`plugin.json`, skill, `/agentdescent:evolve`, `.mcp.json`, a `SessionStart` hook) | `claude --plugin-dir ~/.agentdescent/plugins/claude-code`, or `/plugin marketplace add Birfy/agentdescent` then `/plugin install agentdescent@agentdescent` |
| `codex` | `~/.codex/skills/agentdescent/SKILL.md`; an `[mcp_servers.agentdescent]` block appended to `~/.codex/config.toml` | restart Codex; check with `codex mcp list` |
| `opencode` | `~/.config/opencode/skill/agentdescent/SKILL.md`; an `mcp.agentdescent` entry merged into `~/.config/opencode/opencode.jsonc` | restart OpenCode; check with `opencode mcp list` |

### DeepSeek Harness: a native plugin, or the files

`install dsh` writes files and needs no npm. The alternative is the Cordis
plugin in [`integrations/dsh-agentdescent`](https://github.com/Birfy/agentdescent/tree/main/integrations/dsh-agentdescent):

```bash
dsh plugin --profile web add link:/path/to/dsh-agentdescent
dsh --profile web --dump-config | grep -n agentdescent      # both rows compose
```

It registers the skill through `ctx.skills.register()` at runtime -- no file to
install, and it cannot drift from the package -- and its own `cordis.patch.yml`
adds the MCP row. Use one route or the other, not both.

The plugin also contributes a **runs panel** to the dsh web UI: an `evolve`
action in the session header showing how many runs are live, and the list
behind a click. It reads `agentdescent serve`, a read-only loopback view of the
run store:

```bash
agentdescent serve &            # http://127.0.0.1:8787/ -- GET only
```

The panel is a plain browser page as well, so it is useful without dsh. The
server answers a cross-origin read **only for a loopback `Origin`**, so the dsh
page on `:3080` can read it and a website you happen to visit cannot; without
the server running the panel just says how to start it.

Two things a dsh plugin needs that are easy to miss, both found by installing it
for real: `package.json` must carry `"dsh": {"bundle": {"patch":
"./cordis.patch.yml"}}` or `dsh plugin add` warns *"declares no dsh.bundle ...
not a profile layer"* and the plugin is inert; and rows go under `- insert:`,
because a patch file otherwise **overrides** rows by id.

It is idempotent (`--dry-run` shows what it would do) and honours `DSH_HOME`,
`CODEX_HOME` and `XDG_CONFIG_HOME`. Every manifest here was verified by writing
it and having the host read it back (dsh 0.1.2-rc.1, Claude Code 2.1.261,
codex-cli 0.153.4, opencode 1.18.29).

!!! warning "DeepSeek Harness scrubs provider keys"
    Before starting an MCP server, dsh removes every ambient variable matching
    `KEY|PASSWORD|SECRET|TOKEN`. The patch entry `install dsh` writes forwards
    `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` and
    `ANTHROPIC_API_KEY` explicitly (`!!js process.env.X`). Without that block
    the server starts with no credentials and `doctor` reports every reflector
    as unavailable, with nothing else to explain why.

## What the agent does

The skill file teaches the host model one procedure, and the tool surface
enforces its shape:

```
doctor  →  write a spec  →  plan (show the user)  →  start  →  status (once a round)  →  show  →  ask  →  apply
```

* **`plan` is separate from `start`.** `plan` resolves every reference, loads
  the data and builds the policy bundle without running anything, then returns
  the composed `evolve()` arguments and an upper bound on agent calls. A wrong
  field fails here, by name, not in round one of a background process.
* **`start` returns in under a second.** The run is a detached process; the host
  polls `status`. MCP tool calls time out and hosts restart their servers
  between sessions, so nothing blocks on `evolve()`.
* **`apply` is its own tool and says it is destructive.** The evolved artifact
  reaches your real directory only through it, after `show` has displayed the
  diff and the plan (`write_to(dry_run=True)`), and it backs up first.

**The tool names differ by how the server was added**, which matters when you
allowlist them or write a probe that greps for one (verified against Claude Code
2.1 and the dsh mcp-client docs):

| added as | name |
|---|---|
| a Claude Code **plugin** (what `install claude-code` writes) | `mcp__plugin_agentdescent_agentdescent__doctor` — `mcp__plugin_<plugin>_<server>__<tool>` |
| a plain MCP server entry in your own `.mcp.json` | `mcp__agentdescent__doctor` |
| a dsh `mcp-client` entry (what `install dsh` writes) | `mcp__agentdescent__doctor` — `mcp__<serverName>__<tool>` |
| a Codex or OpenCode server entry | `agentdescent`'s tools under that server's name |

In an interactive session you approve the tools when they are first called. In
headless use (`claude -p`) nothing prompts, so pass the names explicitly:

```bash
claude -p --plugin-dir <dir> --permission-mode acceptEdits \
  --allowedTools mcp__plugin_agentdescent_agentdescent__doctor \
                 mcp__plugin_agentdescent_agentdescent__status
```

`--allowedTools` is variadic, so give the prompt on stdin (or put it before the
flag) or it will be swallowed as another tool name.

## The spec

What the agent writes from your words, and what you can write yourself. It is
an `evolve()` call as data: every field is an ordinary argument or a public
building block, and `agentdescent.evolvespec.compose` is the only place the two
are joined.

```json
{
  "kind": "skill_dir",
  "target": "~/.claude/skills/pdf-audit",
  "data": {"path": "eval/cases.jsonl", "prompt": "question", "gold": "answer"},
  "score": "contains",
  "agent": {"ref": "claude_code", "extra_args": ["--permission-mode", "acceptEdits"]},
  "reflect": {"ref": "openai_compatible", "model": "deepseek-v4-flash"},
  "evolve": {"rounds": 6, "n_workers": 4, "asynchronous": true, "max_seconds": 3600}
}
```

`kind` picks a row of the composition table, which mirrors the quickstarts
line for line:

| `kind` | strategy | `run` | `propose` | reward | layer | defaults added |
|---|---|---|---|---|---|---|
| `text` | `SingleSlot(target)` | `model(template)` | `reflector` | `scorer` | L2 | rounds 8, patience 3, target 0.98 |
| `skill_dir` | `FileTree(load_tree(target))` | `tree_runner` | `tree_reflector` | `scorer` | L2 | rounds 6, `self_verify=False`, `cheap_eval_tasks=4` |
| `agent_dir` | same | `tree_runner(layout="claude_agent")` | same | same | L1 | same |
| `agent_code` | same, tests frozen | `code_runner(entrypoint, test_cmd)` | same | `gated_reward(scorer)` | L1 | same |
| `plugin` | same, hooks frozen | `plugin_runner(host)` | same | `gated_reward(scorer)` | L1 | same, container advised |

The other fields:

* **`data`**: one of `path` (`.json` / `.jsonl` / `.csv`), `hf`
  (`{"dataset", "split", "config", "limit"}`, through
  [the data layer](dataloader.md)) or `inline` (a list of rows); `prompt` and
  `gold` name the columns. A `fixtures` column is staged into the workspace.
* **`score`**: a name from `SCORERS` (`contains`, `exact`, `last_number`,
  `numeric_close`); `{"cmd": "./grade.sh"}` to grade with **any program** (the
  task as JSON on stdin, the answer in `$ANSWER`, a number in `[0, 1]` on
  stdout — a linter, a compiler, a golden-file diff); or `{"ref": "pkg.mod:fn"}`.
* **`agent` / `reflect`**: a short name (`claude_code`, `codex`, `dsh`,
  `opencode`, `openai_compatible`, `claude`, `echo`), or `module:attribute` inside the
  import allowlist, with keyword arguments beside it. `"call": false` names a
  callable rather than a factory. A cheap `reflect` behind an expensive `agent`
  is the usual trade.
* **`policies`**: one reference per slot (`selection`, `task_sampler`,
  `acceptance`, `conflict`, `fusion`, `promotion`, `proposal`, `staleness`),
  built from JSON scalars alone because the aggregator installs them through
  `bind` / `configure` ([the policy guide](policy-guide.md)). `reflective_merge`
  fills `conflict` and `fusion` together. Thresholds go in **`agg_config`**;
  decisions go here.
* **`evolve`**: anything `evolve()` takes, overriding the kind's defaults.
* **`allow`**: extra import prefixes for your own code. The package's own
  modules are always allowed and nothing else is; a spec is saved beside its
  run and shown to the user, so it never carries code or secrets.

`agentdescent init <path>` writes a starter spec with `kind` guessed from what
the path is.

Relative paths (`target`, `data.path`, a `cmd` grader) are resolved **when the
spec is read**, not when it runs: the detached run has the run directory as its
working directory and an MCP server has whatever the host launched it in, so a
relative path would otherwise mean three different files. The spec stored beside
the run holds the resolved paths and is re-runnable from anywhere.

## Where a run lives

```
~/.agentdescent/runs/<run_id>/       ($AGENTDESCENT_HOME/runs to move it)
    spec.json      what was asked
    status.json    state, round, best reward, calls, dollars if priced — replaced atomically each round
    rounds.jsonl   one RoundInfo per line
    result.json    EvolutionResult.save() at the end
    tree/          the evolved directory, materialised
    ledger/        the git ledger; resume re-launches on it
    log.txt        the detached process's output
```

`agentdescent status`, `watch`, `show`, `apply`, `cancel` and `resume` all read
this directory, so a run started from an agent can be inspected from a shell
and a run started from a shell can be picked up by an agent. `cancel` signals
the process group, so the worker CLIs die with the run; `resume` re-launches the
spec on the same ledger and the engine continues.

A dollar budget (`--budget` with `--usd-per-call`, or `budget_usd` on `start`)
uses `evolve(stop_when=)`: the run stops between rounds once
`calls × price` crosses the line and reports the spend it actually incurred.

## Evolving the plugins themselves

A host **plugin** — the DSH Cordis package that adds tools and skills to `dsh`,
a Claude Code plugin directory, a Codex skills-plus-config bundle — is one level
up from a skill, and `kind: plugin` evolves it. The AgentDescent plugins this
page installs are themselves valid targets.

```json
{"kind": "plugin", "host": "claude_code", "target": "./my-plugin",
 "data": {"path": "eval/plugin-cases.jsonl"}, "score": "contains",
 "agent": "echo", "reflect": {"ref": "openai_compatible", "model": "deepseek-v4-flash"},
 "env_passthrough": ["ANTHROPIC_API_KEY"]}
```

Each rollout materialises the candidate plugin, loads it into an **isolated copy
of the host** that lives inside the workspace (`HOME` is the workspace, so
`~/.dsh`, `~/.claude` and `~/.codex` are sandboxed for free), runs the host's
validate gate, then runs the host CLI on the task:

| host | loads the candidate with | gate | worker |
|---|---|---|---|
| `dsh` | `dsh plugin --profile headless add link:<plugin>` after `pnpm install && pnpm build` | `pnpm test`; `dsh --profile headless --dump-config` composes | `dsh --profile headless "<task>"` |
| `claude_code` | `claude -p --plugin-dir <plugin> --strict-mcp-config` | `claude plugin validate <plugin>` | the same command |
| `codex` | skills copied to `.agents/skills/`, `config.toml` to `.codex/` | the TOML parses | `codex exec --sandbox workspace-write --skip-git-repo-check "<task>"` |
| `opencode` | skills copied to `.opencode/skills/`, `opencode.jsonc` to `.config/opencode/` | the JSON parses | `opencode run "<task>"` |

A failing gate scores 0 in-band and its text is what the reflector reads, so
"the plugin no longer registers its tool" is a learning signal, not a crashed
round. `agent` is ignored for this kind — the host *is* the agent — but must be
present; `env_passthrough` names the variables the host needs (its provider
key) and never carries values.

Three things are stricter than for the other kinds:

* **Governance.** A plugin is a harness (`HARNESS_BLAST_RADIUS`, L1: every
  merge through the oracle), and **hooks, permission config, lockfiles and
  tests are frozen by default** (`PLUGIN_FROZEN`), enforced twice — the
  strategy refuses proposals that touch them and the runner overlays pristine
  copies after materialisation. A hook that blocks a tool call is the plugin's
  own L0; an optimiser that could loosen it would be optimising the guard away.
* **Isolation.** Candidate plugin code runs *inside the host process* with the
  host's tool access. Use a container sandbox
  ([sandboxes](sandboxes.md)) for anything you would not run by hand; `plan`
  says so in its notes.
* **Recursion.** Every worker runs with `AGENTDESCENT_NESTED=1`. When the
  plugin being evolved is the one that hosts this MCP server, the worker's
  `start` returns a stub — the transcript still shows the host *called* it,
  which is what a grader looks for, and no nested run starts. A self-evolved
  plugin reaches your real installation only through `apply`, a human decision
  on a diff after the L1 oracle, with a backup: the loop can improve its own
  skill text and tool descriptions, and it cannot change what is running while
  it runs.

What "a better plugin" means is up to the dataset. Three task shapes are worth
having: **capability tasks** (what the plugin adds — gold is the answer),
**regression probes** ("which tools do you have?" with a `cmd` grader that
greps for the tool name) and **cost probes** (`claude -p --output-format json`
gives turn count and `total_cost_usd`; a `cmd` grader folds "under N turns and
under $X" into a score). Reward stays scalar; a multi-objective plugin run is a
task mix.

## From Python

Nothing here is a second entry point. `EvolveSpec` is a wire format for callers
that are not Python; `compose()` produces the same `evolve()` call the
quickstarts show and is tested against them, so if you are writing Python,
write the call:

```python
from agentdescent import EvolveSpec, compose, load_spec

comp = compose(load_spec(".agentdescent/pdf-audit.evolve.json"))
comp.kwargs["rounds"]          # what the spec composed to
result = comp.run()            # == evolve(comp.tasks, comp.reward, **comp.kwargs)
```

## Next

* [Testing the plugins](testing-the-plugins.md) — what to run for each host,
  and what no test can tell you
* [Quickstart — evolve a directory](quickstart-directory.md) — the Python call
  the `skill_dir` kind composes
* [Using the policy slots](policy-guide.md) — what the `policies` block can name
* [Sandboxes](sandboxes.md) — the container provider the `plugin` kind wants
* [Design record](plugin-design.md) — why it is built this way
