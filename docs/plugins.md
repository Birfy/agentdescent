# Use it from your agent — DeepSeek Harness, Claude Code, Codex, OpenCode

Everything else in these docs is Python: you write a dozen lines and hand them
to `evolve()`. This page is for the other audience — a person sitting *inside*
an agent (DeepSeek Harness, Claude Code, Codex, OpenCode) who wants to say *"evolve this
skill against these examples"* and have the agent do it. It is also for the
agent: the same surface is what the agent calls.

The design record — why it is shaped this way and what was considered — is
[Plugin for DSH, Claude Code, Codex and other agents](plugin-design.md).

## Install

Install **0.7.0** from PyPI:

```bash
pip install "agentdescent[mcp]>=0.7.0"                 # the MCP server needs this extra (and Python >= 3.10)
agentdescent demo                      # a complete evolution, offline, no key
agentdescent doctor                    # which agent CLIs, keys and optional pieces are here
agentdescent install dsh               # or: claude-code, codex, opencode
```

Or clone the repository as described in [install and first run](install.md) and
install the MCP extra from that checkout:

```bash
pip install -e ".[mcp]"
```

Research demos beyond `agentdescent demo` use the
[source examples](install.md#source-examples).

The MCP SDK requires Python >= 3.10; AgentDescent supports 3.9. On 3.9 the
extra installs nothing and the tool surface is unavailable — the CLI verbs are
the whole API, which is what the shipped `SKILL.md` falls back to. `doctor` and
`install` both say so rather than letting it surface as a host reporting
`CONNECTION_CLOSED`.

If you have never run one, do `demo` before `install`: it builds a skill that is
wrong on purpose, twelve cases with known answers and a local program as the
agent, then runs the real loop — same staging, ledger, workers, merge and
held-out gate, only the model replaced. [Start here — the plugin in three
commands](plugin-quickstart.md) walks the whole path.

`install` writes the shared skill and the host's manifest, and nothing else:

| host | what it writes | how to load it |
|---|---|---|
| `dsh` | `~/.dsh/skills/agentdescent/SKILL.md` + `hooks.json`; an `@deepseek-ai/dsh-mcp-client` entry and the `hooks-claude-code` bridge appended to `~/.dsh/cordis.patch.yml` | restart `dsh`; check with `dsh --profile web --dump-config \| grep agentdescent` |
| `claude-code` | a plugin directory at `~/.agentdescent/plugins/claude-code/` (`plugin.json`, skill, `/agentdescent:evolve`, `/agentdescent:algorithms`, `.mcp.json`, a `SessionStart` hook) | `claude --plugin-dir ~/.agentdescent/plugins/claude-code`, or `/plugin marketplace add Birfy/agentdescent` then `/plugin install agentdescent@agentdescent` |
| `codex` | `~/.codex/skills/agentdescent/SKILL.md`; `/prompts:evolve` and `/prompts:algorithms` as `~/.codex/prompts/{evolve,algorithms}.md` (same bodies as Claude `/agentdescent:evolve` / `/agentdescent:algorithms`); an `[mcp_servers.agentdescent]` block appended to `~/.codex/config.toml` | restart Codex; check with `codex mcp list`. **Or skip `install` entirely** and use the plugin — see below |
| `opencode` | `~/.config/opencode/skill/agentdescent/SKILL.md`; `/agentdescent/evolve` and `/agentdescent/algorithms` under `~/.config/opencode/command/agentdescent/` (legacy singular `command/`, nested to mirror Claude's namespaced slash); an `mcp.agentdescent` entry merged into `~/.config/opencode/opencode.jsonc` (into `opencode.json` beside it when your `opencode.jsonc` has comments; OpenCode merges the two) | restart OpenCode; check with `opencode mcp list` |
| `grok-bot` | scaffold under `~/.agentdescent/plugins/grok-bot/` (`SKILL.md`, `HOST.md`, `mcp.example.json`) — **not** a one-click wire-up: Grok Bot has no public plugin API for local skill+stdio MCP; `HOST.md` names `algorithms list`/`show` and MCP discovery | read `HOST.md`; ask Grok Bot to save the skill as a user skill; drive runs with the CLI, or run `agentdescent mcp --transport streamable-http` behind an HTTPS tunnel and add the URL as a custom MCP server. `doctor` reports `mcp n/a` when the scaffold is current |

`agentdescent doctor` also reads what `install` left behind, without writing
anything: one `host <name>: ok | stale | partial | missing` line per host, and
for any installed host the line names `skill` and `mcp` too (for example
`host opencode: partial (skill ok, mcp missing; cli …)`), so a stale skill after
`pip install -U` or a missing MCP entry is visible without reading the problems
section (the same data is `hosts` in the JSON report and the MCP `doctor` tool).
`stale` means the `SKILL.md` a host holds differs from the one this package would
write now -- what a `pip install -U` leaves behind -- and `partial` means the
skill or the MCP entry is absent; both are problems (exit code 1) and the fix is
to re-run `agentdescent install <host>`. A host CLI on `PATH` with nothing
installed is a hint, not a problem, because `pip install` wires nothing and not
every host on a machine should be touched. An OpenCode config with comments (`opencode.jsonc`, or an
`opencode.json` you keep by hand) that cannot be parsed but names `agentdescent`
reports its MCP entry as `unknown` rather than guessing; one that never names it
holds no entry of ours, so with the skill installed the host is `partial`.
`unknown` alone never makes a host `partial`, so a user who never installed
anything is `missing`. Grok Bot's scaffold reports `mcp n/a` (no host-native
MCP config to check); with a current skill that is still `ok`, not `partial`. An install made through a
host's own marketplace (`codex plugin add`, Claude Code's `/plugin install`)
leaves no files here, so it reads `missing` and the hint names that route too.
Only presence is checked: a launcher pinned to an interpreter that has since
been deleted still reads `ok`.

Supported hosts today are DeepSeek Harness (`dsh`), Claude Code, Codex,
OpenCode, and a **Grok Bot scaffold** (`agentdescent install grok-bot`). The
Grok Bot path writes the shared skill plus `HOST.md` / `mcp.example.json`
under `~/.agentdescent/plugins/grok-bot/`; it does **not** edit a host config,
because Grok Bot has no public one-click plugin API for local skill directories
or stdio MCP (custom connectors need public HTTPS). For those, run
`agentdescent mcp --transport streamable-http` (loopback, port 8000, path
`/mcp`) and expose it through an HTTPS tunnel; the server has no auth, and a
non-loopback `--host` needs `--allow-remote`. `doctor` reports
`mcp n/a` for that host when the scaffold is current. A future one-click route
(if/when a host-native API appears) is still tracked on #165.

### Uninstall

`agentdescent uninstall <host>` (`--dry-run` to preview, `--home` as for
`install`) takes back what the table above says `install` wrote: the skill
files, the Claude Code plugin directory, the Grok Bot scaffold under
`~/.agentdescent/plugins/grok-bot/`, the OpenCode
`command/agentdescent/{evolve,algorithms}.md` slash entries and the Codex
`prompts/{evolve,algorithms}.md` slash entries (only while still
byte-for-byte stock), the `cordis.patch.yml` block, the
`[mcp_servers.agentdescent]` table in `config.toml`, the `mcp.agentdescent`
entry in `opencode.jsonc` or `opencode.json`. Everything around them is left as it was, and so is
the run store in `~/.agentdescent`.

What counts as "ours" differs by file:

* **dsh**: the block between the two marker lines is owned by the markers and
  removed whole, including any edit made inside it. A block from before the end
  marker existed ends at its hooks row (`configPath: .../agentdescent/hooks.json`);
  rows you added after it stay. If that row is not found, the block is kept and
  reported.
* **Codex**: the table is removed wherever it now sits, so tables Codex added
  after it (`[projects."..."]`) are fine, but only if it is still byte for byte
  what `install` wrote (the current block with its `env_vars` line, or the
  older one without it). A key you added under it, or a table you wrote
  yourself, is kept.
* **OpenCode**: only the pinned launcher `install` writes. The bare
  `["agentdescent", "mcp"]` entry (what `opencode mcp add` writes) is treated
  as yours. `install` adds a `$schema` key only to a file it creates, and
  `uninstall` deletes such a file once our entry was all it held; an empty
  `mcp` object you had is left behind. A file with comments is never
  rewritten: `install` writes the entry into `opencode.json` beside it instead,
  and if that file cannot be parsed either it prints a `FAILED:` line with the
  entry to add by hand and exits 1 (`--json` reports `"ok": false`). The
  `command/agentdescent/evolve.md` and `algorithms.md` slash files are removed
  only while each still matches the stock Claude prompt; an edit is kept.
* **Codex slash files**: `prompts/evolve.md` and `prompts/algorithms.md` are
  removed only while stock; an edit is kept. (The skill and MCP table rules
  above still apply.)

Anything kept is printed as `kept ... (...)` and makes the command exit 1, so a
script can tell "fully removed" from "one step is yours"; nothing to remove
exits 0 and prints `nothing to remove`. An install made through the host's own
marketplace is removed with that host's plugin command, not with this one.

### Codex: the same plugin as Claude Code

Codex reads the Claude Code plugin format, marketplace and all, so the package
this repository already publishes installs there in one command — and brings
the MCP server with it, so none of the file edits above are needed:

```bash
codex plugin marketplace add Birfy/agentdescent      # or a local checkout path
codex plugin add agentdescent@agentdescent
codex plugin list                                    # installed, enabled
codex mcp list                                       # agentdescent ... enabled
```

Verified against codex-cli 0.153.4: `codex plugin list` reads
`.claude-plugin/marketplace.json` and resolves the plugin at
`integrations/claude-code`, and the server it declares shows up in
`codex mcp list` with nothing written into `mcp_servers` by hand.

### Why OpenCode is configuration and not a plugin

OpenCode has a plugin system too, but it is a different thing: a plugin is a JS
module (`@opencode-ai/plugin`) whose `Hooks` are interception points —
`tool.execute.before` / `.after`, `chat.message`, `chat.params`,
`permission.ask`, `event`, `auth`, and a `config` hook that can mutate the
loaded config.

It *could* therefore inject the MCP server programmatically. It is not worth it
for what AgentDescent contributes: a skill and an MCP server are both plain
configuration in OpenCode, so the plugin would add an npm package, a publish
step and a version to keep in step, to arrive at the same two entries
`install opencode` writes directly. A plugin would earn its place only for
behaviour configuration cannot express — intercepting tool calls, or pushing
run status into the chat — which is not what this integration does.

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
`CODEX_HOME` and `XDG_CONFIG_HOME`, unless you pass `--home <dir>`: an explicit
`--home` is the host's whole world, so those variables are ignored and nothing
outside `<dir>` is read or written (also for `uninstall`). Every manifest here was verified by writing
it and having the host read it back (dsh 0.1.2-rc.1, Claude Code 2.1.261,
codex-cli 0.153.4, opencode 1.18.29).

!!! warning "DeepSeek Harness scrubs provider keys"
    Before starting an MCP server, dsh removes every ambient variable matching
    `KEY|PASSWORD|SECRET|TOKEN`. The patch entry `install dsh` writes (and the
    native plugin's `cordis.patch.yml`) forwards the same list as Codex,
    explicitly (`!!js process.env.X`): `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`,
    `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ARK_API_KEY`
    and `AGENTDESCENT_HOME`. Without that block the server starts with no
    credentials and `doctor` reports every reflector as unavailable, with
    nothing else to explain why. A spec whose `api_key_env` names another
    variable needs that name added to the `env` expression by hand (`doctor`
    still reads the block as `ok`; re-running `install dsh` rewrites the whole
    block, so add the name again afterwards). An older four-name block reads as
    `stale`, and `install dsh` repairs it.

!!! warning "Codex scrubs the environment of MCP servers"
    Codex starts an MCP server with `HOME LOGNAME PATH SHELL USER LANG LC_ALL TERM
    TMPDIR TZ` and whatever the server's `env_vars` names. `install codex` writes
    an `env_vars` line into its `[mcp_servers.agentdescent]` table, and the plugin's
    `.mcp.json` carries the same key, forwarding `DEEPSEEK_API_KEY`,
    `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
    `ARK_API_KEY` and `AGENTDESCENT_HOME`. Without it the server sees no keys
    (`doctor` reports every reflector as unavailable) and, with `AGENTDESCENT_HOME`
    set, writes its runs somewhere `agentdescent status` in your shell does not
    look. Re-running `install codex` repairs an older block that lacks the line
    (`doctor` reads it as `stale`); a spec that sets `api_key_env` to another
    variable needs that name added to `env_vars` by hand.

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
* **Questions about an archive are read-only tools.** `safety` (safe to keep
  seeding from?), `portfolio` (layer ranking and a suggested budget split) and
  `audit_trace` (one adoption's evidence rings) return the same JSON as the
  CLI's `safety` / `portfolio` / `audit <run_id>` verbs and write nothing.
  A missing archive is an error that names the Python meta API
  (`continuous_meta_run`; see [consolidation](consolidation.md)) — `evolve` /
  plugin runs never write one, and there is no CLI/MCP verb that does. Tell
  the user that; do not invent a "consolidation" step. `exam` runs an oracle
  module, so it stays CLI-only.

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

* **`algorithm`** (optional): a named method from `agentdescent algorithms list`
  (`rrsi` and `archive_selection` today; the catalog is
  [named algorithms](algorithms.md); to add one card see
  [Register a named Algorithm card](algorithm-authoring.md)). A name or `{"name": ..., "params": {...}}`
  replaces the kind's default recipe with that method's policy/knob defaults;
  explicit `policies` / `evolve` fields in the same spec still win. `plan` echoes
  the resolved card, its parameters and its extra cost, and rejects an unknown
  name, an unknown parameter, a wrong type or a kind the method does not support
  before any run starts. A host can discover the same cards with the
  `list_algorithms` / `describe_algorithm` MCP tools, or write the field with
  `agentdescent init --algorithm NAME --param KEY=VALUE`. Third-party packages
  can add a name by exposing an `Algorithm` under the
  `agentdescent.algorithms` entry-point group; a plugin that fails to load is
  skipped, printed as a warning by `agentdescent algorithms list`, and reported
  by `agentdescent.algorithms.external_errors()` rather than taking the shipped
  registry down. The value loads to an `Algorithm` (or a zero-argument factory
  returning one), and its `name` must equal the entry-point name:

  ```toml
  [project.entry-points."agentdescent.algorithms"]
  my_method = "my_pkg:ALGORITHM"
  ```
* **`data`**: one of `path` (`.json` / `.jsonl` / `.csv`), `hf`
  (`{"dataset", "split", "config", "limit"}`, through
  [the data layer](dataloader.md)) or `inline` (a list of rows); `prompt` and
  `gold` name the columns. A `fixtures` column is staged into the workspace.
* **`score`**: a name from `SCORERS` (`contains`, `exact`, `last_number`,
  `numeric_close`); `{"cmd": "./grade.sh"}` to grade with **any program** (the
  task as JSON on stdin, the answer in `$ANSWER`, a number in `[0, 1]` on
  stdout — a linter, a compiler, a golden-file diff); or `{"ref": "pkg.mod:fn"}`.
* **`agent` / `reflect`**: a short name (`claude_code`, `codex`, `dsh`,
  `opencode`, `openai_compatible`, `claude`, `host_model`, `echo`), or `module:attribute` inside the
  import allowlist, with keyword arguments beside it. `"call": false` names a
  callable rather than a factory. A cheap `reflect` behind an expensive `agent`
  is the usual trade.

  `openai_compatible` and `anthropic_compatible` have **three connection
  parameters**, and no others: `api_key_env` (the *name* of the variable holding
  the key), `base_url_env` (the *name* of a variable holding the endpoint) and
  `default_base_url` (the endpoint itself, used when that variable is unset).
  Defaults are `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `https://api.openai.com/v1`
  (`ANTHROPIC_*` / `https://api.anthropic.com` for the Anthropic twin), so a key
  that is not OpenAI's needs an endpoint too, or it is sent to `api.openai.com`:

  ```json
  "reflect": {"ref": "openai_compatible", "model": "deepseek-v4-flash",
              "api_key_env": "ARK_API_KEY",
              "default_base_url": "https://ark.cn-beijing.volces.com/api/coding/v3"}
  ```

  Every other keyword (`temperature`, `thinking`, ...) goes into the request body.
  A **reasoning model** (Ark `doubao-seed-code` thinks by default) sends nothing
  while it thinks, so a non-streaming call can hit the 120 s `"timeout"`: set
  `"stream": true` on the block, raise `"timeout"`, or pass the provider's own
  switch (`"thinking": {"type": "disabled"}` on Ark). A timeout is reported with
  the endpoint, model and seconds, not as a bare `TimeoutError`.
  A spec that writes `base_url`, `api_key` or a similar guess is **refused** by
  `plan`, `evolve` and MCP `start` with the spelling to use; `plan` also warns when
  a non-OpenAI `api_key_env` has no endpoint configured. A key is never written in
  a spec.

### Which model a run uses

A worker agent *is* the host CLI as a subprocess (`claude -p …`, `codex exec …`,
`dsh --profile headless …`, `opencode run …`), so the question "does it use the
model my agent is configured with" has a precise answer: **not by default, and
on purpose.**

`agents.worker_env()` inherits the environment — so provider keys in
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` reach the worker —
but points each host's config directory *inside the rollout workspace*
(`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `DSH_HOME`, `OPENCODE_CONFIG_DIR`). The
worker therefore starts clean: it cannot read your plugins, your memory, your
MCP servers — or your model choice and your subscription login, which live in
those directories. That isolation is what makes a rollout reproducible, and it
is what keeps a plugin evolving *itself* from reading the copy being rewritten.
The worker's working directory **and** its `PWD` are the rollout workspace
(`isolate: false` included): a CLI that resolves its project from `$PWD` rather
than the real cwd, as OpenCode does, would otherwise run in the directory you
launched `agentdescent` from.

Two switches, both ordinary spec fields:

```json
"agent":   {"ref": "claude_code", "extra_args": ["--model", "haiku"]},
"reflect": {"ref": "claude_code", "extra_args": ["--model", "opus"]}
```

**`extra_args` pins a model** — the flag is the host's own: `claude --model`,
`codex exec -m/--model`, `opencode run -m/--model` (as `provider/model`). For
Claude Code and Codex that is all it takes, isolation intact. `dsh` has **no**
`--model`: its model comes from the profile, so select it with `--patch` or a
prepared `DSH_HOME`.

!!! warning "OpenCode: `-m` only reaches providers built into OpenCode"
    An isolated OpenCode worker cannot read `~/.config/opencode`, and a **custom
    provider** — an OpenAI-compatible endpoint such as Volcengine Ark, a local
    vLLM, a company gateway — is defined in your `opencode.json`. `-m ark/model`
    then fails every call with OpenCode's opaque
    `UnknownError: Unexpected server error` (the same text as `-m nosuch/model`),
    while `plan` still says `ok`. Two fixes:

    - `"isolate": false` — the worker uses your own OpenCode setup (see below).
    - **Keep isolation and hand the worker a provider-only config file** through
      the agent's `env`:

    ```json
    "agent": {"ref": "opencode", "extra_args": ["-m", "ark/deepseek-v4-flash"],
              "env": {"OPENCODE_CONFIG": "/abs/path/providers.json"}}
    ```

    ```json
    {
      "$schema": "https://opencode.ai/config.json",
      "provider": {
        "ark": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "Ark",
          "options": {"baseURL": "https://ark.cn-beijing.volces.com/api/coding/v3",
                      "apiKey": "{env:ARK_API_KEY}"},
          "models": {"deepseek-v4-flash": {"name": "deepseek-v4-flash"}}
        }
      }
    }
    ```

    Put **only** the `provider` block in that file, and take keys from the
    environment with `{env:VAR}` — never inline, the file is read by a model-driven
    process. It must go in `env`: an `OPENCODE_CONFIG` exported in the shell that
    launches `agentdescent` is dropped for an isolated worker (a config file would
    otherwise bring its MCP servers with it). AgentDescent does not copy providers
    out of your `opencode.json` for you, because that file also holds inline keys,
    MCP servers and plugins that isolation exists to keep out. A failed isolated
    call now says this in its error, and `plan` warns when `extra_args` pins a
    `provider/model`.

```json
"agent": {"ref": "claude_code", "isolate": false}
```

**`isolate: false` hands the worker your real setup** — your configured model,
your subscription login, your plugins. This is the literal answer to *"use the
model my main agent is configured with"*, and the reason it is not the default:
the worker now sees state you can change between rounds, and a `kind: plugin`
run would load the very plugin it is rewriting.

Either switch works for `reflect` as well as `agent`, and pointing **both** at a
host CLI is the way to run with **no provider key at all** — every call goes
through the CLI's own authentication:

```json
{
  "kind": "skill_dir",
  "target": "~/.claude/skills/pdf-audit",
  "data": {"path": "eval/cases.jsonl", "prompt": "question", "gold": "answer"},
  "score": "contains",
  "agent":   {"ref": "claude_code", "extra_args": ["--permission-mode", "acceptEdits"], "isolate": false},
  "reflect": {"ref": "claude_code", "isolate": false}
}
```

#### Borrowing the host's model: `host_model`

The third route needs no model name and no key at all:

```json
"agent":   {"ref": "claude_code"},
"reflect": {"ref": "host_model"}
```

`host_model` means *the model of the agent that started this run*, and it takes
whichever of two routes the host allows.

**Route 1 — MCP sampling.** `sampling/createMessage` lets a server ask its
client for a completion, so the model is the one the session is running, with
the session's authentication and policy. This is the better route and it is
currently the rarer one. Measured, by logging what each host sends at
`initialize`:

| host | `clientInfo.name` | capabilities it declares | sampling? |
|---|---|---|---|
| Claude Code 2.1.261 | `claude-code` | `roots`, `elicitation` | no |
| OpenCode 1.18.29 | `opencode` | `roots` | no |
| DeepSeek Harness (`dsh-mcp-client` 0.0.1) | `dsh-mcp-client` | none | no |
| Codex | — | — | not measured: `codex mcp list` never opens a session |

**Route 2 — the host's own CLI.** Since no host measured supports sampling,
`host_model` falls back to running that host's CLI (`claude`, `codex`, `dsh`,
`opencode`) with the user's real configuration (`isolate: false`), so it uses
the model and the login they have set up. The server picks the CLI from the
name the client sent, and only if it is on `PATH`.

`start` says which route a run got:

```json
{"ok": true, "host_model_available": true, "host_model_route": "claude_code"}
```

The two are not the same thing, and the difference is worth knowing:

| | sampling | the host's CLI |
|---|---|---|
| model | the live session's | the user's *configured* one |
| outlives the session | no — the bridge dies with it | yes |
| cost | the host's own call | a fresh CLI process per reflection |

Sampling also works across a gap worth understanding. `start` returns in
milliseconds and the run proceeds in a **detached process** that has no MCP
session and cannot get one, so the server stands up a loopback bridge holding
the live session and hands its address to the run it launches. If that bridge
dies mid-run — the usual reason being that you closed the agent — the run falls
back to the CLI route rather than failing.

!!! warning "Sampling is deprecated at the protocol level"
    MCP revision **2026-07-28 (SEP-2577)** deprecates `sampling`, along with
    `roots` and `logging`. The shape is unchanged and the Python SDK still ships
    it, so route 1 works today and warns. Route 2 does not depend on it, and
    neither do `extra_args` and `isolate`.

* **`algorithm`**: a named recipe from the registry, `"rrsi"` or
  `{"name": "rrsi", "params": {"b_max": 3}}`. It supplies defaults for
  `policies` slots, `agg_config` and `evolve` knobs, so anything you set
  explicitly wins and `plan` lists what you overrode. `agentdescent algorithms
  list [--kind K]` and `agentdescent algorithms show <name>` (both take
  `--json`) print what is available: the kinds each applies to, its
  parameters and defaults, its extra cost, and a fidelity label
  (`benchmark_faithful`, `mechanism_microport`, or `preset` for an engine
  component under an algorithm's name). The registry lives in the package, so
  it works from the wheel; today it ships `rrsi` (`text`; screen, proposal and
  acceptance policies, see [RRSI](algo-rrsi.md)) and `archive_selection` (any
  kind; DGM-style archive parent sampling). Unknown names, unknown or
  mistyped parameters and a kind the algorithm does not apply to fail in
  `plan`, naming the field and the valid choices.
* **`policies`**: one reference per slot (`selection`, `task_sampler`,
  `acceptance`, `conflict`, `fusion`, `promotion`, `proposal`, `staleness`),
  built from JSON scalars alone because the aggregator installs them through
  `bind` / `configure` ([the policy guide](policy-guide.md)). `reflective_merge`
  fills `conflict` and `fusion` together. Thresholds go in **`agg_config`**;
  decisions go here.
* **`evolve`**: the JSON-valued knobs `evolve()` takes (`rounds`, `n_workers`,
  `seed`, `held_out_frac`, `asynchronous`, …), overriding the kind's defaults.
  A parameter that takes an *object* (`strategy`, `aggregator_factory`, `run`,
  `propose`, …) cannot be written as a string, so `plan` refuses it, and an
  unknown key, naming the field; use `policies` for rules and `agent` /
  `reflect` for models.
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
    ledger/trajectories/  one record per rollout (task, answer, reward); off with "record_rollouts": false
    log.txt        the detached process's output
```

A run outlives the tool call that started it, but it **inherits the environment
of the process that launched it** — including provider keys and, in sandboxed
or proxied setups, `HTTPS_PROXY`. That is what you want from an interactive
session, which stays open. It bites in one specific case, measured: a run
started from a *headless* `claude -p` kept working until that process exited,
then failed every call with `[Errno 111] Connection refused`, because the proxy
it had been handed belonged to the process that had gone. The run recorded
`state: failed` with that error, and `agentdescent resume <run_id>` from a live
shell picked it up on the same ledger and finished it. If you start runs from a
short-lived host process, expect to resume them from somewhere longer-lived.

`agentdescent status`, `watch`, `show`, `apply`, `cancel` and `resume` all read
this directory, so a run started from an agent can be inspected from a shell
and a run started from a shell can be picked up by an agent. `cancel` signals
the process group, so the worker CLIs die with the run; `resume` re-launches the
spec on the same ledger and the engine continues.

### A run that learned nothing: `propose_errors`

`rounds.jsonl` rows (and `status` / `show`, as `propose_errors=N`) count the
`reflect` calls that **raised** in a round, with the first exception as
`propose_error`. Compare it with `considered`: `considered: 0` and
`propose_errors: 0` is a reflector that had nothing to say; `considered: 0` and
`propose_errors > 0` is one that was never heard — typically `openai_compatible`
without `OPENAI_API_KEY`, or a model name the endpoint does not serve. When
**every** proposal raises in **two consecutive rounds** the run stops as
`state: failed` with `error: reflector failed on every call in rounds 1-2: <first
exception>` and `agentdescent evolve` exits non-zero, so the remaining paid
rollouts are not spent. A reflector that fails only some of the time does not
fail the run; the counts are recorded and a `RuntimeWarning` names the first
error. Fix the key or model (`agentdescent doctor`, `plan` warns about the
missing key) and `resume` the run.

### A run that learned nothing: `samples`

`show` (CLI `show` / `show --json`, and the MCP `show` tool) carries `samples`:
up to five of the lowest-reward rollouts of the final artifact, one per task,
each with `task_id`, `reward`, the agent's `output` and the task's `gold` (both
cut to 300 characters; the whole block stays under 4 KB). That is the answer to
"did the agent answer wrongly, or did scoring reject a right answer": an output
of `21` against a gold of `321` is the first case. The records live under
`ledger/trajectories/` in the run directory, contain only what your own agent
was shown and answered, and never leave your machine. Set `"record_rollouts":
false` in the spec to record nothing; `show` then has no `samples` key.

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
| `dsh` | `dsh plugin --profile headless add link:<plugin>` after `pnpm install` and `pnpm run --if-present build` | `pnpm run --if-present test`; `dsh --profile headless --dump-config` composes | `dsh --profile headless "<task>"` |
!!! note "dsh: `build` and `test` are optional, and the entrypoint needs a provider"
    A dsh plugin is often plain ESM with nothing to compile — the one this page
    installs is — so both scripts run with `--if-present`. A package that
    declares them still runs them and still fails the gate when they fail; for
    one that declares neither, `--dump-config` is the gate, and it is a real
    check that the plugin loads.

    The worker's `dsh` is a fresh profile inside the workspace with none of your
    credentials, so the entrypoint stops at `MISSING_CREDENTIAL` unless you give
    it one. Two spec fields do that: `entrypoint` is appended to the host's
    command, and `env_passthrough` names the variables to carry through.

    ```json
    {"kind": "plugin", "host": "dsh", "target": "./my-dsh-plugin",
     "entrypoint": ["--patch", "/abs/path/to/provider.patch.yml"],
     "env_passthrough": ["DEEPSEEK_API_KEY"]}
    ```

    A patch overriding `llm-deepseek` by id (a bare row, not `- insert:`) points
    it at any OpenAI-compatible endpoint:

    ```yaml
    - id: llm-deepseek
      config:
        baseURL: https://your-endpoint/v3
        apiKeyEnv: YOUR_KEY_VAR
    ```

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

### Fixing input-file errors

`agentdescent plan <spec.json>` reports input errors and exits with status 2.
Spec-file read errors appear on stderr; dataset validation errors appear in the
plan response (`ok: false`). For malformed spec or dataset JSON, the diagnostic names the file,
line and column to edit. JSONL line numbers count physical lines, including
blank lines. Files must be readable UTF-8; a spec must be a JSON object. Fix
the named input, then rerun `plan` before starting an evolution.
