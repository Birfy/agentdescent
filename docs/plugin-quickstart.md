# Start here — install the plugin and see it work

The rest of these docs assume you already have a dataset, a provider key and an
agent CLI. This page assumes none of that. It is the shortest path from *"I
just heard about this"* to *an evolution that actually ran on your machine*, and
then to the plugin living inside your agent.

Three commands, in order. Nothing here costs money and nothing needs an API key
until step 3.

## 1. Install

Install **0.6.0** from PyPI with its MCP extra:

```bash
pip install "agentdescent[mcp]>=0.6.0"
```

Or, from a source checkout:

```bash
pip install -e ".[mcp]"
```

See [Install and first run](install.md) for the source checkout and
[source examples](install.md#source-examples).

`[mcp]` is the extra that lets agents talk to AgentDescent. Skip it and the CLI
still works, but `agentdescent mcp` exits 3 and tells you to add it.

!!! warning "Python 3.9: the CLI, but no MCP server"
    The MCP SDK requires Python **3.10 or newer**, while AgentDescent itself
    supports 3.9. On 3.9 the `[mcp]` extra installs nothing — you get the CLI
    and the skill, and `agentdescent mcp` says why it cannot serve. The shipped
    `SKILL.md` falls back to the CLI verbs, so an agent can still drive it; it
    just calls `agentdescent plan` instead of a tool. `agentdescent doctor` and
    `agentdescent install` both say so up front. For the tool surface, use 3.10+.

Check that the command landed on your `PATH`:

```bash
agentdescent doctor
```

It prints which agent CLIs it can see, which provider keys are set, and where
runs will be stored, plus the AgentDescent `version` and the Python
`executable` it is running (`agentdescent --version` prints just the version).
Every line it marks as a problem is something that will make a real run fail —
but none of them stop step 2.

!!! warning "`command not found: agentdescent`"
    pip installed it somewhere not on your `PATH`. `python -m agentdescent.cli
    doctor` works regardless, and `python -m site --user-base` tells you which
    `bin/` to add. Hosts start the server as a **subprocess**, so it must be on
    the `PATH` of whatever launches your agent, not just your interactive shell.

## 2. Run the packaged demo offline

```bash
agentdescent demo
```

This builds a complete, real example in a temp directory and runs it:

* a skill directory `csv-total/` that totals one column of a CSV,
* whose `references/rules.md` says `COLUMN: id` — **wrong on purpose**, `id` is
  a row number,
* 12 cases with known totals,
* and an agent that is a local Python program reading that skill off disk.

Only the *model* is missing. Staging, the workspace, the ledger, the parallel
workers, the merge and the held-out gate are the same code a real run uses, so
what you watch is the actual loop:

```text
plan: 12 tasks, 4 rounds x 2 workers, up to 64 agent calls

running (20260905-134121-1c695b)...
  round  0  reward=1.000  +1/-0

held-out reward: 1.000   outcomes: {'committed': 1}
what it learned:  rules.md -> 'COLUMN: amount'
```

The last line is the point: nobody told it the column was `amount`. It ran the
skill, saw the answers were wrong, proposed an edit, and the edit survived the
held-out gate.

Your skill on disk is **untouched**. An evolved artifact only lands when you say
so:

```bash
agentdescent show  <run_id>              # the diff
agentdescent apply <run_id> --dry-run    # what it would write
agentdescent apply <run_id>              # write it, keeping a .bak- backup
```

When a run learns little, `show` also lists the worst rollouts (`samples`:
output vs gold) and the diffs the gate measured and refused (`rejected`: files,
short diff, rationale, outcome), so "the agent answered wrongly" is
distinguishable from "the reflector proposed nothing useful". Both are
diagnostics on disk; `record_rollouts: false` keeps neither.

Pass `--dir ./somewhere` to keep the demo's files around and poke at them.

## 3. Put it in your agent

```bash
agentdescent install claude-code    # or: dsh, codex, opencode
```

Then load it, and check it is actually there:

#### Claude Code

```bash
claude --plugin-dir ~/.agentdescent/plugins/claude-code
```

In the session, `/plugin` lists it. To confirm without a human — note the tool
name a **plugin** gets, `mcp__plugin_<plugin>_<server>__<tool>`:

```bash
echo "Call the agentdescent doctor tool and print its run_store field." | \
  claude -p --plugin-dir ~/.agentdescent/plugins/claude-code \
    --permission-mode acceptEdits \
    --allowedTools mcp__plugin_agentdescent_agentdescent__doctor
```

#### Codex

Skip `install` — Codex reads the same plugin package Claude Code does:

```bash
codex plugin marketplace add Birfy/agentdescent
codex plugin add agentdescent@agentdescent
codex mcp list        # agentdescent ... enabled
```

(`codex doctor` says "no MCP servers configured" even when a plugin-provided
server is live; it counts only `config.toml`. Trust `codex mcp list`.)

#### DeepSeek Harness

```bash
dsh --profile web --dump-config | grep agentdescent
```

Or the native plugin instead of the file edits `install` makes:

```bash
dsh plugin --profile web add link:$PWD/integrations/dsh-agentdescent
```

#### OpenCode

```bash
opencode mcp list     # ✓ agentdescent connected
```

Now say it in your own words, inside the agent:

> Evolve the skill at `~/.claude/skills/pdf-audit` against `eval/cases.jsonl`.

The agent writes the spec, shows you the plan and the call count, starts the run
in the background, and comes back with the diff before anything is applied. It
never applies without asking.

## 4. Your own evolution

Three things change from the demo: your directory, your cases, and a real agent
instead of the offline one.

```bash
agentdescent init ./my-skill --data cases.jsonl --out spec.json
agentdescent plan spec.json          # validates and prices it; runs nothing
agentdescent evolve spec.json --detach
agentdescent status
```

`init` writes the worker CLI it finds on `PATH` (`claude`, then `codex`, `dsh`,
`opencode`), the same view `agentdescent doctor` reports, so an OpenCode-only
machine gets `agent.ref: opencode`. Pass `--agent <ref>` to choose another. With
no worker CLI installed it writes `claude_code` as a placeholder and warns;
`plan` refuses a CLI that is not on `PATH`. For a host plugin (`kind: plugin`)
the `host` CLI is checked on its own: it comes from the plugin's format (a
`.claude-plugin/` tree needs `claude`), so an installed `agent` does not make
a missing host pass, and `plan`, `evolve` and MCP `start` refuse it by name.

The reflector (the model that proposes each edit) is an `openai_compatible`
call, and `init` never guesses its model or endpoint: it writes `<your-model>`
(and, for a non-OpenAI key, `<your-endpoint-url>`), names the provider key that is
set here, and `plan` refuses the spec until the placeholders are filled. Pass the
values at `init` time instead, for example with only `ARK_API_KEY` set:

```bash
agentdescent init ./my-skill --data cases.jsonl --out spec.json \
  --model doubao-seed-code --base-url https://ark.cn-beijing.volces.com/api/coding/v3
```

```json
"reflect": {"ref": "openai_compatible", "model": "doubao-seed-code",
            "api_key_env": "ARK_API_KEY",
            "default_base_url": "https://ark.cn-beijing.volces.com/api/coding/v3"}
```

`--key-env NAME` picks the key variable when several are set; with
`OPENAI_API_KEY` alone only `--model` is needed.

`cases.jsonl` is one JSON object per line, and it is the only thing you have to
build yourself:

```json
{"prompt": "How many pages fail the audit?", "gold": "3"}
{"prompt": "Which section is missing?", "gold": "Appendix B"}
```

Twelve to thirty cases is enough to start: at the default `held_out_frac` of 0.3,
12 is the smallest set that leaves 4 cases held out, and every acceptance
decision is measured on those. With fewer, `plan` says how many are held out in
its `notes` (and `status` repeats it for a started run); raise
`evolve.held_out_frac` or add cases. Without known answers there is no
reward, and without a reward there is nothing to descend — so this file, not the
model, is where the effort goes.

Two habits worth keeping from the first run:

* `agentdescent plan` before `evolve`, every time. It is free, and the call
  count it prints is the upper bound you are agreeing to.
* `--budget 5 --usd-per-call 0.02` on the first real run. A dollar budget
  needs a per-call price (the engine counts calls; a tool-using agent's price
  is not known in advance). Together they stop at five dollars rather than
  discovering the cost afterwards.

## Where to go next

* [Use it from your agent](plugins.md) — the full plugin surface: every spec
  field, the MCP tools, where runs live, and evolving the plugins themselves.
* [Set it up and test it](testing-guide.md) — one script that installs
  AgentDescent and wires up every agent CLI you have, then the whole test
  path in order.
* [Testing the plugins](testing-the-plugins.md) — how to verify each host, and
  what no test can confirm.
* [Quickstart — dataset to skill](quickstart-skill.md) — the same thing from
  Python, if you would rather write the twelve lines yourself.
