# Testing the plugins

Four hosts, a native dsh plugin and a web panel, and none of it is exercised by
importing the library. This page is how to check each piece — what runs with no
credentials, what needs a host installed, and what only a human can confirm.

The short version:

```bash
pip install -e ".[dev,mcp]"
pytest tests/test_evolvespec.py tests/test_runstore_cli.py tests/test_mcp.py \
       tests/test_plugin_runner.py tests/test_integrations.py \
       tests/test_run_panel.py tests/test_stop_when.py
```

That is the whole plugin surface, offline, in about a minute. Everything below
is what those tests cannot reach.

## 1. What the test suite already covers

No credentials, no host installed, no network.

| File | What it pins |
|---|---|
| `test_evolvespec.py` | every `kind` composes the `evolve()` call the quickstarts write; refs resolve inside the allowlist; a spec runs end to end with a stub agent |
| `test_runstore_cli.py` | a detached run really detaches, reports, resumes and cancels (with its workers); every CLI verb, including relative paths surviving the process boundary |
| `test_mcp.py` | the tool bodies, the nested-run stub, and — when `mcp` is installed — the real server's tool and resource registration |
| `test_plugin_runner.py` | `kind: plugin` against a stub host CLI: isolated host home, the validate gate, frozen hooks surviving a candidate that rewrites them, `env_passthrough` |
| `test_integrations.py` | every host manifest, the checked-in plugin packages matching their renderers, and the skill naming only tools/kinds/verbs that exist |
| `test_run_panel.py` | the HTTP panel, and that it refuses path traversal and non-loopback CORS |

Two of them reach further when the tools are present, and skip cleanly when not:

```bash
# drives the dsh plugin through dsh's own isSkillName validator
which dsh && pytest tests/test_integrations.py -k real_registry -q

# loads the browser bundle the way dsh's module loader does and renders it
mkdir -p /tmp/react-probe && cd /tmp/react-probe && npm init -y && npm i react react-dom
AGENTDESCENT_NODE_PROBE=/tmp/react-probe pytest tests/test_integrations.py -k client_bundle -q
```

## 2. Check your own machine

```bash
agentdescent doctor
```

Lists which agent CLIs are on `PATH`, which provider keys are set, whether a
container engine and the `mcp` package are present, and where the run store is.
Every "problem" line it prints is something that will make a real run fail.

## 3. The offline end-to-end run

This is the fastest way to see the whole loop without spending anything. It
uses a stub agent that really reads the skill off disk, so staging, the
workspace, the ledger, the merge and `write_to` all run.

```bash
mkdir -p /tmp/ad-demo/csv-total/references && cd /tmp/ad-demo
cat > csv-total/SKILL.md <<'EOF'
# csv-total
Read `references/rules.md` for which column of `data.csv` to total.
Reply with only the number.
EOF
echo "COLUMN: id" > csv-total/references/rules.md      # wrong on purpose

python - <<'EOF'
import json, random
rng = random.Random(0)
rows = []
for _ in range(10):
    amounts = [rng.randint(1, 99) for _ in range(4)]
    csv = "id,amount\n" + "\n".join(f"{i+1},{a}" for i, a in enumerate(amounts))
    rows.append({"prompt": "What is the total?", "gold": str(sum(amounts)),
                 "fixtures": {"data.csv": csv}})
open("cases.jsonl", "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
EOF

agentdescent init ./csv-total --data cases.jsonl --out spec.json
# point it at an offline stub instead of a real agent: see docs/plugins.md
agentdescent plan spec.json
agentdescent evolve spec.json --detach
agentdescent status
agentdescent show <run_id>          # the diff: COLUMN: id -> COLUMN: amount
agentdescent apply <run_id> --dry-run
```

A successful run ends `done`, `best=1.000`, and `show` prints a diff that fixes
`rules.md`. `apply` without `--dry-run` writes it back and leaves a backup at
`csv-total.bak-0`.

## 4. Each host

`install` is idempotent and `--dry-run` prints what it would write, so it is
safe to inspect first. Use `--home` to keep a probe out of your real setup.

```bash
agentdescent install dsh --dry-run --home /tmp/probe
```

### Claude Code

```bash
agentdescent install claude-code
claude plugin validate ~/.agentdescent/plugins/claude-code     # structural check
claude --plugin-dir ~/.agentdescent/plugins/claude-code        # interactive
```

Headless, which is what a CI check would run — note the tool names a **plugin**
gets (`mcp__plugin_<plugin>_<server>__<tool>`), and that `--allowedTools` is
variadic so the prompt goes on stdin:

```bash
echo "Call the agentdescent doctor tool and print its run_store field." | \
  claude -p --plugin-dir ~/.agentdescent/plugins/claude-code \
    --permission-mode acceptEdits \
    --allowedTools mcp__plugin_agentdescent_agentdescent__doctor
```

Expected: the run-store path. If it says the server failed to connect, the
usual cause is `agentdescent` not being on `PATH` for the process that started
Claude Code, or the `[mcp]` extra missing — `agentdescent mcp` then exits 3 and
says so. Claude Code caches a failed connection for ~15 minutes; a different
`--plugin-dir` path is the quickest way to retry.

### DeepSeek Harness

Either the files or the native plugin, not both.

```bash
agentdescent install dsh
dsh --profile web --dump-config | grep -n agentdescent      # two rows compose
```

```bash
dsh plugin --profile web add link:$PWD/integrations/dsh-agentdescent
dsh --profile web --dump-config | grep -n agentdescent
```

Expected: rows `mcp-agentdescent` (and `dsh-agentdescent` for the plugin) under
a `# == ...` heading. **`patch: entry "..." not found` means the rows were
written without `- insert:`** — a dsh patch file overrides by id, it does not
append. **A `declares no dsh.bundle` warning on install** means the package is
inert; it needs `"dsh": {"bundle": {"patch": "./cordis.patch.yml"}}`.

### Codex

```bash
agentdescent install codex
codex mcp list                    # agentdescent ... enabled
```

### OpenCode

```bash
agentdescent install opencode
opencode mcp list                 # ✓ agentdescent connected
```

`connected` here really did start the server; `Executable not found in $PATH`
means `agentdescent` is not on the `PATH` OpenCode inherited.

## 5. The MCP server on its own

Independent of any host, using the protocol SDK as a client:

```python
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="agentdescent", args=["mcp"], env={**os.environ})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print(sorted(t.name for t in (await s.list_tools()).tools))
            print((await s.call_tool("doctor", {})).content[0].text[:200])

asyncio.run(main())
```

Expected: the eight tools (`apply cancel doctor plan resume show start status`)
and a doctor report.

## 6. The web panel

```bash
agentdescent serve                     # http://127.0.0.1:8787/
curl -s localhost:8787/api/runs | head -c 200
```

The page lists runs and refreshes itself. In dsh's web UI the plugin adds an
`evolve` action to the session header with the live count; clicking it shows the
same list. If the panel says *"No run panel"*, `agentdescent serve` is not
running.

## 7. What no test here can tell you

Be aware of these rather than assume they are covered.

* **Whether the dsh panel looks right.** The bundle is checked for loading,
  registering into `conversation.session.header.actions` and rendering HTML
  through React. Whether it is *legible in the actual UI* needs a browser and a
  dsh session.
* **A full Codex or DSH session.** Both were verified at the config layer
  (`codex mcp list`, `dsh --dump-config`) because neither had credentials here.
  Claude Code and OpenCode were driven to the point of tools returning data.
* **`kind: plugin` against a real dsh plugin repo.** The host table is
  exercised with a stub host CLI; the `pnpm install && pnpm build` chain in the
  dsh row has not been run against a real Cordis package.
* **Cost.** Every number `plan` reports is an upper bound on *calls*; the
  dollar figure only appears when you give it a per-call price. Run a real
  evolution with `--budget` and a small `rounds` the first time.
