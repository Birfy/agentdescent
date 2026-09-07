# Set it up and test it

One page, in order: install, prove it works offline, wire it into your agents,
then drive it in plain language. Every command here was run against the real
CLIs; where something is known not to work, it says so rather than leaving you
to find out.

---

## 1. Install

One command does all of it — AgentDescent, and the wiring for every agent CLI
you already have:

```bash
git clone https://github.com/Birfy/agentdescent && cd agentdescent
bash scripts/setup-hosts.sh
```

Don't have the agent CLIs yet? Add `--with-clis` and it tries to npm-install
them first. `--dry-run` prints what it would do and changes nothing.

All four are public npm packages: `@anthropic-ai/claude-code`, `@openai/codex`,
`opencode-ai`, `@deepseek-ai/dsh`. The script prints npm's own error for
anything that fails, because the fixes are completely different:

| npm says | What it means | Fix |
|---|---|---|
| `ENOSPC` | the disk is full | free space; nothing else will work either |
| `ENOTEMPTY ... rename` | a half-installed copy is in the way | `rm -rf` the path npm printed, re-run |
| `EACCES` | npm's global prefix is not yours | see below |
| `ETIMEDOUT` / `ECONNREFUSED` | registry unreachable | check your proxy |

For `EACCES`, point npm somewhere you own (the script checks this up front):

```bash
npm config set prefix ~/.npm-global
export PATH="$HOME/.npm-global/bin:$PATH"    # add this to your shell rc
```

None of the CLIs are required — `agentdescent demo` runs with none of them
installed, and the script wires up whichever ones it finds.

The script is safe to re-run: config blocks it already owns are left alone
unless their content is out of date.

!!! warning "Two things that bite"
    **`agentdescent` must be on `PATH`** — hosts start it as a *subprocess*, so
    it must be on the PATH of whatever launches your agent, not just your
    interactive shell. The script warns if it isn't. `python3 -m
    agentdescent.cli` works regardless.

    **Python 3.10+ for the MCP server.** The `mcp` package requires it while
    AgentDescent supports 3.9. On 3.9 you get the CLI and the skill; the tools
    are unavailable and `agentdescent mcp` says exactly that.

Not in a PyPI release yet, so it installs from git. `pip install agentdescent`
gets 0.4.6, which has the engine and none of this.

## 2. Prove it works, offline

```bash
agentdescent demo
```

No key, no cost, ~10 seconds. It builds a skill whose `references/rules.md`
names the **wrong** column, twelve CSVs with known totals, and an agent that is
a local program reading that skill off disk — then runs the real loop:

```
plan: 12 tasks, 4 rounds x 2 workers, up to 64 agent calls
running (20260905-134121-1c695b)...
  round  0  reward=1.000  +1/-0

held-out reward: 1.000   outcomes: {'committed': 1}
what it learned:  rules.md -> 'COLUMN: amount'
```

Nobody told it the column was `amount`. Only the *model* is replaced here —
staging, the ledger, the parallel workers, the merge and the held-out gate are
the same code a real run uses. **If this fails, the problem is the install, not
your host.**

Your files are untouched until you say so:

```bash
agentdescent show  <run_id>            # the diff
agentdescent apply <run_id> --dry-run  # what it would write
agentdescent apply <run_id>            # writes it, keeping a backup
```

## 3. Check each host

The script already wired them. Confirm each one sees it:

| Host | Confirm with | Expect |
|---|---|---|
| **Claude Code** | `claude plugin validate ~/.agentdescent/plugins/claude-code` | `✓ Validation passed` |
| **OpenCode** | `opencode mcp list` | `✓ agentdescent connected` |
| **Codex** | `codex mcp list` | `agentdescent ... enabled` |
| **DSH** | `dsh --profile web --dump-config \| grep agentdescent` | two rows |

Then load it:

```bash
claude --plugin-dir ~/.agentdescent/plugins/claude-code   # Claude Code
```

Codex reads the same plugin format, so it can skip the file edits entirely:

```bash
codex plugin marketplace add Birfy/agentdescent
codex plugin add agentdescent@agentdescent
```

Known quirks, all measured:

* `codex doctor` reports "no MCP servers configured" even when a
  plugin-provided server is live. It counts only `config.toml`. Trust
  `codex mcp list`.
* `dsh` failing with `patch: entry "..." not found` means the rows were written
  without `- insert:`; `declares no dsh.bundle` means the package is inert.
  Neither should happen with a current install.
* Claude Code caches a failed MCP connection for ~15 minutes. A different
  `--plugin-dir` path is the quickest way to retry.

### Driving each host without a terminal session

The plain-language test below is the real one, but each host also has a
non-interactive form worth having in a script. All three were run end to end
against a real endpoint; each carries one thing that is not obvious.

**Claude Code.** `--permission-mode bypassPermissions` is refused by the safety
classifier when an automated caller asks for it, so name the tools:

```bash
P=mcp__plugin_agentdescent_agentdescent
claude -p "improve ./prompt.txt against ./cases.jsonl" \
  --plugin-dir ~/.agentdescent/plugins/claude-code \
  --allowedTools "Read,Glob,Grep,${P}__doctor,${P}__plan,${P}__start,${P}__status,${P}__show,${P}__apply"
```

`-p` is stateless. It stops for your yes as the contract requires; continue
with `claude -c -p "yes, start it"`.

**Codex.** The sandbox flag matters more than anything else here:

```bash
codex exec --skip-git-repo-check -s workspace-write \
  -c 'sandbox_workspace_write.network_access=true' \
  "improve ./prompt.txt against ./cases.jsonl, run it to completion"
```

Without `network_access=true` the run *starts*, burns a full round and then
dies with `URLError: [Errno 8] nodename nor servname provided`. The Seatbelt
sandbox blocks the network, the MCP server is a child of `codex`, and the
detached run inherits that — the same "a detached run inherits its launcher's
environment" trap as §6, wearing a different hat. Nothing about the message
points at the sandbox.

**DSH.** `dsh --profile headless "<task>"` answers one task and exits. Its
default provider is `deepseek-official`, so without `DEEPSEEK_API_KEY` you get
`MISSING_CREDENTIAL` before anything else happens. To point it at another
OpenAI-compatible endpoint, override the provider by id in a patch overlay —
a bare row is an override, which is why there is no `- insert:` here:

```yaml
# ark.patch.yml
- id: llm-deepseek
  config:
    baseURL: https://your-endpoint/v3
    apiKeyEnv: YOUR_KEY_VAR
```

```bash
dsh --profile headless --patch ./ark.patch.yml "improve ./prompt.txt against ./cases.jsonl"
```

The `headless` profile installs its own dependencies on first boot through
corepack, which must be new enough for the pnpm the profile pins — the same
corepack requirement §2's dsh test skips on.

## 4. Drive it in plain language

This is the actual test. Make a target and some examples:

!!! tip "No cases yet? Skip the next block and just ask."
    Drafting them is step one of the procedure, not a prerequisite for it — the
    skill's own description says so. Point it at a bare `prompt.txt` and it
    writes 8–20 cases into `eval/cases.jsonl` and **stops for you to read
    them**. Measured on a support-agent prompt with no data and no spec: twelve
    cases, whose `gold` came back as behavioural rubrics ("acknowledges the
    frustration without grovelling; does not promise a refund; asks for the
    invoice number") rather than keyword strings, followed by the observation
    that `contains` cannot score that and an offer of `{"cmd": "./grade.sh"}`
    instead. Reviewing those cases is the one step you cannot skip: the search
    optimises whatever you scored it on, very efficiently.


```bash
mkdir -p /tmp/try && cd /tmp/try
cat > prompt.txt <<'EOF'
Answer the question.
EOF
python3 - <<'EOF'
import json
rows = [("What is the capital of France?", "FINAL Paris"),
        ("What is the capital of Japan?", "FINAL Tokyo"),
        ("What is the chemical symbol for gold?", "FINAL Au"),
        ("What is the capital of Italy?", "FINAL Rome"),
        ("What is the chemical symbol for iron?", "FINAL Fe"),
        ("What is the capital of Egypt?", "FINAL Cairo"),
        ("What is the chemical symbol for sodium?", "FINAL Na"),
        ("What is the capital of Canada?", "FINAL Ottawa"),
        ("What is the chemical symbol for potassium?", "FINAL K"),
        ("What is the capital of Spain?", "FINAL Madrid"),
        ("What is the chemical symbol for silver?", "FINAL Ag"),
        ("What is the capital of Greece?", "FINAL Athens")]
open("cases.jsonl", "w").write(
    "\n".join(json.dumps({"prompt": p, "gold": g}) for p, g in rows) + "\n")
EOF
```

The prompt scores **zero** — every gold answer is prefixed `FINAL` and the
prompt never says so. Nothing states the rule anywhere; it is discoverable only
from the failures.

Now open your agent in that directory and say:

> Improve `/tmp/try/prompt.txt` against the examples in `/tmp/try/cases.jsonl`.

What should happen, in order — this is the skill's whole contract:

1. it runs **`doctor`** and reports anything missing;
2. it writes a spec and runs **`plan`**, and **shows you the call count before
   starting**;
3. it waits for your yes;
4. it starts, polls, and reports what changed;
5. it **does not apply** without asking, and names the backup path when it does.

The run typically ends at reward `1.0` having learned something equivalent to
*"Start your final response with FINAL followed by the answer."* It is a search,
not a guarantee — of five runs of this exact task, four reached 1.0 in the first
round and one proposed nothing at all, for the reason in the next paragraph.

**If it finishes with reward 0 and `outcomes: {}`, read `considered` in the
rounds, not the reward.** `considered: 0` every round means no evidence card
was ever proposed — the reflector could not run, which is a configuration
problem, not a failed search. The commonest cause is a spec with
`"reflect": {"ref": "host_model"}` resumed from a shell instead of from the
agent: `host_model` borrows *the host that started the run*, and a shell is not
one. `evolve` and `resume` warn about this now; a run that predates the warning
just goes quiet.

### Things worth trying deliberately

| Try saying | What should happen |
|---|---|
| "improve this skill" with **no examples** | it offers to draft 8–20 cases and **stops for you to check them** |
| "use 60 rounds and 32 workers" | it quotes the cost **before** starting, even if you said not to ask |
| "apply it" | it names the file it would overwrite, then backs up and reports the path |
| "stop the run" | `cancel` kills the workers too, and it tells you what you lose |
| ask it to evolve **a plugin** | it sets `host`, and refuses to start a nested run from inside a worker |

!!! note "Add `reflective_merge`, or the workers are not merging"
    `prompt.txt` is a one-key artifact, and under the **shipped** conflict rule
    its worker proposals contradict by construction: they collapse to a single
    candidate and no fusion is ever built. Four workers there are per-round
    best-of-N selection. `plan` warns when a spec is in that shape.

    Installing the reflective pair is what makes it a merge, and it is what you
    want by default:

    ```json
    "policies": {
      "reflective_merge": {
        "ref": "reflective_merge",
        "complete": {"ref": "openai_compatible", "model": "your-model"}
      }
    }
    ```

    Measured on this exact scenario, with and without. Default: four proposals,
    three dropped as conflicts, `n_candidates: 1`, `single-candidate`, no
    fusion. With `reflective_merge`: `conflicts_dropped: 0`, `n_candidates: 4`,
    `fused: 1`, and the ledger reads
    `merge synth(w0:value:1+w1:value:1+w2:value:1+w3:value:1)` -- all four
    workers synthesised into the committed candidate. Same on a multi-file
    skill directory, where one round also came back `synthesis-failed` and fell
    back to the best single, which is the fallback working rather than a fault.

    Read `contested` in `fusion_stats()`, or `fusion_trials` in `result.json`,
    rather than inferring the merge from the worker count.

## 5. Using a real model

Two ways, neither needing a spec change beyond one field.

**A provider key.** For an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-endpoint/v1     # if not OpenAI itself
```

`doctor` reports the base URL back, because a non-OpenAI endpoint has its own
model names — the agent will **ask you which model** rather than guessing, and
that is correct behaviour, not a stall.

**The host's own model**, no key at all:

```json
"reflect": {"ref": "host_model"}
```

`host_model` uses MCP sampling where the host supports it and otherwise runs
that host's CLI with your real configuration. Measured: **no host implements
sampling today** — Claude Code declares `roots` and `elicitation`, OpenCode
`roots`, dsh nothing — so in practice it takes the CLI route. `start` reports
`host_model_route` so you can see which one you got.

!!! warning "Worker isolation and logins"
    A worker runs with the host's config directory redirected, so **a CLI you
    signed into interactively is not signed in for the run**. Use
    `"isolate": false` to hand it your real setup, or supply a provider key.
    `plan` warns about this now. One measured trap: `codex` ignores both
    `OPENAI_API_KEY` and `OPENAI_BASE_URL` — it called `api.openai.com`
    unauthenticated — so for codex, `"isolate": false` is the only route.

## 6. When something is wrong

```bash
agentdescent doctor        # what is missing, in one screen
agentdescent status        # every run, newest first
agentdescent show <id>     # the diff, and the file apply would overwrite
agentdescent serve         # a read-only panel on http://127.0.0.1:8787/
```

Run logs are at `~/.agentdescent/runs/<id>/log.txt`, and the spec that produced
them sits beside it, re-runnable from anywhere.

Two failure modes worth recognising:

* **A run fails with `Connection refused` partway through.** A detached run
  inherits the environment of whatever launched it, including a proxy. If that
  process was short-lived — a headless `claude -p`, say — the run loses its
  network when it exits. `agentdescent resume <id>` picks it up on the same
  ledger from somewhere longer-lived.
* **A run finishes having proposed nothing** — reward 0, `outcomes: {}`, and
  `considered: 0` in every round of `rounds.jsonl`. The reflector never
  produced a card. Check `start`'s reply for `host_model_available` /
  `host_model_unavailable`, and the warnings `evolve` and `resume` print. A
  `host_model` spec resumed from a shell does exactly this: it has no host to
  borrow from, so every proposal raises and the rounds pass in silence.

## 7. Running the test suite

```bash
pip install -e ".[dev,mcp]"
pytest tests/test_evolvespec.py tests/test_runstore_cli.py tests/test_mcp.py \
       tests/test_plugin_runner.py tests/test_integrations.py \
       tests/test_run_panel.py tests/test_demo.py tests/test_host_sampling.py
```

That is the whole plugin surface, offline, in about a minute. Two of them reach
further when the tools are there and skip cleanly when not: the dsh tests boot
real `dsh`, and the client-bundle test needs node with react.

[Testing the plugins](testing-the-plugins.md) covers what those tests cannot
reach and how to check each host by hand.
