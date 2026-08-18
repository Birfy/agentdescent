# DeepSeek Harness — evolving your own harness

*Bundle:* [`plugin/`](https://github.com/Birfy/agentdescent/tree/main/plugin)
· *Design record:* [design-dsh-plugin.md](design-dsh-plugin.md)

`dsh` makes every layer of the agent a swappable plugin, which is the first time
an agent's **parameters** — its skills, prompt sections, tool set, a plugin's own
source — have been addressable rather than buried in a fixed stack. This page is
how you point an optimiser at them.

```sh
dsh plugin --profile web add dsh-plugin-agentdescent
pip install agentdescent
```

Two decisions are yours and nothing invents them for you: **which tasks**, and
**what counts as better**. Everything below is those two.

## 1. Which tasks

A dataset is JSONL — one object per line, `prompt` required, everything else
kept as metadata:

```json
{"prompt": "how many orders shipped last week?", "gold": "shipped_at"}
{"prompt": "total revenue by region", "gold": "group by"}
```

**A line that is not JSON is taken as a bare prompt**, so a plain list of
questions is a valid dataset. That is deliberate: when your objective is a
command rather than an answer, a list of questions is exactly what you have.

```yaml
    datasets:
      - { name: sql-regressions, file: /home/you/data/sql.jsonl }
```

Three other sources exist. `transcripts` turns this harness's own past turns
into tasks — off by default, because replaying a turn sends its content to the
model provider again. Inline `tasks:` covers a handful not worth a file. And the
Python runner takes `--tasks q.jsonl` directly.

## 2. What counts as better

Four kinds, strongest signal first. Pick the highest one you can actually
supply — the ladder exists because most people have an objective, it just is not
a benchmark.

| kind | you have | example |
|---|---|---|
| `gold` | expected answers | `{ kind: gold, match: contains }` |
| `command` | a check that already exists | `{ kind: command, run: ["make", "check"] }` |
| `judge` | a description of good | `{ kind: judge, rubric: "..." }` |
| `replay` | neither, but a log of what happened | `{ kind: replay }` |

```yaml
    objectives:
      - { name: right-answer, kind: gold, match: contains }
      - { name: runs-on-my-db, kind: command,
          run: ["psql", "--quiet", "-f", "{output}"] }
      - { name: house-style, kind: judge, rubric: |
            snake_case table names. Never SELECT *.
            Never touch anything under migrations/. }
```

`{output}` in a command becomes a path to that rollout's answer; the same text
also arrives on stdin. The command runs **without a shell**, so `|` and `>` are
literal — ask for one explicitly with `["bash", "-lc", "..."]`.

### Combining them

`all` is an **and**, and takes the minimum rather than the mean — a candidate
that nails one requirement and fails another is not a partial success:

```yaml
      - { name: correct-and-cheap, kind: all, efficiency: true,
          of: [{ kind: gold }, { kind: command, run: ["make", "check"] }] }
```

`efficiency: true` rewards reaching the same result in fewer steps. It
**multiplies a hard pass gate** rather than adding to it, so cheap can never buy
off wrong. That is the one composition mistake with no visible symptom: a
weighted sum lets the aggregator accept artifacts that are faster and worse
while the held-out score still climbs.

### Two rules the scorers follow

- A **timeout** scores 0 rather than raising. A check that hangs on this output
  *is* a failing check, and raising would drop the sample — biasing the run
  toward candidates that happen not to hang the checker.
- A **missing executable** raises. That is a configuration mistake, and scoring
  it 0 makes every rollout fail identically, which is indistinguishable from a
  model that cannot do the task.

Everything is validated when the plugin mounts. A misspelled kind stops it
loading and names the row, because a run that quietly measured the wrong
objective is not something you can detect from its output.

## 3. What gets evolved

| artifact | what it is | layer |
|---|---|---|
| `skill:<name>` | one skill — refinement | L2, merges on score |
| `skills:<root>` | a whole skill root — **add, edit, retire** | L2 |
| `prompt:<name>` | a system-prompt section | L2, commits at a turn boundary |
| `preset:<name>` | a persona plus a tool mask | L1, waits for you |
| `plugin:<pkg>` | a dsh plugin's own source | L1, waits for you |

```yaml
    skills:    [{ name: sql,  dir: /home/you/.dsh/skills/sql }]
    libraries: [{ name: user, dir: /home/you/.dsh/skills }]
    prompts:   [{ name: house-style, dir: /home/you/.dsh/prompts/house-style }]
    presets:   [{ name: careful, dir: /home/you/.dsh/presets/careful }]
    plugins:   [{ name: dsh-plugin-mine, dir: /home/you/src/dsh-plugin-mine }]
```

A **library** is a different artifact from a skill, not a bigger one: with the
root as the state, a run can add a skill nobody wrote. That is the only setting
in which evolution discovers rather than refines.

A **prompt section** is in every request, so its commit waits until no agent is
mid-turn — swapping the request prefix costs every live session's KV cache, and
that is worth a few seconds' wait.

A **preset** is `PERSONA.md` plus `TOOLS.md` (one tool name per line, `-name`
denies). Publishing one changes the persona immediately; the tool mask applies
per agent, because dsh refuses a process-global restriction — it would mask
every agent, including the ones not using the preset.

## 4. Running it

```
/evolve                        what can be evolved, scored how, on which tasks
/evolve skill:sql              start a run
/evolve status <run>           where it is
/evolve pending                L1 changes waiting for you
/evolve approve <id>           publish one
```

`/evolve` **refuses to guess** when several objectives are registered and you
name none — naming the wrong one produces a confidently wrong artifact.

Set `idleRuns` to have the harness start these itself once everything has been
quiet, and stop the moment you come back. Off by default.

## Without the plugin

The same two decisions, from Python, with no harness config:

```sh
python -m examples.dsh.evolve_dsh_skill --skill sql --tasks q.jsonl --score contains
python -m examples.dsh.evolve_dsh_skill --skill sql --tasks q.jsonl --check "make check"
python -m examples.dsh.evolve_dsh_skill --library --tasks q.jsonl        # the whole root
python -m examples.dsh.evolve_dsh_plugin --plugin dsh-plugin-mine --tasks q.jsonl \
    --entrypoint "dsh --profile web -p"
```

`--dry-run` resolves everything and calls nothing, which is the part worth
checking before spending tokens. `--list` shows what this machine actually has.
