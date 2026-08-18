# dsh-plugin-agentdescent

A training loop for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

dsh makes every layer of the agent a swappable plugin, which is the first time
an agent's *parameters* — its skills, prompt sections, tool set, subagent
presets, even a plugin's own source — have been addressable rather than buried
in a fixed stack. This bundle adds the third term:

> **model + harness + descent = an agent that gets better.**

The optimiser is [AgentDescent](https://github.com/Birfy/agentdescent): N
workers propose diffs in parallel, and an aggregator resolves conflicts, fuses
complements and accepts on a Beta posterior over held-out reward.

## Install

```sh
dsh plugin --profile web add dsh-plugin-agentdescent
pip install agentdescent            # the optimiser itself
```

Then point it at something evolvable in your profile's `cordis.patch.yml`:

```yaml
- id: evolution-agentdescent
  name: dsh-plugin-agentdescent
  inject: [evolution, agents, skills, commands, tools]
  config:
    cwd: /path/where/agentdescent/imports/from
    skills:
      - { name: sql, dir: /home/you/.dsh/skills/sql }
    plugins:
      - { name: dsh-plugin-mine, dir: /home/you/src/dsh-plugin-mine }
```

## Saying what to train on, and what counts as better

These are the two decisions that are yours, so neither needs TypeScript.

**Which tasks.** A JSONL file — `{"prompt": ..., "gold": ...}` per line. A line
that is not JSON is taken as a bare prompt, so a plain list of questions is a
valid dataset:

```yaml
    datasets:
      - { name: sql-regressions, file: /home/you/data/sql.jsonl }
```

**What counts as better.** Four kinds, strongest signal first — pick the highest
one you can actually supply:

```yaml
    objectives:
      # A: you have expected answers
      - { name: right-answer, kind: gold, match: contains }

      # B: you have a check that already exists
      - { name: runs-on-my-db, kind: command,
          run: ["psql", "--quiet", "-f", "{output}"] }

      # B: you can describe what good looks like
      - { name: house-style, kind: judge, rubric: |
            snake_case table names. Never SELECT *.
            Never touch anything under migrations/. }

      # C: you have neither, but the log has what happened last time
      - { name: better-than-before, kind: replay }
```

`{output}` in a command becomes a path to that rollout's answer; the same text
also arrives on stdin. Stack objectives with `all` — the score is the
**minimum**, because a list of requirements is an "and", not an average:

```yaml
      - { name: correct-and-cheap, kind: all, efficiency: true,
          of: [{ kind: gold }, { kind: command, run: ["make", "check"] }] }
```

`efficiency: true` rewards reaching the same result in fewer steps. It
multiplies a hard pass gate rather than adding to it, so being cheap can never
buy off being wrong.

Everything here is validated when the plugin mounts. A misspelled kind or a
missing dataset stops it loading and names the row — because a run that quietly
measured a different objective than the one you meant is not something you can
detect afterwards.

## Use

```
/evolve                          what can be evolved, scored how, on which tasks
/evolve skill:sql                start a run
/evolve status <run>             where it is
/evolve pending                  L1 changes waiting for you
/evolve approve <id>             publish one
```

The model can start its own run with `evolve_start` — it is the participant
that knows a class of task keeps going wrong, usually before anyone else does.

## How it is put together

| Piece | Role |
|---|---|
| `ctx.evolution` | the capability seam — Service Definition, no optimiser in it |
| the engine | drives a Python sidecar over a bidirectional NDJSON-RPC bridge |
| artifacts | `skill:<name>` (L2) and `plugin:<pkg>` (L1) |
| rollouts | run **inside this harness**, one child agent per candidate |
| scorers | gold ▸ command ▸ rubric ▸ replay-comparison |
| the queue | L1 changes wait for a person |

Four things are worth knowing before you rely on it.

**Rollouts measure this harness, not a replica.** A candidate is registered
against one child agent's own context, so it runs with your real tools,
sandbox, approval policy and prompt assembly. A registration scoped to
`agent.ctx` wins its name in that agent alone, which is also why N workers can
hold N different candidates in one process.

**The engine never talks to a model.** Every completion it needs — reflection
included — comes back through the harness, so credentials, the token budget and
the sandbox have exactly one home.

**Objectives are yours.** Bring gold answers and they are scored directly;
bring a command and its exit code is the judge; bring a rubric and a model
applies it. Only if you bring none does it fall back to comparing against what
the log says happened last time. Nothing here invents an objective for you, and
`/evolve` refuses rather than guessing when the choice is ambiguous.

**Harness-shaped changes wait for you.** Skills merge on held-out score.
Anything at L1 — a plugin's source, a preset — is staged in `/evolve pending`
until you approve it. When a plugin is the artifact, its `cordis.patch.yml`,
`package.json` and tests cannot be written at all: those decide which rows
mount, what runs at install time, and what the tests check.

## Development

```sh
npm ci && npm run typecheck && npm test && npm run build
```

The last test spawns the real Python engine and drives a whole run through the
bridge; it skips itself when `agentdescent` is not importable, and CI installs
it so that it does not.

Design record: [`docs/design-dsh-plugin.md`](../docs/design-dsh-plugin.md).
