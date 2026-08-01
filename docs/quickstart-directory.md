# Quickstart — evolve a directory

Ten minutes from a skill folder on disk to an improved one, with a real agent
doing the work. No API key needed for the offline run.

The full reference is [evolving a directory](directory-evolution.md); this is the
path through it.

## 1. Run it offline first

```bash
python -m examples.skill_dir_evolution
```

```
Skill    : /tmp/agentdescent-example-…/csv-total
Agent    : offline
Reflector: stub
Start    : rules.md = 'COLUMN: id'  (wrong column)

round   0  reward=1.000 on 5  size=2  +1/-0
round   0  target_reward=0.98 reached, stopping

final reward : 1.000
outcomes     : {'committed': 1}

evolved rules.md:
COLUMN: amount
```

The skill directory starts **wrong** (it names the `id` column), the agent
therefore totals the wrong column, and the loop fixes `references/rules.md`. The
"agent" here is a real subprocess bound to a workspace — it opens the file — so
the staging, layout, overlay and isolation logic all run. Only the reflector is a
stub.

## 2. Point it at your own directory

```python
from agentdescent import evolve_skill_dir
from agentdescent.agents import claude_code, openai_compatible

result = evolve_skill_dir(
    "~/.claude/skills/pdf-audit",          # your directory
    rows,                                   # your data
    agent=claude_code(extra_args=["--permission-mode", "acceptEdits"]),
    reflect_with=openai_compatible(model="deepseek-v4-flash"),
    prompt="question", gold="answer", score="contains")

print(result.final_reward, result.outcomes())
```

Three decisions are yours and the rest has a default: **your directory**, **your
data**, **which agent runs it**. A cheap `reflect_with` model behind an expensive
agent is usually the right trade.

The run **never writes to your directory**. Installing the result is a separate,
explicit call:

```python
plan = result.write_to("~/.claude/skills/pdf-audit", dry_run=True)
# {'written': [...], 'extra': ['notes.md'], 'deleted': [], 'backup': []}

result.write_to("~/.claude/skills/pdf-audit")     # backs up to <path>.bak-0 first
```

## 3. What happens per rollout

```
your directory  --load_tree-->  {"SKILL.md": "...", "references/rules.md": "..."}
                                          |
                             a fresh workspace per rollout
                                          v
        /tmp/agentdescent-ws-xxxx/.claude/skills/pdf-audit/SKILL.md
        /tmp/agentdescent-ws-xxxx/<the task's fixtures>
                                          |
                     claude_code().in_workspace(ws)(prompt)
                                          v
                    answer -> reward -> reflection -> a multi-file diff
```

State keys are file paths, so two workers editing different files **fuse** and
two editing the same file are **resolved** on held-out score — the same
[aggregator](aggregator.md) as every other strategy, with no special cases.

## 4. Staging the task's own inputs

A task usually needs data of its own beside the skill:

```python
from agentdescent import Task

Task(id="t1", prompt="What is the total?",
     meta={"gold": "417", "fixtures": {"data.csv": "id,amount\n1,200\n2,217\n"}})
```

Anything in `meta["fixtures"]` is written into the workspace next to the tree.

## 5. The three entry points

```python
evolve_skill_dir(path, data, agent=...)          # a skill folder      — L2
evolve_agent_dir(path, data, agent=...)          # subagent definitions — L1
evolve_agent_code(path, data, entrypoint=[...])  # code that executes   — L1 + tests
```

They differ in [governance](governance.md) and in what guards them. Agent code
runs behind a **frozen test suite the candidate cannot rewrite**:

```python
result = evolve_agent_code(
    "./my-agent", rows,
    entrypoint=["python", "main.py"],
    test_cmd=["python", "-m", "pytest", "-q"],
    frozen=["tests/**", "conftest.py"],          # the default
    reflect_with=openai_compatible(model="deepseek-v4-flash"))
```

Frozen paths are enforced twice: the proposal filter stops the reflector editing
them, and the runner overlays the pristine copies after materialisation so the
*candidate* cannot rewrite them at run time either.

!!! danger "Isolation, not a sandbox"
    Candidate code runs in a throwaway workspace with a trimmed environment
    (`HOME` and `TMPDIR` point inside it) under a hard timeout — but as your
    user, with your network. Use a container for anything you would not run by
    hand.

## 6. Two things to check on a real run

**Is the skill actually being used?** Run one round against an empty skill
directory as a control. If the score does not move, you are measuring the model's
prior knowledge, not your skill.

**What is it costing?** One rollout is one agent invocation. The three
`evolve_*_dir` functions already default `self_verify=False` and
`cheap_eval_tasks=4` — the plain-engine defaults are right for text and expensive
here. The full accounting is in the
[cost model](directory-evolution.md#cost-the-first-order-design-constraint).

## Next

* [Evolving a directory](directory-evolution.md) — the complete reference
* [Design record](design-directory-evolution.md) — why it is built this way
* [Strategies](strategies.md) — `FileTree` alongside the text strategies
