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
from agentdescent import FileTree, evolve, load_tree, scorer, tree_reflector, tree_runner
from agentdescent.agents import claude_code, openai_compatible
from agentdescent.governance import SKILL_BLAST_RADIUS

path = "~/.claude/skills/pdf-audit"                      # your directory
tree = load_tree(path)                                   # -> {"SKILL.md": ..., ...}
strategy = FileTree(tree, max_files_per_diff=2)          # file paths are the state keys
run = tree_runner(claude_code(extra_args=["--permission-mode", "acceptEdits"]),
                  layout="claude_skill", name="pdf-audit",
                  overlay=strategy.frozen_files(tree))   # a fresh workspace per rollout

result = evolve(tasks, scorer("contains"), run=run, strategy=strategy,
                propose=tree_reflector(openai_compatible(model="deepseek-v4-flash"),
                                       strategy=strategy),
                artifact_id="pdf-audit", blast_radius=SKILL_BLAST_RADIUS,   # L2
                self_verify=False, cheap_eval_tasks=4,   # a rollout is a real agent call
                rounds=6, n_workers=4, max_concurrency=4, held_out_frac=0.3)

print(result.final_reward, result.outcomes())
```

Three decisions are yours: **your directory**, **your data** (`tasks`, from
[`tasks_from`](quickstart-skill.md#the-pieces) or ready-made `Task`s), **which
agent runs it**. A cheap reflection model behind an expensive agent is usually
the right trade. The four lines before `evolve()` are the whole directory
adapter: `load_tree` reads the folder, `FileTree` makes paths the keys, and
`tree_runner` / `tree_reflector` are the actor pair — the same
[`run` / `propose` contract](evolution.md#bring-an-agent-you-already-have) as
every other run.

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

## 5. The three workloads

Same call, three settings — governance and what guards the tree are the only
differences:

| workload | `blast_radius=` | runner | reward |
|---|---|---|---|
| a skill folder | `SKILL_BLAST_RADIUS` (L2) | `tree_runner(layout="claude_skill")` | `scorer(…)` |
| subagent definitions | `HARNESS_BLAST_RADIUS` (L1 — every merge also passes the oracle) | `tree_runner(layout="claude_agent")` | `scorer(…)` |
| code that executes | `HARNESS_BLAST_RADIUS` + a test gate | `code_runner(entrypoint, test_cmd=…)` | `gated_reward(scorer(…))` |

Agent code runs behind a **frozen test suite the candidate cannot rewrite**:

```python
from agentdescent import code_runner, gated_reward
from agentdescent.governance import HARNESS_BLAST_RADIUS

tree = load_tree("./my-agent")
strategy = FileTree(tree, frozen=["tests/**", "conftest.py"], max_files_per_diff=2)
run = code_runner(["python", "main.py"], name="my-agent",
                  test_cmd=["python", "-m", "pytest", "-q"],
                  overlay=strategy.frozen_files(tree))     # pristine tests, every rollout

result = evolve(tasks, gated_reward(scorer("contains")), run=run, strategy=strategy,
                propose=tree_reflector(openai_compatible(model="deepseek-v4-flash"),
                                       strategy=strategy, context_files=("**/*.py",)),
                artifact_id="my-agent", blast_radius=HARNESS_BLAST_RADIUS,
                self_verify=False, cheap_eval_tasks=4)
```

Frozen paths are enforced twice: the strategy's `frozen=` stops the reflector
editing them, and the runner's `overlay=` puts the pristine copies back after
materialisation so the *candidate* cannot rewrite them at run time either.
`gated_reward` scores a failed gate as zero, and the failure text is what the
reflector sees.

!!! danger "Isolation, not a sandbox"
    Candidate code runs in a throwaway workspace with a trimmed environment
    (`HOME` and `TMPDIR` point inside it) under a hard timeout — but as your
    user, with your network. Use a container for anything you would not run by
    hand.

## 6. Two things to check on a real run

**Is the skill actually being used?** Run one round against an empty skill
directory as a control. If the score does not move, you are measuring the model's
prior knowledge, not your skill.

**What is it costing?** One rollout is one agent invocation. Pass
`self_verify=False` and `cheap_eval_tasks=4`, as every block on this page does —
the plain-engine defaults are right for text and expensive here. The full
accounting is in the
[cost model](directory-evolution.md#cost-the-first-order-design-constraint).

## Next

* [Evolving a directory](directory-evolution.md) — the complete reference
* [Design record](design-directory-evolution.md) — why it is built this way
* [Strategies](strategies.md) — `FileTree` alongside the text strategies
