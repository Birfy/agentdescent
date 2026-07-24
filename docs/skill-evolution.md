# Skill self-evolution with any agent

`concordia.skillevo` is the **ergonomic front door**: evolve a real skill,
driven by *any* agent, in a few lines. Everything else in Concordia (ledger,
aggregator, staleness, governance) is the machinery underneath; this is the
convenient API on top.

```python
from concordia.skillevo import evolve_skill

result = evolve_skill(agent, tasks, reward, rounds=15, n_workers=4)
print(result.playbook)      # the evolved skill text
print(result.final_reward)  # held-out reward
```

---

## The idea

The evolving artifact is a **playbook** — an accumulating set of rules/lessons
(the ExpeL / "lessons learned" pattern). Each round:

1. Workers run tasks through the current playbook using **your agent**.
2. On a failure, the agent is asked to **propose** one new rule.
3. The **aggregator** does the hard part — dedup, contradiction resolution,
   fusion of complementary rules, and a **held-out statistical test**: a rule is
   committed only if it actually improves held-out reward. Bad rules are
   rejected automatically; good rules from parallel workers merge into one
   playbook.

So you get parallel, merge-based skill evolution with a real quality gate —
without writing any of the ledger/merge/acceptance code yourself.

---

## Bring any agent — the whole interface is two methods

```python
from concordia.skillevo import Agent, Task

class MyAgent:
    def solve(self, skill_text: str, task: Task) -> str:
        # run the task using the skill playbook; return the output
        ...
    def propose(self, skill_text: str, task: Task, output: str, reward: float) -> str | None:
        # reflect on a failure; return ONE new rule, or None
        ...
```

Anything with those two methods works — an LLM, a tool-using agent loop, a rule
engine, a mock. `isinstance(MyAgent(), Agent)` is `True` structurally.

### Wrap a completion function

If your agent is "call a model with a prompt", use `LLMAgent`:

```python
from concordia.skillevo import LLMAgent

agent = LLMAgent(lambda prompt: my_model.complete(prompt))
```

### Ready-made Claude agent

```python
from concordia.skillevo import claude_agent

agent = claude_agent(model="claude-opus-4-8")   # needs: pip install anthropic + credentials
# for many rounds, a cheaper tier keeps cost down:
agent = claude_agent(model="claude-haiku-4-5")
```

---

## What you supply

| Argument | Meaning |
|---|---|
| `agent` | Anything implementing the two-method `Agent` protocol |
| `tasks` | A list of `Task(id, prompt, meta)` |
| `reward` | `reward(task, output) -> float` in `[0, 1]` — how well the output solves the task |
| `rounds`, `n_workers` | Loop size and parallel worker count |
| `held_out_frac` | Fraction of tasks reserved for the acceptance test |

`evolve_skill` returns a `SkillEvoResult` with `.playbook` (text), `.rules`
(dict), `.final_reward`, and a per-round `.history`.

---

## Run the example — a real dataset, driven by real Claude

[`examples/skill_evolution.py`](https://github.com/Birfy/concordia/blob/main/examples/skill_evolution.py)
evolves a skill on a **BIG-Bench-Hard** task using a **Claude** agent to both
solve problems and propose lessons. BBH tasks are deliberately hard for LLMs and
scored by exact match / graded overlap, so there is genuine headroom for a
learned skill to raise the score.

```bash
# inspect the dataset + a cost estimate, no API calls:
python -m examples.skill_evolution --dry-run

# the real thing (needs ANTHROPIC_API_KEY or `ant auth login`):
python -m examples.skill_evolution
python -m examples.skill_evolution --task logical_deduction_seven_objects --rounds 5
python -m examples.skill_evolution --task word_sorting --model claude-haiku-4-5
```

`--dry-run` output:

```
Dataset : BIG-Bench-Hard / word_sorting
Loaded  : 250 examples; using 22 (12 train / 10 held-out)
Scoring : graded token overlap
Plan    : model=claude-opus-4-8, rounds=4, workers=2
Budget  : up to ~232 Claude calls (cached repeats are free; use --model claude-haiku-4-5 to cut cost)
```

Each round, parallel workers run *train* problems through the current playbook
via Claude and, on a failure, ask Claude to propose one lesson. The aggregator
dedupes, fuses complementary lessons, and **commits a lesson only if it improves
held-out score** — so unhelpful lessons are rejected automatically, and good
lessons from parallel workers merge into one playbook.

!!! warning "Cost"
    A real-LLM run makes many calls (rollouts + held-out scoring + the
    aggregator's cheap-eval subsets). The defaults are small on purpose; the
    script prints an estimate and asks before spending. Use
    `--model claude-haiku-4-5` for cheap runs. Identical `(playbook, task)`
    evaluations are memoized within a run.

The self-contained, no-API mechanics (any-agent protocol, merge-over-fork,
harmful-rule rejection) are exercised deterministically in the test suite
([`tests/test_skillevo.py`](https://github.com/Birfy/concordia/blob/main/tests/test_skillevo.py)).

---

## Cost

Evaluation runs the agent on held-out tasks, so a real LLM agent makes many
calls per round (rollouts + the held-out acceptance test + the aggregator's
cheap-eval subsets). Two levers keep this tractable:

- Keep task counts and `held_out_frac` modest.
- Identical `(playbook, task)` evaluations are **memoized within a run**, so
  repeated scoring of an unchanged playbook is free.

For high-volume evolution, use a cheaper model tier for the agent.
