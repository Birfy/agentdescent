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

## Run the example (no API key needed)

```bash
python -m examples.skill_evolution           # deterministic mock agent
python -m examples.skill_evolution --claude   # real Claude agent (needs ANTHROPIC_API_KEY)
```

The example evolves a text-normalization skill from an empty playbook. Sample
run (mock agent):

```
round   0  reward=0.773  rules=1  +1/-0
round   1  reward=1.000  rules=4  +1/-0
...
=== evolved playbook ===
# Skill Playbook
- Collapse runs of whitespace into a single space.
- Remove all punctuation characters.
- Strip leading and trailing whitespace from the text.
- Convert the text to lowercase.

held-out reward: 0.773 -> 1.000
rules learned: 4
```

Two things worth noting: parallel workers propose *different* missing rules, and
the aggregator **fuses** them into one playbook (merge-over-fork); and the mock
agent's occasional harmful proposal ("convert to UPPERCASE") is **rejected** on
held-out reward — it never enters the playbook.

---

## Cost

Evaluation runs the agent on held-out tasks, so a real LLM agent makes many
calls per round (rollouts + the held-out acceptance test + the aggregator's
cheap-eval subsets). Two levers keep this tractable:

- Keep task counts and `held_out_frac` modest.
- Identical `(playbook, task)` evaluations are **memoized within a run**, so
  repeated scoring of an unchanged playbook is free.

For high-volume evolution, use a cheaper model tier for the agent.
