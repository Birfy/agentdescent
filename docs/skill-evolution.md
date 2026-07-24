# Skill evolution (one application)

`concordia.skillevo` is **one application** built on the framework — a
convenience layer, not the framework itself. It wires the general pieces
(ledger, aggregator, staleness, governance) into a simple loop for the common
case: evolve a text skill by accumulating lessons. Every part of it is
customizable, and for full control you can skip it and drive
[`Ledger`](architecture.md) + [`Aggregator`](concepts.md#4-the-aggregator-a-discrete-space-optimizer)
directly.

```python
from concordia.skillevo import evolve_skill

result = evolve_skill(agent, tasks, reward, rounds=15, n_workers=4)
print(result.playbook)      # the evolved skill text
print(result.final_reward)  # held-out reward
```

Each round, parallel workers run tasks through the current skill and, on a
failure, ask the agent to propose an improvement. The aggregator dedupes,
resolves contradictions, fuses complementary changes, and **commits a change
only if it improves held-out reward** — so unhelpful changes are rejected
automatically.

---

## The three customization points

`evolve_skill` has exactly three things you plug in. Everything else (the
parallel loop, merging, acceptance) is the framework's job.

### 1. The agent — *how tasks get solved and improvements proposed*

Any object with two methods. Bring your own, or wrap a completion from
[`concordia.agents`](agents.md):

```python
from concordia.skillevo import Agent, Task

class MyAgent:
    def solve(self, skill_text: str, task: Task) -> str:
        ...                                   # run the task using the skill
    def propose(self, skill_text: str, task: Task, output: str, reward: float) -> str | None:
        ...                                   # reflect on a failure; return ONE improvement

# or: LLMAgent(claude(model="claude-haiku-4-5"))
```

The provider/LLM connection is **not** part of skillevo — see
[Connecting agents & LLMs](agents.md).

### 2. The reward — *how success is measured*

```python
def reward(task: Task, output: str) -> float:   # in [0, 1]
    ...
```

### 3. The strategy — *the evolution rule & logic*

A `SkillStrategy` decides **what the skill is, how it renders into a prompt, and
how an agent's proposal becomes a change** (a `Diff` the aggregator merges). A
skill's state is a flat `{key: value}` dict — the op-space the aggregator
resolves conflicts and fusion over.

```python
from typing import Optional
from concordia.evolvable import Diff

class SkillStrategy(Protocol):
    def initial(self) -> dict: ...                     # starting state
    def render(self, state: dict) -> str: ...          # state -> prompt text
    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]: ...
```

Two strategies ship built in:

| Strategy | Evolution logic |
|---|---|
| **`AppendRules`** (default) | Each proposal becomes a new **content-addressed** rule. Identical proposals from different workers dedupe; complementary rules are **fused**. Append-only. |
| **`KeyedRules(categories)`** | One lesson **per category**. Two workers proposing different text for the same category **contradict**, and the aggregator keeps the one that scores better. Demonstrates conflict resolution over user-defined logic. |

```python
from concordia.skillevo import evolve_skill, KeyedRules

result = evolve_skill(agent, tasks, reward,
                      strategy=KeyedRules(categories=["method", "format", "edge_case"]))
```

#### Writing your own strategy

Anything implementing the three methods works — you are not limited to "a
playbook of rules". For example, a skill that is a **single slot** each proposal
replaces (so proposals always compete and the best-scoring one wins):

```python
from concordia.evolvable import Diff

class SingleValueStrategy:
    def initial(self):
        return {}
    def render(self, state):
        return state.get("v", "(none)")
    def to_diff(self, state, proposal, author, base_version, target):
        if state.get("v") == proposal:
            return None                                  # no change
        return Diff(diff_id=f"{author}:{base_version}", target=target,
                    ops={"v": proposal}, author=author)  # overwrite the slot

result = evolve_skill(agent, tasks, reward, strategy=SingleValueStrategy())
```

The key idea: your `to_diff` maps a proposal into `Diff.ops` (a `{key: value}`
edit). Distinct keys → complementary → the aggregator **fuses** them; the same
key with different values → a contradiction → the aggregator **resolves** it on
held-out score. That is how your evolution logic composes with the framework's
merge machinery for free.

---

## Evolving a harness or verifier — not just skills

**What** evolves is chosen by registration and blast radius, not hard-coded. The
same machinery that evolves a local skill evolves a **harness module, context
policy, tool router, or learned verifier** — you just raise `blast_radius` so the
artifact lands in the **L1 (slow) governance layer** instead of L2:

```python
result = evolve_skill(
    agent, tasks, reward,
    strategy=KeyedRules(categories=["route", "context", "retry"]),
    blast_radius=0.6,          # 0.2 = L2 skill; 0.6 = L1 harness/verifier
    skill_id="harness",
)
```

The design deliberately treats high-blast-radius artifacts more conservatively
(design spec §6), and the aggregator does exactly that automatically:

| | L2 skill (`blast_radius ≤ 0.3`) | L1 harness / verifier (`0.3 < radius ≤ 0.85`) |
|---|---|---|
| Governance layer | fast, full async merge | slow, conservative |
| Oracle audit | cheap rule/learned layers may pass a merge | **every merge is forced through the oracle** |
| Staleness tolerance `α` | tight (tail) | wider (head) |

So a harness change can't ride cheap approximate evaluation alone — it must clear
the ground-truth oracle before it commits, exactly because its blast radius is
large. Nothing else in your code changes: same agent, same reward, same
strategy. (`L0` artifacts — the oracle, audit budget, merge permissions, safety
constraints — are frozen and rejected by the loop; see
[Concepts §6](concepts.md#6-governance-blast-radius-decides-parallelism).)

---

## Run the example — a real dataset, real Claude

[`examples/skill_evolution.py`](https://github.com/Birfy/concordia/blob/main/examples/skill_evolution.py)
evolves a skill on a **BIG-Bench-Hard** task with a real Claude agent.

```bash
python -m examples.skill_evolution --dry-run                  # dataset + cost estimate, no API
python -m examples.skill_evolution --task word_sorting --model claude-haiku-4-5
```

!!! warning "Cost"
    A real-LLM run makes many calls (rollouts + held-out scoring + the
    aggregator's cheap-eval subsets). Defaults are small; the script prints an
    estimate and asks before spending. Identical `(skill, task)` evaluations are
    memoized within a run.

The no-API mechanics (any-agent protocol, custom strategies, merge-over-fork,
harmful-change rejection) are exercised deterministically in the test suite
([`test_skillevo.py`](https://github.com/Birfy/concordia/blob/main/tests/test_skillevo.py),
[`test_strategy.py`](https://github.com/Birfy/concordia/blob/main/tests/test_strategy.py)).
