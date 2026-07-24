# Evolving anything

`concordia.evolution` is **the module** — a domain-agnostic engine that evolves
*any* artifact. It knows nothing about "skills" or "harnesses". You describe
**what evolves** and **the rules of evolution**, and it runs the parallel,
merge-based loop (ledger + aggregator + staleness + governance) for you.

Skill evolution and harness evolution are just two *examples* of using it
(below). For total control you can also skip the engine and drive
[`Ledger`](architecture.md) + [`Aggregator`](concepts.md#4-the-aggregator-a-discrete-space-optimizer)
directly.

```python
from concordia.evolution import evolve

result = evolve(tasks, reward, agent=my_agent, strategy=AppendRules(),
                blast_radius=0.2, rounds=15, n_workers=4)
print(result.rendered)      # the evolved artifact
print(result.final_reward)  # held-out reward
```

---

## What you provide

Four things — everything else (the parallel loop, merging, statistical
acceptance) is the engine's job.

### 1. A `Strategy` — *what the artifact is, and how a proposal becomes a change*

An artifact's state is a flat `{key: value}` dict (the diff op-space the
aggregator resolves conflicts and fusion over). A strategy decides the initial
state, how it renders (into a prompt or config), and how a proposal becomes a
`Diff`:

```python
from typing import Protocol, Optional
from concordia.evolvable import Diff

class Strategy(Protocol):
    def initial(self) -> dict: ...
    def render(self, state: dict) -> str: ...
    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]: ...
```

Two ship built in:

| Strategy | Rule |
|---|---|
| **`AppendRules`** | each proposal → a content-addressed rule; identical proposals dedupe, complementary ones **fuse** (append-only) |
| **`KeyedRules(categories)`** | one entry per category; competing proposals for the same category **contradict** and are resolved on held-out score |

The key idea: `to_diff` maps a proposal into `Diff.ops`. Distinct keys →
complementary → **fused**; same key, different value → contradiction →
**resolved**. That is how *your* logic composes with the framework's merge
machinery for free.

### 2–3. `run` and `reward` — *apply the artifact, and score it*

```python
def run(rendered: str, task) -> str: ...     # apply the artifact to a task
def reward(task, output) -> float: ...        # score in [0, 1]
```

### 4. `propose` — *turn a failure into an improvement*

```python
def propose(rendered: str, task, output, reward) -> str | None: ...
```

`run` + `propose` can be plain functions, or bundled into an `Agent`
(`solve`/`propose`) — `evolve(agent=...)` accepts either. The LLM connection is
separate: see [Connecting agents & LLMs](agents.md).

---

## `blast_radius` chooses the governance layer

The *same* engine evolves a fast local artifact or a high-impact one — you just
set how much blast radius it has:

| `blast_radius` | Layer | Treatment |
|---|---|---|
| `≤ 0.30` | **L2** (skill, prompt, few-shot) | full async merge; cheap layers may pass a merge |
| `0.30–0.85` | **L1** (harness, context policy, tool router, verifier) | conservative; **every merge forced through the oracle**; wider staleness tolerance |
| frozen | **L0** (oracle, audit budget, permissions, safety) | read-only — the loop rejects mutations |

Nothing else in your code changes between L2 and L1 (design spec §6).

---

## Example A — evolve a skill (real dataset, real LLM)

Artifact = a lesson playbook; `run` = an LLM applying it. Full walkthrough on
the [skill example](skill-evolution.md) page.

```python
from concordia.agents import claude
from concordia.evolution import evolve, LLMAgent, AppendRules

result = evolve(tasks, reward,
                agent=LLMAgent(claude(model="claude-haiku-4-5")),
                strategy=AppendRules(), blast_radius=0.2, artifact_id="skill")
```

```bash
python -m examples.skill_evolution --dry-run   # BIG-Bench-Hard, no API calls
```

## Example B — evolve a harness (L1, no LLM)

Artifact = a request-processing pipeline (route / normalize / trim); driven by
plain functions, registered at L1. Full walkthrough on the
[harness example](harness-evolution.md) page.

```python
from concordia.evolution import evolve, KeyedRules

result = evolve(tasks, reward, run=run, propose=propose,
                strategy=KeyedRules(categories=["route", "normalize", "trim"]),
                blast_radius=0.6, artifact_id="harness")
```

```bash
python -m examples.harness_evolution           # deterministic, runs anywhere
```

Same engine, same `evolve` call — only the artifact, its strategy, and its blast
radius differ. That is the point: **write the rules of evolution, and it runs.**
