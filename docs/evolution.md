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

### 5. A `parallel` method (optional) — *how work is partitioned across workers*

The parallelism method is a first-class, pluggable argument:
`evolve(..., parallel=DataParallel())`. Pick DP / TP / PP or write your own —
see [Customizable parallelism](parallelism.md). It defaults to `DataParallel`,
so you only set it when you want to change how each round's work is sharded.

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

## Same engine, different artifact

A **skill** — artifact = a lesson playbook, `run` = an LLM applying it, L2:

```python
from concordia.agents import claude
from concordia.evolution import evolve, LLMAgent, AppendRules

result = evolve(tasks, reward,
                agent=LLMAgent(claude(model="claude-haiku-4-5")),
                strategy=AppendRules(), blast_radius=0.2, artifact_id="skill")
```

A **harness / verifier** — artifact = a pipeline config, driven by plain
functions, registered at L1 (oracle-gated merges):

```python
from concordia.evolution import evolve, KeyedRules

result = evolve(tasks, reward, run=run, propose=propose,
                strategy=KeyedRules(categories=["route", "normalize", "trim"]),
                blast_radius=0.6, artifact_id="harness")
```

Same `evolve` call — only the artifact, its strategy, and its blast radius
differ. That is the point: **write the rules of evolution, and it runs.** The
complete, runnable end-to-end example (real dataset, real LLM, every module) is
on the [skill-evolution](skill-evolution.md) page.
