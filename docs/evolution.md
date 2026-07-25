# The `evolve` method

`evolve()` is the **one entry point** to the framework. You describe *what
evolves* and *the rules of evolution*, and it runs the parallel, merge-based loop
(ledger → workers → aggregator → commit) for you. Every capability in Concordia
is a **plug-in to a single `evolve()` parameter** — this page is the map.

```python
from concordia.agents import claude
from concordia.evolution import evolve, LLMAgent

result = evolve(
    tasks,                                   # what to work on
    reward,                                  # how to score an output
    agent=LLMAgent(claude(model="claude-haiku-4-5")),
)
print(result.rendered)        # the evolved artifact
print(result.final_reward)    # held-out reward
```

That's the minimum. Everything below is optional and swappable.

---

## Every knob is a module

| `evolve(...)` parameter | Module | What it plugs in | Default |
|---|---|---|---|
| `agent=` / `run=`+`propose=` | [`concordia.agents`](agents.md) + `LLMAgent` | the actor: solve a task, propose a change | — (required) |
| `strategy=` | `Strategy` (`AppendRules` / `KeyedRules` / yours) | the **evolution rule** — how a proposal becomes a diff | `AppendRules()` |
| `parallel=` | [`concordia.parallel`](parallelism.md) | the **parallelism method** — DP / TP / PP | `DataParallel()` |
| `blast_radius=` | governance | which layer (L2 skill vs L1 harness/verifier) | `0.2` (L2) |
| `agg_config=` | `AggregatorConfig` | merge & acceptance **tuning** | sensible defaults |
| `aggregator_factory=` | `AggregatorProtocol` | **swap the whole optimizer** (custom merge/acceptance) | reference `Aggregator` |
| `staleness_policy=` | staleness (`get_policy(...)`) | how stale diffs are handled | `guarded` |
| `rounds=`, `n_workers=` | driver | loop size, parallel worker count | `15`, `4` |
| `blast_radius`, `oracle_budget` | governance + verifier | audit budget for L1 merges | `0.2`, `200` |

The building blocks in detail:

---

## 1. The actor — `agent=` (or `run=` + `propose=`)

*What:* the thing that runs a task against the current artifact and, on a
failure, proposes an improvement. *Module:* [`concordia.agents`](agents.md)
provides the provider-agnostic **completion** (`prompt -> text`); `LLMAgent`
adapts a completion into the two-method actor.

```python
from concordia.agents import claude, openai_compatible, from_callable
from concordia.evolution import LLMAgent

evolve(tasks, reward, agent=LLMAgent(claude(model="claude-haiku-4-5")))        # Claude
evolve(tasks, reward, agent=LLMAgent(openai_compatible(model="glm-4.6")))      # GLM / OpenAI-style
evolve(tasks, reward, agent=LLMAgent(from_callable(my_llm)))                   # any prompt->text fn
```

Or skip `LLMAgent` and pass two plain functions — no LLM needed:

```python
evolve(tasks, reward,
       run=lambda rendered, task: my_solver(rendered, task),
       propose=lambda rendered, task, out, score: my_lesson(task, out))
```

---

## 2. The evolution rule — `strategy=`

*What:* how the artifact is represented and **how an agent's proposal becomes a
`Diff`**. The artifact's state is a flat `{key: value}` dict — the op-space the
aggregator resolves conflicts and fusion over.

```python
from concordia.evolution import AppendRules, KeyedRules

evolve(tasks, reward, agent=agent, strategy=AppendRules())                       # default
evolve(tasks, reward, agent=agent, strategy=KeyedRules(categories=["route","fmt"]))
```

| Strategy | Rule |
|---|---|
| `AppendRules` | each proposal → a content-addressed rule; identical ones dedupe, complementary ones **fuse** (append-only) |
| `KeyedRules(categories)` | one entry per category; competing proposals for the same category **contradict** and are resolved on held-out score |

Write your own by implementing three methods (`initial` / `render` / `to_diff`):

```python
from concordia.evolvable import Diff

class SingleSlot:                     # the artifact is one value each proposal replaces
    def initial(self): return {}
    def render(self, state): return state.get("v", "(none)")
    def to_diff(self, state, proposal, author, base_version, target):
        if state.get("v") == proposal: return None
        return Diff(diff_id=f"{author}:{base_version}", target=target,
                    ops={"v": proposal}, author=author)

evolve(tasks, reward, agent=agent, strategy=SingleSlot())
```

Distinct `Diff.ops` keys → **fused**; same key, different value → **resolved** on
held-out. That's how your logic composes with the merge machinery for free.
Full detail: [strategies on the concepts page](concepts.md).

---

## 3. The parallelism method — `parallel=`

*What:* how each round's tasks are partitioned across the `n_workers`. *Module:*
[`concordia.parallel`](parallelism.md).

```python
from concordia.parallel import DataParallel, TensorParallel, PipelineParallel

evolve(tasks, reward, agent=agent, parallel=DataParallel())                # default (shard tasks)
evolve(tasks, reward, agent=agent, parallel=TensorParallel(n_sections=4))  # disjoint sections
evolve(tasks, reward, agent=agent, parallel=PipelineParallel(stages=[...]))# per-stage workers
```

Or your own — implement `plan(n_workers, round_index, keys) -> [WorkUnit]`:

```python
from concordia.parallel import WorkUnit

class Blocks:
    name = "block"
    def plan(self, n_workers, round_index, keys):
        keys = list(keys); size = (len(keys)+n_workers-1)//n_workers
        return [WorkUnit(worker=i, keys=keys[i*size:(i+1)*size]) for i in range(n_workers)]

evolve(tasks, reward, agent=agent, parallel=Blocks())
```

Details + the DP/TP/PP semantics: [Customizable parallelism](parallelism.md).

---

## 4. Governance — `blast_radius=`

*What:* which governance layer the artifact lives in — the aggregator treats
high-impact artifacts more conservatively, automatically.

```python
evolve(tasks, reward, agent=agent, blast_radius=0.2)   # L2: a local skill/prompt
evolve(tasks, reward, agent=agent, blast_radius=0.6)   # L1: a harness / verifier
```

| `blast_radius` | Layer | Treatment |
|---|---|---|
| `≤ 0.30` | **L2** (skill, prompt, few-shot) | full async merge; cheap layers may pass a merge |
| `0.30–0.85` | **L1** (harness, context policy, tool router, verifier) | **every merge forced through the oracle**; wider staleness tolerance |
| frozen ids | **L0** (oracle, audit budget, permissions, safety) | read-only — the loop rejects mutations |

`oracle_budget=` caps how many ground-truth oracle checks the L1 audit may spend.
See [governance in concepts](concepts.md#6-governance-blast-radius-decides-parallelism).

---

## 5. The aggregator — `agg_config=` (tune) / `aggregator_factory=` (replace)

*What:* the optimizer that decides what to merge (staleness filter → conflict
resolution → fusion → statistical acceptance → transactional commit). `agg_config`
tunes the reference pipeline; `aggregator_factory` swaps in your own.

```python
from concordia.aggregator import AggregatorConfig, Aggregator

# tune: keep the pipeline, change the knobs
evolve(tasks, reward, agent=agent,
       agg_config=AggregatorConfig(base_delta=0.5, trust_region_ops=6))

# replace: subclass one decision (or satisfy AggregatorProtocol from scratch)
class StrictAggregator(Aggregator):
    def _tournament(self, artifact, diffs):
        return super()._tournament(artifact, [diffs[0]] if diffs else diffs)  # never fuse

evolve(tasks, reward, agent=agent,
       aggregator_factory=lambda ledger, verifier, audit, config, policy:
           StrictAggregator(ledger, verifier, audit, config, staleness_policy=policy))
```

Full field reference, the 7-stage pipeline, override points, and a from-scratch
aggregator: **[the aggregator page](aggregator.md)**.

---

## 6. Staleness — `staleness_policy=`

*What:* what to do with a diff proposed against an out-of-date artifact version.
*Module:* the Full / Guarded / Reflective policies.

```python
from concordia.staleness import get_policy

evolve(tasks, reward, agent=agent, staleness_policy=get_policy("reflective"))
```

| Policy | Behaviour |
|---|---|
| `full` | use stale diffs as-is (max throughput) |
| `guarded` | version-gated: accept `η=0`, rebase `η≤α`, discard beyond (default) |
| `reflective` | always rebase + re-verify; discard only if the gain no longer holds |

Staleness bites when workers lag head — which is most visible in the **async
runtime** (`async_ratio`), below. In synchronous `evolve()` each round proposes
against the current head, so η is usually 0. Deep dive:
[staleness in concepts](concepts.md#3-staleness).

---

## What `evolve` returns

```python
result = evolve(tasks, reward, agent=agent, rounds=6, verbose=True)

result.rendered       # the evolved artifact, rendered to text
result.state          # its {key: value} state
result.final_reward   # held-out reward of the final artifact
result.history        # per-round: RoundInfo(round, held_out_reward, n_items, committed, rejected)
result.ledger_log     # the git commit log of accepted merges
```

The engine returns **partial results** if the model backend fails mid-run (rate
limit, credit exhaustion) — progress isn't lost.

---

## Putting it all together

The one complete, runnable example threads every block above on a real dataset
with a real LLM:

```python
from concordia.agents import claude
from concordia.evolution import evolve, LLMAgent, AppendRules
from concordia.parallel import DataParallel

result = evolve(
    tasks, reward,
    agent=LLMAgent(claude(model="claude-haiku-4-5")),   # 1. actor       (agents)
    strategy=AppendRules(),                              # 2. rule        (strategy)
    parallel=DataParallel(),                             # 3. parallelism (parallel)
    blast_radius=0.2,                                    # 4. governance  (L2)
    rounds=6, n_workers=4,
)
```

Walkthrough with a real result (`0.750 → 0.792`): the
[skill-evolution example](skill-evolution.md).

---

## The other execution mode: the async runtime

`evolve()` is synchronous (round barrier). For a **barrier-free** pipeline where
workers never wait for each other — and where the staleness policies,
[`async_ratio`](concepts.md#34-async_ratio-roll-flash-the-global-lag-budget), and
[duration-aware straggler checkpointing](duration-scheduling.md) come into their
own — use `AsyncConcordia` (same aggregator, staleness, and governance
underneath). Measured trade-offs: [efficiency experiments](efficiency.md).
