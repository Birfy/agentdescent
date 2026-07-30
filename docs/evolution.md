# The `evolve` method

`evolve()` is the **one entry point** to the framework. You describe *what
evolves* and *the rules of evolution*, and it runs the parallel, merge-based loop
(ledger → workers → aggregator → commit) for you. Every capability in AgentDescent
is a **plug-in to a single `evolve()` parameter** — this page is the map.

```python
from agentdescent.agents import claude
from agentdescent.evolution import evolve, LLMAgent

result = evolve(
    tasks,                                   # what to work on
    reward,                                  # how to score an output
    agent=LLMAgent(claude(model="claude-haiku-4-5")),
)
print(result.rendered)        # the evolved artifact
print(result.final_reward)    # held-out reward
```

That's the minimum. Everything below is optional and swappable.

!!! tip "Where do `tasks` come from?"
    The `tasks` and `reward` are yours to define. To pull them from a public
    benchmark without writing HuggingFace paging/caching boilerplate, use the
    [`agentdescent.dataloader`](dataloader.md) data layer (`hf_rows`, `fetch_text`,
    `load_gated_hf`) — it is how every
    [self-evolution example](self-evolution-examples.md) loads its dataset.

---

## Every knob is a module

| `evolve(...)` parameter | Module | What it plugs in | Default |
|---|---|---|---|
| `agent=` / `run=`+`propose=` | [`agentdescent.agents`](agents.md) + `LLMAgent` | the actor: solve a task, propose a change | — (required) |
| `strategy=` | `Strategy` (`AppendRules` / `KeyedRules` / yours) | the **evolution rule** — how a proposal becomes a diff | `AppendRules()` |
| `parallel=` | [`agentdescent.parallel`](parallelism.md) | the **parallelism method** — DP / TP / PP | `DataParallel()` |
| `task_sampler=` | [`agentdescent.sampling`](#task-selection-which-rollout-to-spend) | **which task** a worker rolls out next | `RoundRobin()` |
| `blast_radius=` | governance | which layer (L2 skill vs L1 harness/verifier) | `0.2` (L2) |
| `agg_config=` | `AggregatorConfig` | merge & acceptance **tuning** | sensible defaults |
| `aggregator_factory=` | `AggregatorProtocol` | **swap the whole optimizer** (custom merge/acceptance) | reference `Aggregator` |
| `staleness_policy=` | staleness (`get_policy(...)`) | how stale diffs are handled | `guarded` |
| `rounds=`, `n_workers=` | driver | loop size, parallel worker count | `15`, `4` |
| `max_concurrency=` | driver | run a round's workers **concurrently** (thread pool); aggregator = barrier (synchronous DP) | `1` (sequential) |
| `round_timeout=` | driver | cap how long a round waits for its workers — abandons stragglers | `None` (wait forever) |
| `asynchronous=`, `async_ratio=` | [`async_evolve`](#the-barrier-free-runtime-async_evolve) | **barrier-free** async: workers never wait for the merge; lag budget | `False`, `3` |
| `self_verify=` | [`async_evolve`](#the-barrier-free-runtime-async_evolve) | async only: a worker re-runs its trajectory with the diff applied for a local before/after signal; faithful ports that score the candidate on held-out only pass `False` | `True` |
| `on_round=` | driver | **progress callback** — fires per round / merger sweep | `None` |
| `blast_radius`, `oracle_budget` | governance + verifier | audit budget for L1 merges | `0.2`, `200` |

The building blocks in detail:

---

## 1. The actor — `agent=` (or `run=` + `propose=`)

*What:* the thing that runs a task against the current artifact and, on a
failure, proposes an improvement. *Module:* [`agentdescent.agents`](agents.md)
provides the provider-agnostic **completion** (`prompt -> text`); `LLMAgent`
adapts a completion into the two-method actor.

```python
from agentdescent.agents import claude, openai_compatible, from_callable
from agentdescent.evolution import LLMAgent

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
from agentdescent.evolution import AppendRules, KeyedRules

evolve(tasks, reward, agent=agent, strategy=AppendRules())                       # default
evolve(tasks, reward, agent=agent, strategy=KeyedRules(categories=["route","fmt"]))
```

| Strategy | Rule |
|---|---|
| `AppendRules` | each proposal → a content-addressed rule; identical ones dedupe, complementary ones **fuse** (append-only) |
| `KeyedRules(categories)` | one entry per category; competing proposals for the same category **contradict** and are resolved on held-out score |

Write your own by implementing three methods (`initial` / `render` / `to_diff`):

```python
from agentdescent.evolvable import Diff

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

**Strategies implemented in the [algorithm ports](self-evolution-examples.md)** —
each is a real `Strategy` you can read and reuse:

| Strategy | Example | What the artifact is |
|---|---|---|
| `ACEPlaybook` | [ACE](algo-ace.md) | an itemised, incremental-delta context playbook (append-only + grow-and-refine de-dup) |
| `InstructionSlot` | [GEPA](algo-gepa.md) | one instruction prompt each proposal replaces |
| `SkillLibraryStrategy` | [EvoSkill](algo-evoskill.md) | a library of `SKILL.md` skills (a proposal appends one) |
| `SkillDocStrategy` | [SkillOpt](algo-skillopt.md) | one markdown skill doc mutated by bounded `append/insert_after/replace/delete` edits |
| `AgentDesignStrategy` | [ADAS](algo-adas.md) | one agentic-system design (a control-flow program) each proposal replaces |
| `HarnessStrategy` | [DGM](algo-dgm.md) | a coding-agent harness's capability set (a proposal adds one) |

---

## 3. The parallelism method — `parallel=`

*What:* how each round's tasks are partitioned across the `n_workers`. *Module:*
[`agentdescent.parallel`](parallelism.md).

```python
from agentdescent.parallel import DataParallel, TensorParallel, PipelineParallel

evolve(tasks, reward, agent=agent, parallel=DataParallel())                # default (shard tasks)
evolve(tasks, reward, agent=agent, parallel=TensorParallel(n_sections=4))  # disjoint sections
evolve(tasks, reward, agent=agent, parallel=PipelineParallel(stages=[...]))# per-stage workers
```

Or your own — implement `plan(n_workers, round_index, keys) -> [WorkUnit]`:

```python
from agentdescent.parallel import WorkUnit

class Blocks:
    name = "block"
    def plan(self, n_workers, round_index, keys):
        keys = list(keys); size = (len(keys)+n_workers-1)//n_workers
        return [WorkUnit(worker=i, keys=keys[i*size:(i+1)*size]) for i in range(n_workers)]

evolve(tasks, reward, agent=agent, parallel=Blocks())
```

Details + the DP/TP/PP semantics: [Customizable parallelism](parallelism.md).

---

## Task selection — which rollout to spend

*What:* which task a worker rolls out next. *Module:*
[`agentdescent.sampling`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/sampling.py).

A rollout is the expensive unit of work — one LLM call, or an entire tool-using
agent trajectory that can run for minutes. Spending it on a task the agent
**already solves** teaches the system nothing: there is no failure, so no
proposal, so no diff. The same is true of a task it can *never* solve. Only tasks
somewhere in between carry a usable gradient (the GRPO zero-advantage argument).

```python
from agentdescent.sampling import DifficultyWeighted, RoundRobin

evolve(tasks, reward, agent=agent, task_sampler=RoundRobin())          # default
evolve(tasks, reward, agent=agent, task_sampler=DifficultyWeighted())  # focus the budget
```

| Sampler | Rule |
|---|---|
| **`RoundRobin`** (default) | cycle through the shard — deterministic, but spends rollouts uniformly |
| **`DifficultyWeighted`** | track each task's pass rate; prefer those away from the all-pass / all-fail extremes, with a UCB bonus so untried tasks are still explored |

Measured on a 40-task workload where only 6 tasks carry signal — the share of
rollouts that landed on an informative task:

| | round-robin | difficulty-weighted |
|---|---|---|
| clean reward | 14.5% | **23.4%** |
| 15% reward noise | 7.3% | **16.3%** |

!!! warning "That is a targeting measurement, not an accuracy claim"
    Landing more rollouts on failing tasks does **not** automatically produce a
    better artifact. On real [ACE / FiNER-139 runs](algo-ace.md#empirical-results-finer-139-with-deepseek)
    the difficulty-weighted sampler reached a lesson sooner (round 0 versus round
    2) but did not score better — and two runs of the *same* round-robin
    configuration differed by 4.8 points, so at that sample size neither sampler
    is distinguishable from the other. Concentrating on the hardest tasks can also
    yield lessons that fit those tasks and generalise worse. Treat this sampler as
    **worth trying and worth measuring on your own data**, not as a free win;
    `RoundRobin` stays the default.

Where it should help most is a *strong* base agent on a large task pool: failures
are sparse, so round-robin spends most of its budget re-solving solved tasks.
Write your own by implementing `pick` + `record`:

```python
class PreferRecentFailures:
    def pick(self, keys, round_index): ...      # -> one task id
    def record(self, task_id, score): ...       # learn from the outcome
```

!!! note "Default stays deterministic"
    `RoundRobin` remains the default so existing runs and the RQ experiments stay
    bit-reproducible. `DifficultyWeighted` is opt-in.

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
from agentdescent.aggregator import AggregatorConfig, Aggregator

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
from agentdescent.staleness import get_policy

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
result.error          # None on a clean run; "<ExcType>: <msg>" if a backend failure ended it
```

Watch a long run as it happens — an LLM run can take hours, and `history` is only
available once it returns:

```python
evolve(tasks, reward, agent=agent, rounds=20,
       on_round=lambda info: print(info.round, info.held_out_reward))
```

`on_round` fires per round (per merger sweep on the async path, where it runs on
the merger thread and so must be cheap and thread-safe). An exception inside it is
reported as a warning and never aborts the run.

Keep the artifact a run produced:

```python
result.save("playbook.json")                     # state + rendered + history + error
restored = EvolutionResult.load("playbook.json")
```

### Resuming a run

The ledger is a real git repo, so **passing the same `repo_path` again continues
where the last run stopped** — which is what you want when a multi-hour run dies
to a rate limit or a dropped connection:

```python
evolve(tasks, reward, agent=agent, rounds=10, repo_path="runs/finer")   # dies at round 6
evolve(tasks, reward, agent=agent, rounds=10, repo_path="runs/finer")   # picks up the artifact
```

The second call starts from the artifact the first one committed, not from
`strategy.initial()`. Two consequences worth knowing:

* `rounds` is **not** remaining work — the second call runs its own `rounds`
  rounds on top of the existing artifact.
* `initial_state=` is ignored when the artifact already exists (a `RuntimeWarning`
  says so). Use a fresh `repo_path` to start over.

Omit `repo_path` and the ledger is a scratch directory, cleaned up at exit.

The engine returns **partial results** if the model backend fails mid-run (rate
limit, credit exhaustion) — progress isn't lost.

!!! warning "Always check `result.error`"
    A run that dies after two rounds and a run that converges both return an
    `EvolutionResult`. `error` is what distinguishes them — it is `None` only on a
    clean run. Treating a failed run as a converged one is the easiest way to
    misread an experiment:

    ```python
    if result.error:
        print(f"incomplete: {result.error}")   # partial artifact still usable
    ```

    `error` means *the run ended because of this failure* — a transient error the
    workers retried past leaves it `None`. A `RuntimeWarning` is also emitted, so
    a failed run is never completely silent even at the default `verbose=False`.

**Actor signatures are checked before the first rollout.** `run` and `propose` are
bound-tested up front, so a plain typo (a `propose` missing its `reward`
parameter) raises `TypeError` immediately instead of surfacing as a
backend-shaped failure with zero rounds run and an empty artifact.

**Backend failures are tolerated, not fatal.** In the async runtime a transient
error (a rate limit, a flaky endpoint) is retried with exponential backoff; a
worker retires only after `3` *consecutive* failures, and the run ends when every
worker has retired — so one hiccup in one worker no longer kills a long run. The
first failure seen is always reported through `result.error`.

---

## Putting it all together

The one complete, runnable example threads every block above on a real dataset
with a real LLM:

```python
from agentdescent.agents import claude
from agentdescent.evolution import evolve, LLMAgent, AppendRules
from agentdescent.parallel import DataParallel

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

## Parallelism & async — the framework's core

Parallel, merge-based evolution is the whole point (targeting **O(N / T_iter)**),
so it shows up at two levels:

* **Within a round — `max_concurrency`.** `evolve()` runs a round's `n_workers`
  **concurrently** (a thread pool): every worker's rollout+propose overlaps, then
  the single `aggregator.step()` is the barrier. This is *synchronous
  data-parallelism* — real wall-clock speedup for I/O-bound LLM rollouts (Python
  releases the GIL during network I/O). Every
  [self-evolution example](self-evolution-examples.md) passes
  `max_concurrency=n_workers`, so its workers genuinely run in parallel; custom
  strategies/aggregators guard the shared state they mutate from `propose`/
  `to_diff` with a lock.

```python
evolve(tasks, reward, agent=agent, n_workers=4, max_concurrency=4)   # 4 workers overlap
```

!!! tip "Bound the barrier — `round_timeout`"
    Because the aggregator *is* the barrier, a round waits for its slowest worker
    for as long as that takes: one hung rollout stalls the run indefinitely. Cap it:

    ```python
    evolve(tasks, reward, agent=agent, n_workers=4, max_concurrency=4,
           round_timeout=300)          # give up on stragglers after 5 min
    ```

    Abandoned work keeps running in the background — Python cannot cancel a
    thread — it is simply no longer waited for, and a genuine backend error still
    surfaces through `result.error`. This is the achievable part of the
    heavy-tailed-rollout problem for an opaque `run`; true turn-level resume would
    need a rollout contract exposing its turns (see
    [duration-aware scheduling](duration-scheduling.md)).

* **Across rounds — `asynchronous=True` (barrier-free).** Removing the round
  barrier entirely is [`async_evolve()`](#the-barrier-free-runtime-async_evolve),
  reachable as `evolve(asynchronous=True, async_ratio=…)`. It takes the **same**
  plug-ins, so every example runs async with a `--async` flag.

```python
evolve(tasks, reward, agent=agent, asynchronous=True, async_ratio=3, max_seconds=30)
```

## The barrier-free runtime: `async_evolve()`

`evolve()`'s round barrier means the aggregator waits for all workers each round.
[`async_evolve()`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/async_evolve.py)
removes it while accepting the identical `run`/`reward`/`propose`/`strategy`/
`aggregator_factory` plug-ins — so **any** task that runs under `evolve()` (ACE,
GEPA, EvoSkill, SkillOpt, ADAS, DGM) also runs async:

* **Workers** (`n_workers` threads) hold a snapshot and keep producing evidence
  against it, refreshing only once head drifts past **`async_ratio`** (the lag
  budget) — so staleness (η > 0) genuinely arises. The lag budget bounds
  **un-merged** work too: a worker won't pile up more than `async_ratio`
  candidates ahead of the merger. This matters at **cold start** — before the
  first commit, head hasn't advanced, so a version-only budget can't engage and
  workers would flood the buffer while the merger is busy on the first slow
  held-out eval; gating on pending intake prevents that.
* **One merger** drains a thread-safe buffer, runs each card through the
  **staleness policy** (`accept η=0` / `rebase`+re-verify / `discard`), then
  `ingest` + `step`. It is the only writer, so there are no CAS conflicts and
  every custom optimizer sees only rebased cards — async-safe unchanged.
* **`self_verify`** controls whether a worker, after producing a diff, re-runs
  its own trajectory with the diff applied to record a local before/after signal
  (`before_after_delta`, used by the staleness gate's cheap re-verify). Ports
  that only score the *candidate* on held-out — e.g. [EvoSkill](algo-evoskill.md),
  whose repo evaluates the child on the validation set and never re-runs the
  sampled task — pass `self_verify=False` to skip that extra rollout.

```python
from agentdescent import async_evolve
result = async_evolve(tasks, reward, agent=agent,
                      n_workers=4, async_ratio=3, max_seconds=30,   # or max_iters / target_reward
                      staleness_policy=get_policy("reflective"))
```

Reach it via `evolve(asynchronous=True)` or directly. Small `async_ratio` →
near-synchronous, few stale diffs; large → highly asynchronous, many stale diffs
the policy must rebase or discard.

### More async / parallel recipes

The two levers compose; pick per workload:

```python
# 1. Synchronous data-parallel: a round's workers overlap, aggregator is the barrier.
evolve(tasks, reward, agent=agent, n_workers=8, max_concurrency=8, rounds=10)

# 2. Barrier-free async, time-bounded: run for 20 min, keep the best head so far.
evolve(tasks, reward, agent=agent, asynchronous=True, async_ratio=3, max_seconds=1200)

# 3. Async to a target: stop as soon as held-out reward crosses a bar.
async_evolve(tasks, reward, agent=agent, n_workers=6, target_reward=0.85)

# 4. Async, rollout-bounded: cap total worker rollouts (budget), not wall-clock.
async_evolve(tasks, reward, agent=agent, n_workers=4, max_iters=200)

# 5. Faithful port on the async path: skip the per-trajectory re-run, score the
#    candidate on held-out only (see EvoSkill), with concurrent held-out eval.
evolve(tasks, reward, run=run, propose=propose, strategy=strat,
       aggregator_factory=factory, asynchronous=True, self_verify=False)

# 6. Highly-async, staleness-heavy: large lag budget + reflective rebase-and-verify.
async_evolve(tasks, reward, agent=agent, n_workers=8, async_ratio=8,
             staleness_policy=get_policy("reflective"))
```

An aggregator can amortise the expensive held-out eval on the async path — apply
each diff as a cheap step and only validate every *N* steps, rolling back on no
gain (SGD-style). See [the async optimizer variant](aggregator.md#the-async-optimizer-variant-sgd-style-descent).

### The reference async orchestrator: `AsyncAgentDescent`

For the router reference domain there is also `AsyncAgentDescent` — the original
stage-orchestration runtime with duration-aware straggler checkpointing, where
the staleness policies,
[`async_ratio`](concepts.md#34-async_ratio-roll-flash-the-global-lag-budget), and
[duration-aware straggler checkpointing](duration-scheduling.md) come into their
own — use `AsyncAgentDescent` (same aggregator, staleness, and governance
underneath). Measured trade-offs: [efficiency experiments](efficiency.md).
