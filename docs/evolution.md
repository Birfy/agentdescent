# The `evolve` method

`evolve()` is the **one entry point** to the framework. You describe *what
evolves* and *the rules of evolution*, and it runs the parallel, merge-based loop
(ledger → workers → aggregator → commit) for you. Every capability in AgentDescent
is a **plug-in to a single `evolve()` parameter** — this page is the map.

```python
from agentdescent import claude
from agentdescent import evolve, LLMAgent

result = evolve(
    tasks,                                   # what to work on
    reward,                                  # how to score an output
    agent=LLMAgent(claude(model="claude-haiku-4-5")),
)
print(result.rendered)        # the evolved artifact
print(result.final_reward)    # held-out reward
```

That's the minimum. Everything below is optional and swappable.

!!! tip "Three companion pages"
    [Module map](modules.md) is the index of every module and a reading
    order; [API reference](api.md) has every signature, generated from the
    code; [Concepts](concepts.md) is why any of it is shaped this way.

## Bring an agent you already have

The common case is not "write an agent for the framework" — it is "I have an
agent, make it better". That needs three lines: adapt it, pick something to
reflect with, say what evolves.

```python
from agentdescent import claude
from agentdescent import SingleSlot, evolve, reflector

def my_agent(system_prompt, question):        # whatever you already have
    ...

result = evolve(
    tasks, reward,
    run=lambda rendered, task: my_agent(rendered, task.prompt),   # adapt it
    propose=reflector(claude(model="claude-haiku-4-5")),          # who reflects
    strategy=SingleSlot(initial_value="Answer concisely."),       # what evolves
    n_workers=4, max_concurrency=4,        # ...in parallel
    # asynchronous=True,                   # ...or barrier-free
)
print(result.rendered)      # the evolved system prompt
```

`reflector(completion)` turns any model into the thing that looks at a failure and
says what to change — it need not be the model your agent uses, and a cheap one is
often the right reflector for an expensive agent. `SingleSlot` is the artifact
being one value that each accepted proposal replaces, which is what you want for a
system prompt. Switching between parallel and barrier-free async is one argument;
nothing else in the call changes.

!!! tip "Put the expected answer in `Task.meta` — the reflector reads it"
    A reflector that sees only the score is told *that* it was wrong, never what
    right looks like. It can then fix reasoning errors, but it cannot discover a
    **convention** it has no way to guess — an output format, a unit, a required
    field. Whatever you put in `meta` is shown to it:

    ```python
    Task(id="7", prompt="What do 7 pens cost?", meta={"gold": "2800"})   # cents
    ```

    `meta` is free-form and yours: the gold answer, a rubric, the failing
    assertion. It is rendered truncated (`meta_chars=600`) so a whole document in
    there cannot blow up the prompt, and the template tells the reflector to state
    its rule in general terms rather than naming this task's answer. Withhold it
    with `reflector(model, show_meta=False)` if your meta holds something you would
    rather the reflector not see.

    Measured on a real two-step `deepseek-v4-flash` agent over 12 money word
    problems whose scorer demands **integer cents** — a convention stated nowhere
    in the prompt:

    | | held-out |
    |---|---|
    | initial prompt (`"You are a helpful assistant."`) | 3/12 = **0.250** |
    | reflector blind to `meta`, 8 rounds | 0.500 (plateau) |
    | reflector reading `meta` | 12/12 = **1.000**, in one round (141 s, 63 calls) |

    And it generalised rather than memorising — the rule it wrote was
    *"Express all monetary amounts as integers representing cents, without dollar
    signs or decimal points."*, not this task's answer.

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
| `strategy=` | [`Strategy`](strategies.md) (`SingleSlot` / `AppendRules` / `KeyedRules` / [`FileTree`](directory-evolution.md) / yours) | the **evolution rule** — what the artifact is and how a proposal becomes a diff | `AppendRules()` |
| `parallel=` | [`agentdescent.parallel`](parallelism.md) | the **parallelism method** — DP / TP / PP | `DataParallel()` |
| `task_sampler=` | [`agentdescent.sampling`](sampling.md) | **which task** a worker rolls out next | `RoundRobin()` |
| `blast_radius=` | [governance](governance.md) | which layer (L2 skill vs L1 harness/verifier) | `0.2` (L2) |
| `agg_config=` | `AggregatorConfig` | merge & acceptance **tuning** | sensible defaults |
| `aggregator_factory=` | `AggregatorProtocol` | **swap the whole optimizer** (custom merge/acceptance) | reference `Aggregator` |
| `staleness_policy=` | [staleness](staleness.md) (`get_policy(...)`) | how stale diffs are handled | `guarded` |
| `refresh_interval=` | driver | rounds a worker keeps its snapshot — what *creates* staleness on the synchronous path | `1` (no staleness) |
| `rounds=`, `n_workers=` | driver | loop size, parallel worker count | `15`, `4` |
| `max_concurrency=` | driver | run a round's workers **concurrently** (thread pool); aggregator = barrier (synchronous DP) | `1` (sequential) |
| `round_timeout=` | driver | cap how long a round waits for its workers — abandons stragglers | `None` (wait forever) |
| `asynchronous=`, `async_ratio=` | [`async_evolve`](#the-barrier-free-runtime-async_evolve) | **barrier-free** async: workers never wait for the merge; lag budget | `False`, `3` |
| `self_verify=` | [`async_evolve`](#the-barrier-free-runtime-async_evolve) | async only: a worker re-runs its trajectory with the diff applied for a local before/after signal; faithful ports that score the candidate on held-out only pass `False` | `True` |
| `on_round=` | driver | **progress callback** — fires per round / merger sweep | `None` |
| `target_reward=`, `patience=` | driver | **early stopping** — stop at a reward, or after N rounds without improvement | `None`, `None` |
| `max_worker_errors=` | driver | how much total failure to tolerate — only while *no* worker has ever succeeded | `3` |
| `oracle_budget=`, `cheap_eval_tasks=` | [verifier](verifier.md) | audit budget for L1 merges, and how many held-out tasks the *ranking* passes score | `200`, all |
| `repo_path=` | [ledger](ledger.md) | where the git-backed store lives — pass the same path again to **resume** | scratch, removed on return |
| `reward` | [rewards](rewards.md) | `(task, output) -> [0, 1]` — ready-made scorers for the common cases | — (required) |

The building blocks in detail:

---

## 1. The actor — `agent=` (or `run=` + `propose=`)

*What:* the thing that runs a task against the current artifact and, on a
failure, proposes an improvement. *Module:* [`agentdescent.agents`](agents.md)
provides the provider-agnostic **completion** (`prompt -> text`); `LLMAgent`
adapts a completion into the two-method actor.

```python
from agentdescent import claude, openai_compatible, from_callable
from agentdescent import LLMAgent

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

!!! important "The framework never injects the artifact into your prompt — you do"
    `run(rendered, task)` hands you `rendered`, the current artifact as text. Where
    it goes is entirely your call: a system prompt, a prefix, a few-shot block, a
    tool description. Nothing is inserted behind your back.

    ```python
    evolve(tasks, reward,
           run=lambda rendered, task: model(f"{rendered}\n\nQ: {task.prompt}"),
           #                                  ^^^^^^^^ you decide where it lands
           propose=reflector(model),
           strategy=SingleSlot(initial_value="You are a helpful assistant."))
    ```

    So "what evolves" is set by two things together: `strategy=` fixes the
    artifact's *shape*, and `run=` decides how that shape reaches the model. The
    same `SingleSlot` is a system prompt or a tool description depending only on
    where you interpolate it.

```python
from agentdescent import AppendRules, KeyedRules

evolve(tasks, reward, agent=agent, strategy=AppendRules())                       # default
evolve(tasks, reward, agent=agent, strategy=KeyedRules(categories=["route","fmt"]))
```

| Strategy | Rule |
|---|---|
| **`SingleSlot`** | the artifact **is one value** (a system prompt, an instruction) and each accepted proposal replaces it — the most common case |
| `AppendRules` | each proposal → a content-addressed rule; identical ones dedupe, complementary ones **fuse** (append-only) |
| `KeyedRules(categories)` | one entry per category; competing proposals for the same category **contradict** and are resolved on held-out score |
| `FileTree(files)` | the artifact **is a directory**: one entry per *file*, so two workers editing different files fuse and two editing the same file are resolved — see [evolving a directory](directory-evolution.md) |

`FileTree` is the one strategy whose artifact does not reach the model through the
prompt: the rendered tree is written to a throwaway workspace and a real agent
reads it off disk (`tree_runner`). Everything else on this page applies unchanged.

Write your own by implementing three methods (`initial` / `render` / `to_diff`):

```python
from agentdescent import Diff

class OneValue:                       # this is `SingleSlot`, written out longhand
    def initial(self): return {}
    def render(self, state): return state.get("v", "(none)")
    def to_diff(self, state, proposal, author, base_version, target):
        if state.get("v") == proposal: return None   # None -> propose nothing
        return Diff(diff_id=f"{author}:{base_version}", target=target,
                    ops={"v": proposal}, author=author)

evolve(tasks, reward, agent=agent, strategy=OneValue())
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
| `SkillLibraryTree` | [EvoSkill](algo-evoskill.md) | a **directory** of `SKILL.md` skills, one per subdirectory (a `FileTree` subclass keeping the repo's `name :: body` proposal protocol) |
| `SkillDocStrategy` | [SkillOpt](algo-skillopt.md) | one markdown skill doc mutated by bounded `append/insert_after/replace/delete` edits |
| `AgentDesignStrategy` | [ADAS](algo-adas.md) | one agentic-system design (a control-flow program) each proposal replaces |
| `HarnessStrategy` | [DGM](algo-dgm.md) | a coding-agent harness's capability set (a proposal adds one) |

---

## 3. The parallelism method — `parallel=`

*What:* how each round's tasks are partitioned across the `n_workers`. *Module:*
[`agentdescent.parallel`](parallelism.md).

```python
from agentdescent import DataParallel, TensorParallel

evolve(tasks, reward, agent=agent, parallel=DataParallel())                # default (shard tasks)
evolve(tasks, reward, agent=agent,                                         # disjoint sections
       strategy=KeyedRules(categories=CATS),
       parallel=TensorParallel(n_sections=4, route=category_of))
```

`TensorParallel` splits the **artifact** into disjoint sections, one per worker, so
the merge is a conflict-free union. It needs a strategy with a fixed key space
(`KeyedRules`; `AppendRules` content-addresses its keys and is refused), and
`route=` maps a task to the artifact key its failure will edit so each worker only
sees tasks it may act on. Out-of-section proposals are rejected and counted as
`section-violation` in [`result.outcomes()`](#what-evolve-returns). See
[Parallelism](parallelism.md).

`PipelineParallel` is **not** an `evolve()` mode — it needs one artifact per stage
and `evolve()` evolves one, so passing it raises. Its stage ordering and blame
attribution live in `agentdescent.parallel.PipelineChain`.

Or your own — implement `plan(n_workers, round_index, keys) -> [WorkUnit]`:

```python
from agentdescent import WorkUnit

class Blocks:
    name = "block"
    def plan(self, n_workers, round_index, keys):
        keys = list(keys); size = (len(keys)+n_workers-1)//n_workers
        return [WorkUnit(worker=i, keys=keys[i*size:(i+1)*size]) for i in range(n_workers)]

evolve(tasks, reward, agent=agent, parallel=Blocks())
```

Details + the DP/TP/PP semantics: [Customizable parallelism](parallelism.md).

---

## Task selection — `task_sampler=`

*What:* which task a worker rolls out next, from the shard
[`parallel=`](parallelism.md) gave it. *Module:*
[`agentdescent.sampling`](sampling.md).

A rollout is the expensive unit of work. Spending it on a task the agent already
solves teaches the system nothing — no failure, no proposal, no diff — and the
same is true of one it can never solve. Only tasks in between carry a usable
gradient (the GRPO zero-advantage argument).

```python
from agentdescent import DifficultyWeighted, RoundRobin

evolve(tasks, reward, agent=agent, task_sampler=RoundRobin())          # default
evolve(tasks, reward, agent=agent, task_sampler=DifficultyWeighted())  # focus the budget
```

| Sampler | Rule |
|---|---|
| **`RoundRobin`** (default) | cycle through the shard — deterministic, but spends rollouts uniformly |
| **`DifficultyWeighted`** | weight by `4·p·(1−p)`, plus a UCB bonus so untried tasks are still explored |

The `c` sweep, the `pass_threshold` trap for graded scorers, how to write your
own, and the **caveat that the measured gain is a targeting number rather than an
accuracy claim** are all on [the sampling page](sampling.md).

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
from agentdescent import AggregatorConfig, Aggregator

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
from agentdescent import get_policy

evolve(tasks, reward, agent=agent, staleness_policy=get_policy("reflective"),
       refresh_interval=3)          # <- without this there is no staleness to handle
```

!!! warning "On the synchronous path, `refresh_interval` is what makes this knob do anything"
    `evolve()` snapshots at the top of every round, so by default every worker
    proposes against the current head and `η` is 0 by construction — all three
    policies then behave identically. `refresh_interval=N` lets a worker keep its
    snapshot for N rounds, staggered by worker id, which is what produces a spread
    of `η`. On the barrier-free path `async_ratio` already does this. See
    [staleness](staleness.md).

| Policy | Behaviour |
|---|---|
| `full` | use stale diffs as-is (max throughput) |
| `guarded` | version-gated: accept `η=0`, rebase `η≤α`, discard beyond (default) |
| `reflective` | always rebase + re-verify; discard only if the gain no longer holds |

Staleness bites when workers lag head — which is most visible in the **async
runtime** (`async_ratio`), below. In synchronous `evolve()` each round proposes
against the current head, so η is usually 0. Deep dive: [staleness policies](staleness.md) for the module,
[concepts §3](concepts.md#3-staleness) for why it is the heart of the async story.

---

## What `evolve` returns

```python
result = evolve(tasks, reward, agent=agent, rounds=6, verbose=True)

result.rendered       # the evolved artifact, rendered to text
result.state          # its {key: value} state
result.final_reward   # held-out reward of the final artifact
result.history        # RoundInfo per round — reward, size, and what the merge did
result.outcomes()     # {'below-threshold': 7, 'committed': 2} — why the run went as it did
result.ledger_log     # the git commit log of accepted merges
result.error          # None on a clean run; "<ExcType>: <msg>" if a backend failure ended it
```

### Why did nothing commit?

The first question about a disappointing run, and `committed`/`rejected` cannot
answer it — the fixes are opposite. `outcomes()` tallies the merge outcome of
every round by a stable category:

| category | what happened | where to look |
|---|---|---|
| `committed` | accepted and written to the dev branch | — |
| `below-threshold` | reached the acceptance gate and failed to beat the baseline | the **reflector** — its proposals do not help. Check it can see enough (`Task.meta`), and that it is not returning empty (`max_tokens`) |
| `all-stale` | never reached the gate; the world moved on first | the **lag budget** — lower `async_ratio`, or use the sync path |
| `oversized` | outside the trust region: too many ops, or one value too long | the **reflector** again, but the opposite fix — it is emitting whole documents where a rule was wanted. Raise `trust_region_ops` / `trust_region_chars` only if the size is genuinely intended |
| `cas-conflict` | lost a commit race; the evidence is re-filed for retry | usually self-correcting; persistent means too many workers on one artifact |
| `oracle-rejected` | the audit's oracle disagreed with the cheap evaluator | the **cheap evaluator** is miscalibrated |
| `unknown-artifact` | diffs targeted an id the ledger does not hold | a caller bug in `artifact_id`/strategy |
| `section-violation` | tensor parallelism only: a worker edited a key outside its own section | the strategy/`route=` pairing — see [§3](#3-the-parallelism-method-parallel) |

`RoundInfo.reasons` is the same tally per round. A custom aggregator sees the
underlying `MergeReport`, which carries both `category` and a human-readable
`reason` with the measured values (`"P(delta>0)=0.42 <= 0.75"`) — good for a log
line, but it interpolates numbers, so count on `category`.

### What the merge did, per round

Categories say *why* a round ended as it did; these four say *how it got there*.
They come straight off the `MergeReport`s the round produced.

| field | question it answers |
|---|---|
| `considered` | how many evidence cards the merge looked at — the **denominator** for the next two |
| `discarded_stale` | how many the staleness filter dropped. Rising, with a flat reward, is the lag budget rather than the reflector |
| `conflicts_dropped` | how often two workers proposed different values for one key. Zero under `AppendRules` (content-addressed keys rarely collide), interesting under `KeyedRules` / `SingleSlot` / `FileTree` |
| `fused` | commits whose winning candidate was the **fusion** of several diffs rather than any single one — the model-soup question. Counted only when it committed: the tournament builds a fused candidate whenever the survivors are complementary, so counting the ones it *built* says nothing about whether combining beat picking |

```python
for h in result.history:
    print(h.round, h.considered, h.discarded_stale, h.conflicts_dropped, h.fused)
```

They were computed all along and thrown away — the reference runtimes reported
them (`RoundStat.fused`, `AsyncStats.conflicts_dropped`) and the engine every
real workload uses did not.

Watch a long run as it happens — an LLM run can take hours, and `history` is only
available once it returns:

```python
evolve(tasks, reward, agent=agent, rounds=20,
       on_round=lambda info: print(info.round, info.held_out_reward))
```

`on_round` fires per round (per merger sweep on the async path, where it runs on
the merger thread and so must be cheap and thread-safe). An exception inside it is
reported as a warning and never aborts the run.

!!! note "`history` counts rounds on the sync path, merger sweeps on the async one"
    Same field, different unit. Synchronous `evolve(rounds=5)` yields exactly 5
    entries. `async_evolve` appends one per **non-empty merge**, so the count
    depends on how fast the workers produce — a 3-second run with a fast reward
    produced 221 — and it is not bounded by any parameter. `RoundInfo.round` is the
    sweep index there, not a round number. Compare `final_reward` across paths, not
    `len(history)`.

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

!!! note "The train/held-out split is positional"
    The last `held_out_frac` of `tasks` is held out, **in the order given** — no
    shuffle. That keeps `Dataset.val_frac`'s promise (the engine's held-out split
    is exactly that `Dataset`'s `val`, which only holds because `trainval` is
    train + val in that order) and keeps a seeded run reproducible.

    It is the wrong default for **grouped** data — anything ordered by category,
    source, difficulty or date, which is most benchmarks loaded raw through
    `hf_rows`. On a 20-task set whose first 12 are class `a`:

    ```
    shuffle=False   held-out classes: ['b']        a/b: 0/8
    shuffle=True    held-out classes: ['a', 'b']   a/b: 5/3
    ```

    Every gate in the run — the acceptance test, `target_reward`, `patience`,
    `final_reward` — is measured on that set, so pass `shuffle=True, seed=...`
    (or pre-split with `dataloader.split_dataset`, which shuffles and can
    stratify). A held-out set of fewer than 4 tasks now warns: at 1 item
    `final_reward` is 0.0 or 1.0 and nothing in between.

Omit `repo_path` and the ledger is a throwaway directory, removed when `evolve()`
returns — not held until the interpreter exits, so a notebook or a parameter sweep
does not accumulate one git repo per run. A process killed outright (SIGKILL, OOM)
skips that cleanup; the next run in a fresh process collects anything older than a
day.

The ledger also runs git with **its own configuration**, ignoring your
`~/.gitconfig` and `/etc/gitconfig`. These are the ledger's internal bookkeeping
commits in a directory you never see — `commit.gpgsign = true` or a global
`core.hooksPath` used to fail them, and with it the whole run, before a single
task had executed.

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

!!! warning "`error` cannot tell convergence from a spent budget — `stop_reason` can"
    A run that reached `target_reward` and a run that ran out of budget both come
    back with `error=None` and a populated `history`:

    ```python
    result.stop_reason   # "target_reward" | "patience" | "rounds"
                         # | "max_seconds" | "max_iters" | "error"
    ```

    This matters most under `asynchronous=True`, where **`max_seconds=None` means
    20 seconds**, not "unbounded" as it does on the synchronous path — flipping one
    boolean could truncate a run and the result looked converged. Both that and
    `rounds` changing meaning (it becomes a `rounds × n_workers` rollout budget,
    and `RoundInfo.round` becomes a merger-sweep index) now emit a
    `RuntimeWarning`.

!!! note "Three failure categories, not two"
    | category | example | what happens |
    |---|---|---|
    | **caller contract** | `reward` returns `47`, `propose` returns an `int` | raises (`ContractError`) — the run is meaningless, so failing fast is the only useful answer |
    | **backend** | 429, dead endpoint, credit exhausted | absorbed, retried, tolerated; ends the run only when nothing can make progress, and then `error` names it |
    | **ledger** | a held `index.lock`, a full `$TMPDIR`, a killed `git` | ends the run, but still returns the artifact evolved so far with `error` naming git |

    The third used to escape as a bare `GitError`, discarding a completed run —
    including when the failing call was only fetching the cosmetic
    `result.ledger_log`, which now degrades to `[]` rather than taking the result
    with it.

**Actor signatures are checked before the first rollout.** `run` and `propose` are
bound-tested up front, so a plain typo (a `propose` missing its `reward`
parameter) raises `TypeError` immediately instead of surfacing as a
backend-shaped failure with zero rounds run and an empty artifact.

**Backend failures are tolerated, not fatal.** In the async runtime a transient
error (a rate limit, a flaky endpoint) is retried with exponential backoff. What
happens next depends on a **global** signal — has *any* worker ever completed a
rollout?

| | what it means | response |
|---|---|---|
| nothing has ever succeeded | misconfiguration: wrong key, dead endpoint | each worker retires after `max_worker_errors=3` consecutive failures; when all have, the run ends and `result.error` names the failure |
| something succeeded, now failing | a transient: the backend demonstrably works | **no one retires** — back off and keep trying until the run's own budget ends it. A `RuntimeWarning` names the worker so a backend dying mid-run is not mistaken for idleness |

The signal is global on purpose. Keyed on each worker's own history instead, an
intermittent backend retires whoever loses its first few rolls — at a 2-in-3
failure rate that is about 30% of workers, none of which were faulty.

!!! warning "Shedding workers cannot fix a throttled backend"
    Every worker shares one backend, so retiring workers over rate limits reduces
    throughput without relieving the limit, and then ends the run. Measured against
    a backend refusing 1 call in 3 (~56% per rollout, an ordinary 429 storm), the
    old blanket rule retired all three workers in **22 s with nothing learned**.

**The synchronous path had the same disease in a worse form.** A worker's
exception propagated out of its future and broke the round loop, so a *single*
transient ended the run — measured, one 429 on call 5 turned a 20-round run into
**0 rounds**, and sync is the default. A failing worker now costs its own
evidence and nothing more; the round merges what the others gathered. The
give-up rule is the same global one, counting consecutive rounds in which
*every* worker failed. A dead backend still ends the run in well under a second.

The per-round held-out scoring sat *outside* that handling, and it runs the agent
too — so a blip there raised straight out of `evolve()`, discarding everything
already committed. It is now treated like a failed round: the last known reward
carries forward so early stopping still has something to compare.

**Every *evaluation* is retried at one choke point.** A held-out score runs the
agent, so it is a backend call — and the engine makes them in more places than is
obvious: each round's measurement, the final measurement, and the aggregator's own
accept/reject comparisons (`cheap_eval`, `eval_counts`, `oracle_eval`). A
they all funnel through one memoised evaluation, which retries there — so a retry
re-runs only the task that actually failed, and every call site is covered at
once.

The **merger** gets the same tolerance, and this matters more than it sounds: it
scores the held-out set every sweep, so it calls the backend too. A single
try/except around its loop made it a single point of failure that one transient
took out permanently — the run then reported `0 sweeps` while every worker was
still healthy. It now retries with a short backoff and never ends the run by
itself, because the two cases that *should* end one are already covered: a dead
backend retires the workers, and a broken aggregator or reward raises
[`ContractError`](aggregator.md), which propagates rather than being absorbed.

`result.retired_workers` counts workers that gave up. A run can finish *cleanly*
at a fraction of its requested concurrency, with `error` still `None` — check it
before reading a fast run as a healthy one.

---

## Putting it all together

The one complete, runnable example threads every block above on a real dataset
with a real LLM:

```python
from agentdescent import claude
from agentdescent import evolve, LLMAgent, AppendRules
from agentdescent import DataParallel

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

!!! tip "Stop paying once it has converged — `target_reward` / `patience`"
    A run spends all `rounds` by default, including after the artifact stops
    changing. On a workload that converges in two rounds, 20 rounds cost 141 model
    calls for a result reached at 69 — **51% of the budget bought nothing**.

    ```python
    evolve(tasks, reward, agent=agent, rounds=50,
           target_reward=0.95,   # stop as soon as held-out reaches this
           patience=5)           # ...or after 5 rounds with no improvement
    ```

    Both work under `asynchronous=True` too. There are no round barriers there, so
    `patience` counts **merge sweeps** (one drain-and-merge by the merger) rather
    than rounds.

!!! note "An abandoned straggler keeps running"
    Python cannot kill a thread, so a rollout abandoned by `round_timeout` runs to
    completion in the background. The round is bounded; the *work* is not. Its late
    evidence carries the version it was built against, so the staleness filter
    judges it like any other stale diff rather than applying it to a newer artifact.

    Rounds run on daemon threads (with a semaphore preserving `max_concurrency`),
    so an abandoned rollout never holds the interpreter open at exit: a rollout
    wedged for 600 s still lets the process exit in **4.5 s**.

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

`evolve()`'s round barrier makes the aggregator wait for every worker each round.
[`async_evolve()`](async.md) removes it while accepting the identical
`run` / `reward` / `propose` / `strategy` / `aggregator_factory` plug-ins — so
**any** task that runs under `evolve()` (ACE, GEPA, EvoSkill, SkillOpt, ADAS, DGM)
also runs async:

```python
from agentdescent import async_evolve, get_policy

result = async_evolve(tasks, reward, agent=agent,
                      n_workers=4, async_ratio=3, max_seconds=30,
                      staleness_policy=get_policy("reflective"))
```

Reach it via `evolve(asynchronous=True)` or directly. Small `async_ratio` →
near-synchronous, few stale diffs; large → highly asynchronous, many stale diffs
for the policy to rebase or discard.

!!! warning "Five arguments change meaning under `asynchronous=True`"
    Three are ignored (`parallel`, `max_concurrency`, `round_timeout`) and two are
    **redefined**: `rounds` becomes a budget of `rounds × n_workers` rollouts, and
    `max_seconds=None` becomes 20 seconds where it meant "no limit". Each warns,
    but check `result.stop_reason` — a budget expiry otherwise looks exactly like
    convergence. The full table, the pipeline's cold-start and backpressure
    behaviour, and what only the async path can report are on
    [the async page](async.md).

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

### The reference async orchestrator

For the router reference domain there is also `AsyncAgentDescent` — the original
stage-orchestration runtime, with duration-aware straggler checkpointing, and the
thing the parallelism claims were measured with. Same aggregator, staleness and
governance underneath. See [async](async.md#asyncagentdescent-the-reference-runtime),
[the reference orchestrator and domain](orchestrator.md), and the measured
trade-offs in [efficiency experiments](efficiency.md).
