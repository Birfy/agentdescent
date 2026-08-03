# AgentDescent

> **Gradient descent — but the parameters are agents.** A parallel, asynchronous
> framework for self-evolving agents (skills, prompts, harnesses) where **diffs
> are the gradients** and **the aggregator is the optimizer**.

[![PyPI](https://img.shields.io/pypi/v/agentdescent)](https://pypi.org/project/agentdescent/)
[![tests](https://github.com/Birfy/agentdescent/actions/workflows/tests.yml/badge.svg)](https://github.com/Birfy/agentdescent/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-1f6feb)](https://birfy.github.io/agentdescent/)
[![CI](https://github.com/Birfy/agentdescent/actions/workflows/docs.yml/badge.svg)](https://github.com/Birfy/agentdescent/actions/workflows/docs.yml)
[![python](https://img.shields.io/badge/python-%E2%89%A53.9-1f6feb)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-3fb950)](https://github.com/Birfy/agentdescent/blob/main/LICENSE)

AgentDescent puts the deep-learning **training stack** on top of agents. The
"parameters" are a **library of evolvable artifacts** (skills, prompts, harness
modules, verifiers); the "gradients" are **diffs carrying evidence cards**; the
"optimizer step" is a **merge decision**. *N* workers propose diffs **in
parallel**, and a **barrier-free asynchronous** merger aggregates them into a
shared, version-controlled artifact library — targeting **O(N / T_iter)**
improvement throughput, where serial self-improvement is bounded at 1 diff / T_iter.

> The one place the analogy *must* break defines the whole system: **gradients
> add, diffs do not.** Aggregation is therefore not averaging but **conflict
> resolution + statistical acceptance + transactional commit**.

## Highlights

- **One entry point — `evolve()`.** Describe *what evolves* (a `Strategy`) and the
  *rules of evolution* (`run` / `reward` / `propose`); the parallel, merge-based
  loop (ledger → workers → aggregator → commit) runs for you.
- **Parallel *and* asynchronous.** Concurrent workers within a round
  (`max_concurrency`) and a **barrier-free** async runtime across rounds
  (`asynchronous=True`) — a ROLL-Flash-style lag budget plus Full / Guarded /
  Reflective staleness policies keep stale diffs safe.
- **The aggregator is a discrete-space optimizer.** Staleness filter → conflict
  resolution → fusion tournament → Beta-posterior acceptance → transactional
  commit — and fully swappable via `aggregator_factory`.
- **Governed by blast radius.** Skills (L2) merge freely; harnesses/verifiers (L1)
  are forced through an oracle; safety/permissions (L0) are frozen.
- **Provider-agnostic.** Any `prompt -> text` is a completion — Claude,
  OpenAI-compatible endpoints (GLM / DeepSeek), or a tool-using agent (OpenHands).
- **Faithful algorithm ports.** Runnable, offline-tested examples of ACE, GEPA,
  EvoSkill, SkillOpt, ADAS, and DGM — faithful to each repo's algorithm *and*
  dataset choice.

## Install

```bash
pip install agentdescent
```

The core engine has **zero required dependencies** and needs only Python ≥ 3.9.
That gives you the library — `evolve()`, the aggregator, the agent layer, the
dataloader.

**To run the examples, clone the repo.** They are research artifacts kept outside
the installed package (they would otherwise squat the top-level `examples` name),
so every `python -m examples.…` command below needs a checkout:

```bash
git clone https://github.com/Birfy/agentdescent && cd agentdescent
pip install -e ".[dev]"          # [dev] adds pytest, [docs] adds MkDocs Material
python -m examples.run_demo      # no API key needed
```

## Quickstart

**Have a dataset? One call.** `evolve_skill` supplies the boilerplate — wrapping
rows as tasks, the lambda that puts the skill in front of the question, the
scorer, the knobs — and leaves you the three decisions that are actually yours:
your data, how to score it, and which model.

```python
from agentdescent import evolve_skill
from agentdescent import openai_compatible
from agentdescent.dataloader import hf_rows

rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor", limit=40)

result = evolve_skill(rows, model=openai_compatible(model="deepseek-v4-flash"),
                      prompt="question", gold="answer", score="exact")

print(result.rendered)        # the skill it learned
print(result.final_reward)    # held-out reward
print(result.outcomes())      # why it went that way
```

It is a thin wrapper over `evolve()` — same engine, same result object — and any
extra argument passes straight through (`asynchronous=True`, a custom
`strategy=`, your own `run=`). Drop to `evolve()` the moment you want something
it does not express.

<details>
<summary>The same thing without the wrapper. Runnable as-is — no API key, no dependencies.</summary>

```python
from agentdescent import Task, evolve

tasks = [Task(id=f"t{i}", prompt=f"item {i}") for i in range(12)]

def reward(task, output):                  # must return [0, 1]
    return 1.0 if "2026" in output else 0.0

def run(rendered, task):                   # your solver
    return "answer" + (" 2026" if "year" in rendered else "")

def propose(rendered, task, output, reward):   # what to add on a failure
    return "always state the year"

result = evolve(tasks, reward, run=run, propose=propose,
                rounds=6, n_workers=3, max_concurrency=3)
print(result.rendered)        # the evolved artifact
print(result.final_reward)    # held-out reward -> 1.0
print(result.error)           # None on a clean run; check this!
```

Swap in a real model or agent by passing `agent=` instead of `run`/`propose` —
they are all the same contract:

```python
from agentdescent import LLMAgent, claude, openai_compatible, claude_code

evolve(tasks, reward, agent=LLMAgent(claude(model="claude-haiku-4-5")))
evolve(tasks, reward, agent=LLMAgent(openai_compatible(model="deepseek-v4-flash")))
evolve(tasks, reward, agent=LLMAgent(claude_code()))     # Claude Code CLI
# ...or run barrier-free: evolve(..., asynchronous=True, async_ratio=3)
```

</details>

## 📖 Documentation

Full docs live in [`docs/`](https://github.com/Birfy/agentdescent/tree/main/docs/) and render as a website via MkDocs Material:

| Page | What's in it |
|---|---|
| [Home](https://github.com/Birfy/agentdescent/blob/main/docs/index.md) | Overview and 30-second tour |
| **[Quickstart — dataset to skill](https://github.com/Birfy/agentdescent/blob/main/docs/quickstart-skill.md)** | **Start here.** One call: your data, how to score it, which model |
| **[Measured results](https://github.com/Birfy/agentdescent/blob/main/docs/results.md)** | Every empirical claim with the setup that produced it — including where there was nothing to learn |
| [Architecture](https://github.com/Birfy/agentdescent/blob/main/docs/architecture.md) | Components, data-flow diagram, the two runtimes, concurrency model |
| [Concepts](https://github.com/Birfy/agentdescent/blob/main/docs/concepts.md) | The training↔RSI analogy, staleness, the aggregator, the three long tails, governance |
|  [Run everything, and extend it](https://github.com/Birfy/agentdescent/blob/main/docs/usage.md) | Every demo with its output, config reference, **plugging in your own `Evolvable` domain** |
| [Evolving anything](https://github.com/Birfy/agentdescent/blob/main/docs/evolution.md) | The general engine — evolve any artifact by writing its `Strategy` + `run`/`reward`/`propose` |
| [Connecting agents & LLMs](https://github.com/Birfy/agentdescent/blob/main/docs/agents.md) | The provider-agnostic completion layer |
| [Loading datasets](https://github.com/Birfy/agentdescent/blob/main/docs/dataloader.md) | The `agentdescent.dataloader` data layer — HF datasets-server + raw-file fetch, cached, dependency-free |
| [Customizable parallelism](https://github.com/Birfy/agentdescent/blob/main/docs/parallelism.md) | Pluggable DP / TP / PP strategies — or write your own |
| [Duration-aware scheduling](https://github.com/Birfy/agentdescent/blob/main/docs/duration-scheduling.md) | Estimate rollout cost from task size; LPT dispatch + straggler checkpointing |
| [Efficiency experiments](https://github.com/Birfy/agentdescent/blob/main/docs/efficiency.md) | Measured parallel scaling and async tail-hiding |
| [Example: skill evolution](https://github.com/Birfy/agentdescent/blob/main/docs/skill-evolution.md) | One complete run — real dataset, real LLM, every module |
| [Self-evolution algorithms](https://github.com/Birfy/agentdescent/blob/main/docs/self-evolution-examples.md) | Faithful ports of ACE, GEPA, EvoSkill, SkillOpt, ADAS, DGM |

```bash
pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static HTML into ./site
```

A GitHub Actions workflow ([`.github/workflows/docs.yml`](https://github.com/Birfy/agentdescent/blob/main/.github/workflows/docs.yml))
builds and deploys the site to GitHub Pages — enable it under *Settings → Pages
→ Source: GitHub Actions*.

## Evolve anything — the general engine

The core is the ledger + aggregator + schedulers + governance.
[`agentdescent.evolution`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py) is the domain-agnostic engine on
top: describe **what evolves** (a `Strategy`) and the **rules of evolution**
(`run` / `reward` / `propose`), and it runs the parallel, merge-based loop.

```python
from agentdescent import evolve, AppendRules

result = evolve(
    tasks, reward,
    agent=my_agent,           # or run=/propose= plain functions
    strategy=AppendRules(),   # or KeyedRules / your own
    blast_radius=0.2,         # 0.2 = L2 skill; 0.6 = L1 harness/verifier
    rounds=15, n_workers=4,
)
print(result.rendered, result.final_reward)
```

The strategy maps a proposal into diff ops, so distinct edits **fuse** and
conflicting edits are **resolved** on held-out score — for free. `blast_radius`
picks the governance layer (a skill is L2; a harness/verifier at `0.6` is L1,
where merges are forced through the oracle). Same `evolve` call for either —
only the artifact, strategy, and blast radius differ.

**Connect any agent/LLM** — [`agentdescent.agents`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/agents.py) is the
separate provider layer; any `prompt -> text` is a completion (`claude(...)`,
`openai_compatible(...)` for GLM/OpenAI-style endpoints, `from_callable(...)`,
`with_retries(...)`).

The one complete end-to-end run — real dataset, real LLM, every module — is
[`examples/skill_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skill_evolution.py)
(`python -m examples.skill_evolution --dry-run` for the no-API preview).
Guides: [the engine](https://github.com/Birfy/agentdescent/blob/main/docs/evolution.md) · [skill example](https://github.com/Birfy/agentdescent/blob/main/docs/skill-evolution.md)
· [agents](https://github.com/Birfy/agentdescent/blob/main/docs/agents.md).

## Evolve a **directory** — a skill folder, an agent folder, its code

Everything above evolves text that ends up *in a prompt*. `evolve_skill_dir`
evolves a **directory**, and each rollout is performed by a real agent that reads
those files off disk with its own tools:

```python
from agentdescent import evolve_skill_dir
from agentdescent.agents import claude_code, openai_compatible

result = evolve_skill_dir(
    "~/.claude/skills/pdf-audit", rows,
    agent=claude_code(extra_args=["--permission-mode", "acceptEdits"]),
    reflect_with=openai_compatible(model="deepseek-v4-flash"),
    prompt="question", gold="answer", score="contains")

result.write_to("~/.claude/skills/pdf-audit")     # opt in; backs up first
```

Each rollout materialises the candidate into a throwaway workspace at
`.claude/skills/<name>/`, stages the task's fixtures beside it and runs the agent
there. The optimizer is untouched: **state keys are file paths**, so two workers
editing different files *fuse* and two editing the same file are *resolved* on
held-out score — the same machinery as every other strategy.

Three entry points, differing only in governance and what guards them:
`evolve_skill_dir` (L2), `evolve_agent_dir` (L1 — an agent definition is a
harness, so every merge also passes the oracle), and `evolve_agent_code`, where
the tree is **executed** behind a frozen test suite that the candidate cannot
rewrite (pristine files are overlaid after materialisation, so weakening the
tests at run time does not work either).

```bash
python -m examples.skill_dir_evolution        # offline, no API key
```

Guide: [evolving a directory](https://github.com/Birfy/agentdescent/blob/main/docs/directory-evolution.md)
· [design record](https://github.com/Birfy/agentdescent/blob/main/docs/design-directory-evolution.md).

## Faithful ports of the latest self-evolution algorithms

To show the engine is faithful to the field, AgentDescent ships one runnable example
per representative **skill** and **harness** self-evolution algorithm — each
faithful to the original repo's *algorithm* and *dataset choice*, each with a
`--dry-run` (no-API) mode and an offline test suite. Every one loads its benchmark
through the shared [`agentdescent.dataloader`](https://github.com/Birfy/agentdescent/blob/main/docs/dataloader.md) data layer
(HF datasets-server + raw files, cached, dependency-free). Full guide:
[docs/self-evolution-examples.md](https://github.com/Birfy/agentdescent/blob/main/docs/self-evolution-examples.md).

| Algorithm | Kind | Dataset | Example |
|---|---|---|---|
| **ACE** (Agentic Context Engineering) | skill / context | FiNER-139 | [`ace_context_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/ace_context_evolution.py) |
| **GEPA** (Reflective Prompt Evolution) | skill / prompt | HotpotQA | [`gepa_prompt_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/gepa_prompt_evolution.py) |
| **EvoSkill** (Automated Skill Discovery) | skill library | OfficeQA | [`evoskill_skill_discovery.py`](https://github.com/Birfy/agentdescent/blob/main/examples/evoskill_skill_discovery.py) |
| **SkillOpt** (ReflACT) | skill document | SearchQA | [`skillopt_skill_training.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillopt_skill_training.py) |
| **ADAS** (Meta Agent Search) | harness | MGSM | [`adas_meta_agent_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/adas_meta_agent_search.py) |
| **DGM** (Darwin Gödel Machine) | harness | SWE-bench Verified | [`dgm_self_improve.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dgm_self_improve.py) |

```bash
python -m examples.ace_context_evolution --dry-run     # skill/context self-evolution (ACE)
python -m examples.dgm_self_improve                    # harness self-evolution (DGM), offline
```

Fidelity is to the **released code**, not just the paper (e.g. EvoSkill's frontier
is top-K aggregate, not per-instance Pareto — the example follows the code and
says so); where a full setup needs heavy infra (SWE-bench Docker, gated data), the
boundary is documented, never hidden.

## Efficiency (measured)

Two different numbers, and the difference is the point.

**Worker scaling alone** ([`examples/efficiency.py`](https://github.com/Birfy/agentdescent/blob/main/examples/efficiency.py))
is near-linear through 8 workers (~8.1×, efficiency ≈1.0), and the async pipeline
is **~2.6–2.9× faster than a sync barrier** under heavy-tailed rollout latency.

**A whole `evolve()` run** — rollouts *and* the merge gate — gets less, and how
much less depends entirely on your latency distribution:

| | overlap with `n_workers=8` |
|---|---|
| uniform latency | **5.9×** |
| heavy tail (a reasoning model) | **2.4×** — the round barrier waits on the slowest worker |
| ...same, barrier-free (`asynchronous=True`) | **3.0×** |

A real HotpotQA run measured 2.0×, squarely in the heavy-tail regime. The gate is
a separate axis: `eval_concurrency` took the same workload from **193.6 s to
90.0 s**. Full breakdown in
[docs/efficiency.md](https://github.com/Birfy/agentdescent/blob/main/docs/efficiency.md).

## The central analogy

| Model training | AgentDescent (parallel RSI) |
|---|---|
| parameter tensor θ | library of `Evolvable` artifacts |
| gradient *g* | `Diff` + `EvidenceCard` |
| parameter server | git-backed, version-vectored `Ledger` |
| optimizer step | `Aggregator` merge decision |
| per-param adaptive LR (Adam) | per-artifact Beta-posterior test |
| staleness / decoupled PPO | per-diff η + rebase re-verify |
| partial rollout | straggler detection (`ResumeQueue`; resume itself not implemented) |
| EMA (weight averaging) | stable/dev dual branch |
| training code (not self-modifiable) | L0 frozen layer |

## Running the examples

```bash
# RQ1 — merge vs fork, end to end (synchronous DP)
python -m examples.run_demo

# Async stage orchestration — Full/Guarded/Reflective policies + async_ratio sweep
python -m examples.run_async

# The flagship: evolve a skill on a real dataset with a real LLM (--dry-run: no API)
python -m examples.skill_evolution --dry-run

# Evolve a skill DIRECTORY that a real agent reads off disk (offline by default)
python -m examples.skill_dir_evolution

# Efficiency: parallel throughput scaling + async vs sync-barrier tail-hiding
python -m examples.efficiency

# Customizable parallelism: DP / TP / PP (+ a custom strategy)
python -m examples.parallelism

# Duration-aware scheduling: online estimator + LPT dispatch + straggler checkpointing
python -m examples.duration_scheduling

# RQ2 — staleness tolerance sweep (alpha in {0,1,5,inf})
python -m examples.rq2_staleness

# tests
pytest
```

No external services or model APIs are required: the reference domain
([`agentdescent/domains/router.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/domains/router.py)) is a fully
deterministic keyword-router skill, so the entire parallel loop runs in-process
and is unit-tested — while still producing genuine diffs that measurably improve
a held-out metric.

## Architecture → code map

Every module cites the design section it implements.

| Component | Module | Design § |
|---|---|---|
| `Evolvable` unit, `Diff`, `EvidenceCard`, version vectors | [`evolvable.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolvable.py) | 3.2, 3.3 |
| Git-backed Ledger: version vectors, CAS, dual branch (2PC available, unused) | [`ledger.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/ledger.py) | 3.1, 4.5 |
| Aggregator: staleness → conflict → fusion → Beta accept → commit | [`aggregator.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/aggregator.py) | 4 |
| **Staleness policies: Full / Guarded / Reflective** | [`staleness.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/staleness.py) | 4.2 |
| **Async stage-orchestration runtime + `async_ratio`** | [`async_runtime.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/async_runtime.py) | 3.1 |
| **Parallel paradigms: DP / TP / PP** | [`parallel.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py) | 8 |
| Statistics: Beta posterior, `P(Δ>0)`, annealed δ, UCB | [`stats.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/stats.py) | 4.4, 5.2 |
| Three schedulers: UCB task / audit / resume queue | [`scheduler.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/scheduler.py) | 5 |
| Three-layer verifier (rule / learned / oracle) | [`verifier.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/verifier.py) | 3.1, 5.3 |
| Layered governance by blast radius (L0/L1/L2) | [`governance.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/governance.py) | 6 |
| Worker: rollout + propose | [`worker.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/worker.py) | 3.1 |
| Orchestrator (sync DP) + fork baseline | [`orchestrator.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/orchestrator.py) | 3.1, RQ1 |
| **Agent/LLM connection layer (provider-agnostic)** | [`agents.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/agents.py) | — |
| **General evolution engine + pluggable `Strategy`** | [`evolution.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py) | 3.2 |

## How aggregation works (the `Aggregator` pipeline)

Cards are bucketed **by artifact**. When a bucket triggers (batch size `B`, or a
`T_max` timeout so cold artifacts don't starve), the aggregator runs one
optimizer step:

1. **Staleness filter (§4.2)** — per-diff `η = max(head − base)` over touched
   artifacts. `η = 0` proceeds; `0 < η ≤ α` is **rebased and cheaply
   re-verified** (does the delta still hold on the new head?); `η > α` is
   discarded and its evidence *settled back* into the pool. `α` adapts to
   artifact heat; contract-breaking diffs force `α = 0`.
2. **Conflict resolution (§4.3)** — syntactic (hunk overlap) and semantic
   (contradictory ops) detection; contradictions are projected out PCGrad-style,
   keeping the better of the pair on a shared subset.
3. **Fusion tournament (§4.3)** — complementary diffs are fused (model-soup
   analogy) and run against the individual candidates on held-out data.
4. **Audit gate (§5.3)** — the merge decision is itself submitted to the
   `AuditScheduler`; high-blast-radius / low-trust merges are forced through the
   oracle, which can **veto outright** (`oracle-rejected`) *before* the
   acceptance test runs. The optimizer audits itself.
5. **Statistical acceptance (§4.4)** — commit only if
   `P(Δ > 0) > 1 − δ` under a per-artifact Beta posterior, not a point threshold.
   `δ` anneals with version (LR decay); a trust-region caps diff size.
6. **Commit (§4.1)** — compare-and-swap on `dev`, one artifact per merge
   (`commit_atomic`/2PC exists in the Ledger but no engine path calls it).
7. **Dual-branch promotion (§4.5)** — `dev → stable` after *K* **regression-free
   rounds** on dev (EMA-style confirmation; one round is one `step()`, and a
   commit *restarts* the clock rather than advancing it — so the artifact most
   likely to be promoted is the one that stopped changing because nothing beat
   it). A clean run publishes its head on the way out.

## Parallelism & asynchrony

AgentDescent ships two execution runtimes and a set of pluggable strategies, so a
run can be moved along the sync↔async and DP↔TP↔PP axes without touching the
merge pipeline.

### Two runtimes

- **Synchronous DP** ([`orchestrator.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/orchestrator.py)) — a round
  barrier: all workers step, then one `aggregator.step()`, then the next round.
  Deterministic; the RQ1/RQ2 baseline.
- **Asynchronous stage orchestration** ([`async_runtime.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/async_runtime.py),
  FlashEvolve-style) — **no barrier**. Worker threads keep producing evidence
  while a dedicated aggregator thread keeps merging, connected by the
  thread-safe `EvidenceBuffer`. The rollout/propose and aggregate/commit stages
  overlap instead of stalling.

### Staleness policies ([`staleness.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/staleness.py), FlashEvolve Full/Guarded/Reflective)

The active policy is the only thing that changes between async regimes — the
aggregator asks it `ACCEPT / REBASE / DISCARD` from each diff's `η` and `α`:

| Policy | Behaviour | Cost |
|---|---|---|
| **Full** | use stale diffs directly (η ignored) | max throughput, min safety |
| **Guarded** | version-gated: accept `η=0`, rebase `η≤α`, discard beyond | AReaL bounded-staleness |
| **Reflective** | always rebase + re-verify; discard only if the delta no longer holds | recovers otherwise-wasted proposals |

### `async_ratio` — the ROLL Flash lag budget

A worker refreshes its snapshot only once head has drifted more than
`async_ratio` versions ahead of it. Small ratio → near-synchronous, few stale
diffs; large ratio → highly asynchronous, many stale diffs the policy must
handle. A **backpressure** signal forces a global sync if the pipeline stalls
(evidence keeps arriving but nothing commits).

`python -m examples.run_async` shows the trade-off — all three policies converge
to 1.000, but at `async_ratio=4`:

| policy | rollouts | stale discarded | wall-clock |
|---|---|---|---|
| Full | ~8k | 0 | ~3.2s |
| Reflective | ~7.8k | ~0.7k | ~3.3s |
| Guarded | ~20k | ~17k | ~5.1s |

The **ratios** are the result; the absolute counts scale with the machine (a
slower host fits fewer rollouts into the same wall-clock window), so rerun it
rather than quoting these — same caveat as
[the efficiency numbers](https://github.com/Birfy/agentdescent/blob/main/docs/efficiency.md).

### DP / TP / PP ([`parallel.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py), §8)

- **DP (data parallel)** — same snapshot, task-sharded, diffs merged. The default
  the async runtime runs.
- **TP (tensor parallel)** — split one *hot* artifact into disjoint sections;
  each worker owns a section, so edits are conflict-free **by construction** and
  the merge is concatenation + a consistency reviewer (`TensorParallelMerge`).
- **PP (pipeline parallel)** — artifacts form a dependency chain; a downstream
  failure back-propagates blame to the earliest failing upstream stage
  (`PipelineChain.blame`, shared with the §7 counterfactual-replay attribution).

## The three long tails (§5)

AgentDescent treats "the long tail" as three separate problems:

- **L-traj** (system): heavy-tailed rollout durations → an online duration
  estimator, LPT dispatch, and **straggler detection**: a rollout that overruns
  its predicted cost is flagged and counted rather than being allowed to define
  the round's wall-clock. *Turn-level checkpoint-and-resume is **not**
  implemented* — `ResumeQueue` records stragglers but nothing resumes them, and
  doing so needs a resumable rollout contract the engine does not have (`run` is
  an opaque callable). The barrier-free
  [async runtime](https://github.com/Birfy/agentdescent/blob/main/agentdescent/async_evolve.py)
  is what actually stops one slow rollout from stalling the others today.
- **L-task** (data): Zipfian artifact triggering → **UCB over
  (cluster × artifact)** so starved tail artifacts get an exploration bonus, plus
  a **difficulty filter** that down-weights all-pass / all-fail groups (the
  zero-advantage argument). The same filter is available to `evolve()` as
  [`DifficultyWeighted` task sampling](https://github.com/Birfy/agentdescent/blob/main/agentdescent/sampling.py).
  *A dedicated tail canary set is **not** implemented* — held-out is a single
  split, not stratified into a canary.
- **L-value** (signal): most diffs are marginal → `AuditScheduler` spends the
  scarce oracle budget on `blast_radius × uncertainty / trust`.

## Governance (§6)

Artifacts sort into layers automatically by `blast_radius`:

- **L2 fast** — local skills/prompts → full async merge.
- **L1 slow** — harness/verifier → serialized in-flight changes + staged rollout.
- **L0 frozen** — oracle, audit budget, merge permissions, safety constraints →
  read-only to the loop. Without a frozen layer, the self-referential loop
  eventually pollutes itself (a verifier that learns to pass itself).

## Scope & honesty

This is a **research reference implementation**, not a production system. It is
faithful to the design's *mechanisms* and runs end-to-end on a synthetic domain
so the mechanisms are observable and testable. AgentDescent's novelty is a
**narrow, defensible engineering synthesis** — concurrent, staleness-bounded,
conflict-resolved **diff-level merge** over a git-backed versioned ledger — and
its throughput premise is a *testable engineering hypothesis*, not community
consensus (cf. FlashEvolve / SkillClaw / CoEvoSkills).
