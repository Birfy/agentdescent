# AgentDescent

> **Gradient descent — but the parameters are agents.** A parallel, asynchronous
> framework for self-evolving agents (skills, prompts, harnesses) where **diffs
> are the gradients** and **the aggregator is the optimizer**.

[![paper](https://img.shields.io/badge/paper-PDF-b31b1b)](https://github.com/Birfy/agentdescent/blob/main/paper/main.pdf)
[![PyPI](https://img.shields.io/pypi/v/agentdescent)](https://pypi.org/project/agentdescent/)
[![tests](https://github.com/Birfy/agentdescent/actions/workflows/tests.yml/badge.svg)](https://github.com/Birfy/agentdescent/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-1f6feb)](https://birfy.github.io/agentdescent/)
[![CI](https://github.com/Birfy/agentdescent/actions/workflows/docs.yml/badge.svg)](https://github.com/Birfy/agentdescent/actions/workflows/docs.yml)
[![python](https://img.shields.io/badge/python-%E2%89%A53.9-1f6feb)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-3fb950)](https://github.com/Birfy/agentdescent/blob/main/LICENSE)

📄 **Paper:** [*AgentDescent: Asynchronous Parallel Self-Evolution of LLM Agents by Merging
Conflicting Edits*](paper/main.pdf) — the design, the closed-form condition under which
merging can fire at all, and a live-model evaluation with every generating script named.
Source and per-result raw data are in [`paper/`](paper/) and [`bench/results/`](bench/results/).

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
- **Nineteen algorithm ports.** Runnable, offline-tested examples — eight
  benchmark-faithful (ACE, GEPA, EvoSkill, SkillOpt, ADAS, DGM, OpenEvolve, ERA)
  and eleven declared microports/analogues measured in the runtime matrix.

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

**Have a dataset?** There is one entry point, `evolve()`, and three building
blocks that turn a dataset into its arguments: `tasks_from` wraps rows as tasks,
`scorer` names a reward, `reflector` turns any model into the proposer. The
decisions that are actually yours — your data, how to score it, which model —
are the ones you still make.

```python
from agentdescent import SingleSlot, evolve, openai_compatible, reflector, scorer, tasks_from
from agentdescent.dataloader import hf_rows

rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor", limit=40)
model = openai_compatible(model="deepseek-v4-flash")

tasks = tasks_from(rows, prompt="question", gold="answer")     # rows -> Task objects
run = lambda skill, task: model(f"{skill}\n\n{task.prompt}")   # the skill meets the question

result = evolve(tasks, scorer("exact"), run=run, propose=reflector(model),
                strategy=SingleSlot(initial_value="You are a helpful assistant."),
                rounds=8, n_workers=8, max_concurrency=8, held_out_frac=0.3,
                patience=3, target_reward=0.98)

print(result.rendered)        # the skill it learned
print(result.final_reward)    # held-out reward
print(result.outcomes())      # why it went that way
```

Everything in that call is an ordinary `evolve()` argument, so there is nothing
to drop down to: swap `strategy=` for a different artifact shape, add
`asynchronous=True`, hand it your own `run=`.

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
| [Start here — the plugin in three commands](https://github.com/Birfy/agentdescent/blob/main/docs/plugin-quickstart.md) | Install, `agentdescent demo`, then the plugin inside your agent |
| [Use it from your agent](https://github.com/Birfy/agentdescent/blob/main/docs/plugins.md) | The plugin: DeepSeek Harness, Claude Code, Codex — spec, CLI, MCP server, evolving plugins themselves |
| [Loading datasets](https://github.com/Birfy/agentdescent/blob/main/docs/dataloader.md) | The `agentdescent.dataloader` data layer — HF datasets-server + raw-file fetch, cached, dependency-free |
| [Customizable parallelism](https://github.com/Birfy/agentdescent/blob/main/docs/parallelism.md) | Pluggable DP / TP / PP strategies — or write your own |
| [Where rollouts run](https://github.com/Birfy/agentdescent/blob/main/docs/execution.md) | The executor seam: threads, supervised worker processes, and describing a rollout as data |
| [Sandboxes](https://github.com/Birfy/agentdescent/blob/main/docs/sandboxes.md) | Workspace leases, one ceiling across processes, and the three isolation levels |
| [Duration-aware scheduling](https://github.com/Birfy/agentdescent/blob/main/docs/duration-scheduling.md) | Estimate rollout cost from task size; LPT dispatch + straggler checkpointing |
| [Efficiency experiments](https://github.com/Birfy/agentdescent/blob/main/docs/efficiency.md) | Measured parallel scaling and async tail-hiding |
| [Example: skill evolution](https://github.com/Birfy/agentdescent/blob/main/docs/skill-evolution.md) | One complete run — real dataset, real LLM, every module |
| [Self-evolution algorithms](https://github.com/Birfy/agentdescent/blob/main/docs/self-evolution-examples.md) | Nineteen algorithm ports, their fidelity classes, and every measured result in one table |
| [Runtime matrix](https://github.com/Birfy/agentdescent/blob/main/docs/matrix-overview.md) | The 11-method serial/sync/async scheduler comparison, with explicit fidelity boundaries |

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

Everything above evolves text that ends up *in a prompt*. The same `evolve()`
call evolves a **directory** when the strategy is a `FileTree` and the runner
is `tree_runner`: each rollout is performed by a real agent that reads those
files off disk with its own tools.

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

result.write_to(path)     # opt in; backs up first
```

Each rollout materialises the candidate into a throwaway workspace at
`.claude/skills/<name>/`, stages the task's fixtures beside it and runs the agent
there. The optimizer is untouched: **state keys are file paths**, so two workers
editing different files *fuse* and two editing the same file are *resolved* on
held-out score — the same machinery as every other strategy.

Three workloads, differing only in governance and what guards them: a skill
folder (`SKILL_BLAST_RADIUS`, L2), an agent folder (`HARNESS_BLAST_RADIUS`, L1 —
an agent definition is a harness, so every merge also passes the oracle), and
agent code, where `code_runner` **executes** the tree behind a frozen test suite
that the candidate cannot rewrite (pristine files are overlaid after
materialisation, so weakening the tests at run time does not work either) and
`gated_reward` scores a failed gate as zero.

```bash
python -m examples.skill_dir_evolution        # offline, no API key
```

Guide: [evolving a directory](https://github.com/Birfy/agentdescent/blob/main/docs/directory-evolution.md)
· [design record](https://github.com/Birfy/agentdescent/blob/main/docs/design-directory-evolution.md).

## Use it from your agent — DeepSeek Harness, Claude Code, Codex

The same engine as a **plugin**: a shared skill teaches the host agent when to
call it, an MCP server exposes `doctor / plan / start / status / show / apply`,
and a CLI mirrors them so a run started from an agent can be inspected from a
shell. Say *"evolve this skill against these examples"* inside the agent; it
writes a spec, shows you the plan and the cost, runs in the background, shows
you the diff, and asks before it applies.

```bash
pip install "agentdescent[mcp]"
agentdescent demo                      # a real evolution, offline, no key
agentdescent doctor                    # which agent CLIs and keys are here
agentdescent install dsh               # or: claude-code, codex, opencode
```

`demo` is the answer to *"show me it working"*: it builds a skill that is wrong
on purpose, twelve cases with known answers, and an agent that is a local
program — then runs the real loop and reports what it learned. No key, no cost,
nothing written to your own files. New here? Start with
[the plugin in three commands](https://github.com/Birfy/agentdescent/blob/main/docs/plugin-quickstart.md).

The spec is an `evolve()` call as data — every field is an ordinary argument or
a public building block, so there is no second API to learn:

```json
{"kind": "skill_dir", "target": "~/.claude/skills/pdf-audit",
 "data": {"path": "eval/cases.jsonl", "prompt": "question", "gold": "answer"},
 "score": "contains",
 "agent": {"ref": "claude_code", "extra_args": ["--permission-mode", "acceptEdits"]},
 "reflect": {"ref": "openai_compatible", "model": "deepseek-v4-flash"}}
```

`kind` is `text`, `skill_dir`, `agent_dir`, `agent_code` — or **`plugin`**: the
host plugins themselves (a DSH Cordis package, a Claude Code plugin directory)
are evolvable, loaded into an isolated copy of the host per rollout, with hooks
and permission config frozen and a recursion guard so a plugin evolving itself
cannot start a nested run.

Guide: [use it from your agent](https://github.com/Birfy/agentdescent/blob/main/docs/plugins.md)
· [design record](https://github.com/Birfy/agentdescent/blob/main/docs/plugin-design.md).

## Ports of the latest self-evolution algorithms

To show the engine is faithful to the field, nineteen published self-evolution
algorithms run on it — one runnable example each, every one with a `--dry-run`
mode and an offline test suite. **Eight** reproduce their paper's own benchmark;
**eleven** preserve the mechanism on a compact domain and carry a fidelity class
that says exactly which kind of port they are.

| | Algorithms |
|---|---|
| **Benchmark-faithful** | ACE (FiNER-139), GEPA (HotpotQA), EvoSkill (OfficeQA / FinQA), SkillOpt (SearchQA), ADAS (MGSM, GPQA), DGM (SWE-bench Verified ids, vendored bugs), OpenEvolve (function minimization), ERA (Kaggle Playground S3E1; the same search also runs the paper's integrals task and 2F1 against a 25-digit mpmath reference, both on suites constructed here, and LLM-SRBench, which is not) |
| **Mechanism microports** | PromptBreeder, AFlow, Self-Refine (GSM8K), Reflexion (GSM-Hard) |
| **Self-edit analogues** | SICA, Gödel Agent (GSM-Hard) |
| **Environment analogues** | Voyager (crafting world), SkillWeaver (settings site) |
| **Inference analogues** | Absolute Zero, R-Zero, Agent0 (self-play carts) |

```bash
python -m examples.ace.ace_context_evolution --dry-run     # skill/context self-evolution (ACE)
python -m examples.dgm.dgm_self_improve                    # harness self-evolution (DGM), offline
python -m examples.openevolve.openevolve_program_evolution --dry-run  # program evolution (OpenEvolve)
python -m examples.era.era_empirical_software --dry-run     # empirical-software tree search (ERA)
python -m examples.era.era_hard_integrals --dry-run         # the same search on hard integrals (ERA)
python -m examples.era.era_hypergeometric --dry-run         # ... and on 2F1, against a 25-digit reference (ERA)
python -m examples.era.era_llm_srbench --dry-run            # ... on LLM-SRBench equation discovery (ERA)
python -m examples.era.era_algotune --dry-run               # ... and on AlgoTune, scored in speedup (ERA)
```

Fidelity is to the **released code**, not just the paper (e.g. EvoSkill's frontier
is top-K aggregate, not per-instance Pareto — the example follows the code and
says so); where a full setup needs heavy infra (SWE-bench Docker, gated data), the
boundary is documented, never hidden. Analogues are labelled as analogues and must
not be cited as paper-benchmark reproductions.

Every port's measured result, with the run file behind it:
[all nineteen](https://github.com/Birfy/agentdescent/blob/main/docs/self-evolution-examples.md#measured-results-all-nineteen).
Full guide:
[docs/self-evolution-examples.md](https://github.com/Birfy/agentdescent/blob/main/docs/self-evolution-examples.md)
· per-port departures: [docs/port-fidelity.md](https://github.com/Birfy/agentdescent/blob/main/docs/port-fidelity.md).

## Efficiency (measured)

Two different numbers, and the difference is the point. Every row is produced by
[`examples/efficiency.py`](https://github.com/Birfy/agentdescent/blob/main/examples/efficiency.py),
named beside it, so it can be re-run rather than trusted.

| | result | command |
|---|---|---|
| Thread parallelism, 8 threads, real API calls | **5.8×** on `glm-5.2` (pure-Python CPU work: 1.1×) | `--only gil` |
| A whole `evolve()` run, uniform latency | **1.8×** of 8 workers, end-to-end | `--only distribution` |
| ...heavy-tailed latency (a reasoning model) | **1.7×** — the round barrier waits on the slowest worker | `--only distribution` |
| ...same, barrier-free (`asynchronous=True`) | **2.65×** on the dispatch microbenchmark | `--only async` |
| Gate concurrency (`eval_concurrency` 1 → 8) | **3.6 s → 1.2 s** | `--only gate` |

`n_workers` buys rollout parallelism and `eval_concurrency` buys gate
parallelism; they are independent, and a run slower than its worker count
suggests usually wants the second. Full breakdown in
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
| Worker role: rollout + propose — the `run`/`propose` callables, not a class | [`evolution.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py) | 3.1 |
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
3. **Fusion (§4.3)** — complementary diffs are fused (model-soup analogy) and the
   union goes to the gate. `fusion_tournament=True` runs it against the
   individual candidates on held-out data first, which costs a cheap sweep per
   candidate and is the only way to get a win rate; see [Does merging just
   average the improvements
   away?](#does-merging-just-average-the-improvements-away) below.
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

### Does merging just average the improvements away?

The sharpest objection to this whole design: two workers each make a local
improvement, and the two together are worse than either alone.

**What stops it is the acceptance gate, not the tournament.** The gate scores the
candidate on the *full* held-out set and refuses a measured regression, so a
fusion that made things worse never commits either way. Work out what the
tournament decides that the gate does not and it is one case — the fusion beats
the artifact but loses to one of the singles — and even that is recoverable,
because the union is a **superset** of every single diff, so committing it loses
no proposal. It merely carries some that looked negative this round, and the next
round proposes from there.

That is why the tournament is **off by default**: an unconditional cost of one
cheap sweep per candidate, every round, against a conditional and recoverable
gain. `evolve()` builds the union and hands it to the gate.

It stays available because it is the only thing that can *measure* the objection.
Turn it on with `fusion_tournament=True`:

```python
result = evolve(tasks, reward, agent=agent, n_workers=4,
                fusion_tournament=True)   # ranks singles, so win_rate exists
```

`result.fusion_stats()` is the evidence, with the denominators the old `fused`
counter was missing:

```python
stats = result.fusion_stats()
stats.win_rate        # fused wins / tournaments where a fusion competed at all
stats.mean_gain       # mean (fused − best single)
stats.negative        # how often the fusion lost, and by how much
stats.below_baseline  # how often it was worse than the artifact it started from
```

Read `contested` before `win_rate`. It counts tournaments where a fusion existed
*and was ranked*, and three things keep it at zero: one survivor, survivors that
contradict, and — the one that used to be counted as a fusion — survivors that
**agree**. `fuse_diffs` is `ops.update()`, so N copies of one diff "fuse" into
that diff; `nothing_to_fuse` names it, because the fix is the opposite of the fix
for contradiction (workers duplicating each other, not a key space that is too
coarse). Without `fusion_tournament=True`, `contested` is zero by construction
and `unranked` counts the unions that were committed instead. `win_rate` is
`None` rather than `0%` throughout, so none of that can be misread as "fusion
always lost".

Read `negative` before believing a high win rate: an empty losing tail usually
means the held-out set is too small to separate the candidates, and `ties` is the
tell.

Three outcomes, all worth having on **your** workload. Well above 50% means
merging recovers the N−1 proposals best-of-N discards. Near 50% means fusion is
noise there, and ranking it is not worth the sweep. Below 50% *with the
tournament catching it* means ranking is doing real work — a reason to leave it
on for that artifact.

**No number is published here, and that is deliberate.** The win rate is a
property of the artifact's key space and of how much the workers' proposals
overlap, not of the mechanism — so a figure measured on one dataset would be read
as a fact about merging and would not transfer to the next one. It is a
diagnostic to run on the workload you care about, which is why it is one keyword
argument and not a benchmark. The synthetic router domain is the clearest case of
why: its diffs are additive by construction, so fusion there wins by definition.

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
