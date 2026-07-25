# Concordia

**A parallel, self-evolving framework for accelerating recursive self-improvement (RSI).**

Concordia is a research reference implementation. It ports the
parallel-training playbook — data/tensor/pipeline parallelism, parameter
servers, decoupled/asynchronous RL, partial rollout — onto RSI, where the
"parameters" are a **library of evolvable artifacts** (skills, prompts, harness
modules, verifiers) and the "gradients" are **diffs carrying evidence cards**.

The core observation (design §1): serial RSI is bounded at **1 diff / T_iter**.
Concordia runs *N* workers in parallel and merges their diffs into a shared,
versioned artifact library, targeting **O(N / T_iter)** improvement throughput.

> The one place the analogy *must* break defines the whole system: **gradients
> add, diffs do not.** Aggregation is therefore not averaging but
> **conflict resolution + statistical acceptance + transactional commit**.

## 📖 Documentation

Full docs live in [`docs/`](docs/) and render as a website via MkDocs Material:

| Page | What's in it |
|---|---|
| [Home](docs/index.md) | Overview and 30-second tour |
| [Architecture](docs/architecture.md) | Components, data-flow diagram, the two runtimes, concurrency model |
| [Concepts](docs/concepts.md) | The training↔RSI analogy, staleness, the aggregator, the three long tails, governance |
| [Usage & extending](docs/usage.md) | Running the demos, config reference, **plugging in your own `Evolvable` domain** |
| [Evolving anything](docs/evolution.md) | The general engine — evolve any artifact by writing its `Strategy` + `run`/`reward`/`propose` |
| [Connecting agents & LLMs](docs/agents.md) | The provider-agnostic completion layer |
| [Customizable parallelism](docs/parallelism.md) | Pluggable DP / TP / PP strategies — or write your own |
| [Duration-aware scheduling](docs/duration-scheduling.md) | Estimate rollout cost from task size; LPT dispatch + straggler checkpointing |
| [Efficiency experiments](docs/efficiency.md) | Measured parallel scaling and async tail-hiding |
| [Example: skill evolution](docs/skill-evolution.md) | One complete run — real dataset, real LLM, every module |

```bash
pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static HTML into ./site
```

A GitHub Actions workflow ([`.github/workflows/docs.yml`](.github/workflows/docs.yml))
builds and deploys the site to GitHub Pages — enable it under *Settings → Pages
→ Source: GitHub Actions*.

## Evolve anything — the general engine

The core is the ledger + aggregator + schedulers + governance.
[`concordia.evolution`](concordia/evolution.py) is the domain-agnostic engine on
top: describe **what evolves** (a `Strategy`) and the **rules of evolution**
(`run` / `reward` / `propose`), and it runs the parallel, merge-based loop.

```python
from concordia.evolution import evolve, AppendRules

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

**Connect any agent/LLM** — [`concordia.agents`](concordia/agents.py) is the
separate provider layer; any `prompt -> text` is a completion (`claude(...)`,
`openai_compatible(...)` for GLM/OpenAI-style endpoints, `from_callable(...)`,
`with_retries(...)`).

The one complete end-to-end run — real dataset, real LLM, every module — is
[`examples/skill_evolution.py`](examples/skill_evolution.py)
(`python -m examples.skill_evolution --dry-run` for the no-API preview).
Guides: [the engine](docs/evolution.md) · [skill example](docs/skill-evolution.md)
· [agents](docs/agents.md).

## Efficiency (measured)

[`examples/efficiency.py`](examples/efficiency.py) — **parallel scaling** is
near-linear (efficiency ≥ 0.99 through 8 workers, 7.9× speedup), and the
**async pipeline** is **2.5× faster than a sync barrier** under heavy-tailed
rollout latency (100% vs 40% worker utilization). See
[docs/efficiency.md](docs/efficiency.md).

## The central analogy

| Model training | Concordia (parallel RSI) |
|---|---|
| parameter tensor θ | library of `Evolvable` artifacts |
| gradient *g* | `Diff` + `EvidenceCard` |
| parameter server | git-backed, version-vectored `Ledger` |
| optimizer step | `Aggregator` merge decision |
| per-param adaptive LR (Adam) | per-artifact Beta-posterior test |
| staleness / decoupled PPO | per-diff η + rebase re-verify |
| partial rollout | turn-level checkpoint / `ResumeQueue` |
| EMA (weight averaging) | stable/dev dual branch |
| training code (not self-modifiable) | L0 frozen layer |

## Install & run

```bash
pip install -e ".[dev]"

# RQ1 — merge vs fork, end to end (synchronous DP)
python -m examples.run_demo

# Async stage orchestration — Full/Guarded/Reflective policies + async_ratio sweep
python -m examples.run_async

# The flagship: evolve a skill on a real dataset with a real LLM (--dry-run: no API)
python -m examples.skill_evolution --dry-run

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
([`concordia/domains/router.py`](concordia/domains/router.py)) is a fully
deterministic keyword-router skill, so the entire parallel loop runs in-process
and is unit-tested — while still producing genuine diffs that measurably improve
a held-out metric.

## Architecture → code map

Every module cites the design section it implements.

| Component | Module | Design § |
|---|---|---|
| `Evolvable` unit, `Diff`, `EvidenceCard`, version vectors | [`evolvable.py`](concordia/evolvable.py) | 3.2, 3.3 |
| Git-backed Ledger: version vectors, CAS, 2PC, dual branch | [`ledger.py`](concordia/ledger.py) | 3.1, 4.5 |
| Aggregator: staleness → conflict → fusion → Beta accept → commit | [`aggregator.py`](concordia/aggregator.py) | 4 |
| **Staleness policies: Full / Guarded / Reflective** | [`staleness.py`](concordia/staleness.py) | 4.2 |
| **Async stage-orchestration runtime + `async_ratio`** | [`async_runtime.py`](concordia/async_runtime.py) | 3.1 |
| **Parallel paradigms: DP / TP / PP** | [`parallel.py`](concordia/parallel.py) | 8 |
| Statistics: Beta posterior, `P(Δ>0)`, annealed δ, UCB | [`stats.py`](concordia/stats.py) | 4.4, 5.2 |
| Three schedulers: UCB task / audit / resume queue | [`scheduler.py`](concordia/scheduler.py) | 5 |
| Three-layer verifier (rule / learned / oracle) | [`verifier.py`](concordia/verifier.py) | 3.1, 5.3 |
| Layered governance by blast radius (L0/L1/L2) | [`governance.py`](concordia/governance.py) | 6 |
| Worker: rollout + propose | [`worker.py`](concordia/worker.py) | 3.1 |
| Orchestrator (sync DP) + fork baseline | [`orchestrator.py`](concordia/orchestrator.py) | 3.1, RQ1 |
| **Agent/LLM connection layer (provider-agnostic)** | [`agents.py`](concordia/agents.py) | — |
| **General evolution engine + pluggable `Strategy`** | [`evolution.py`](concordia/evolution.py) | 3.2 |

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
4. **Statistical acceptance (§4.4)** — commit only if
   `P(Δ > 0) > 1 − δ` under a per-artifact Beta posterior, not a point threshold.
   `δ` anneals with version (LR decay); a trust-region caps diff size.
5. **Commit (§4.1)** — compare-and-swap on `dev` (2PC for contract-breaking
   multi-artifact diffs).
6. **Dual-branch promotion (§4.5)** — `dev → stable` after *K* regression-free
   rounds (EMA-style confirmation).
7. **Audit (§5.3)** — the merge decision is itself submitted to the
   `AuditScheduler`; high-blast-radius / low-trust merges are forced through the
   oracle. The optimizer audits itself.

## Parallelism & asynchrony

Concordia ships two execution runtimes and a set of pluggable strategies, so a
run can be moved along the sync↔async and DP↔TP↔PP axes without touching the
merge pipeline.

### Two runtimes

- **Synchronous DP** ([`orchestrator.py`](concordia/orchestrator.py)) — a round
  barrier: all workers step, then one `aggregator.step()`, then the next round.
  Deterministic; the RQ1/RQ2 baseline.
- **Asynchronous stage orchestration** ([`async_runtime.py`](concordia/async_runtime.py),
  FlashEvolve-style) — **no barrier**. Worker threads keep producing evidence
  while a dedicated aggregator thread keeps merging, connected by the
  thread-safe `EvidenceBuffer`. The rollout/propose and aggregate/commit stages
  overlap instead of stalling.

### Staleness policies ([`staleness.py`](concordia/staleness.py), FlashEvolve Full/Guarded/Reflective)

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

### DP / TP / PP ([`parallel.py`](concordia/parallel.py), §8)

- **DP (data parallel)** — same snapshot, task-sharded, diffs merged. The default
  the async runtime runs.
- **TP (tensor parallel)** — split one *hot* artifact into disjoint sections;
  each worker owns a section, so edits are conflict-free **by construction** and
  the merge is concatenation + a consistency reviewer (`TensorParallelMerge`).
- **PP (pipeline parallel)** — artifacts form a dependency chain; a downstream
  failure back-propagates blame to the earliest failing upstream stage
  (`PipelineChain.blame`, shared with the §7 counterfactual-replay attribution).

## The three long tails (§5)

Concordia treats "the long tail" as three separate problems:

- **L-traj** (system): heavy-tailed rollout durations → turn-level checkpoint +
  `ResumeQueue`, resumed against the latest ledger (a free cross-version A/B
  signal).
- **L-task** (data): Zipfian artifact triggering → **UCB over
  (cluster × artifact)** so starved tail artifacts get an exploration bonus,
  plus a difficulty filter and a tail canary set.
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
so the mechanisms are observable and testable. Concordia's novelty is a
**narrow, defensible engineering synthesis** — concurrent, staleness-bounded,
conflict-resolved **diff-level merge** over a git-backed versioned ledger — and
its throughput premise is a *testable engineering hypothesis*, not community
consensus (cf. FlashEvolve / SkillClaw / CoEvoSkills).
