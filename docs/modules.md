# Module map

Every module, what it is for, and where its design is explained. The
[API reference](api.md) is the generated companion to this page: this is *why*
and *how*, that is *what*.

```
                          evolve()  ── the one entry point
                              │
   ┌───────────────┬──────────┴──────────┬────────────────────┐
   │               │                     │                    │
 what evolves   who does the work    where it runs        how it merges
   │               │                     │                    │
 strategies      agents               executor             aggregator
 filetree        backends             supervisor           defaults
 treestrategy    runners              workspec             ledger
                 sampling             sandbox              verifier
                 dataloader           sandbox_container    evaluator
                 rewards              sandbox_shared       evalcache
                 parallel             pipeline             staleness
                 scheduler                                 governance

              policies ── the contracts, across all four
              metrics · bench ── what a run cost, and comparing runs
```

## The loop

| module | what it is | page |
|---|---|---|
| `evolution` | `evolve()`, the artifact, the actor, the result | [The `evolve` method](evolution.md) |
| `evolvable` | `Evolvable`, `Diff`, `EvidenceCard`, `Contract` — the data model | [Data model](data-model.md) |
| `skill` | `evolve_skill()` — dataset in, instruction out | [Quickstart](quickstart-skill.md) |
| `skilldir` | `evolve_skill_dir()` / `_agent_dir()` / `_agent_code()` | [Directory evolution](directory-evolution.md) |
| `async_evolve` | the same loop without the round barrier | [Async](async.md) |
| `orchestrator`, `async_runtime`, `domains.router` | the reference domain the results were measured with — adapters over the engine above, not a second loop | [Orchestrator](orchestrator.md) |

## What evolves

| module | what it is | page |
|---|---|---|
| `strategies` | `SingleSlot`, `AppendRules`, `KeyedRules` — the text strategies (re-exported from `evolution`, which is the published import path) | [Strategies](strategies.md) |
| `filetree` | a directory ↔ artifact state, path safety, `TreeSpec` | [Directory evolution](directory-evolution.md) |
| `treestrategy` | `FileTree`, the `<EDITS>` proposal protocol, `tree_reflector` | [Directory evolution](directory-evolution.md) |

## Who does the work

| module | what it is | page |
|---|---|---|
| `agents` | any `prompt -> text` is a completion; `WorkspaceAgent` adds a directory | [Agents](agents.md) |
| `backends` | a tool-using agent over a document too big to inline | [Backends](backends.md) |
| `runners` | give a real agent the candidate directory, one workspace per rollout | [Directory evolution](directory-evolution.md) |
| `dataloader` | datasets, splits, cached fetches | [Data layer](dataloader.md) |
| `rewards` | the three scorers everyone writes, with the details right | [Rewards](rewards.md) |

## How the work is spread

| module | what it is | page |
|---|---|---|
| `parallel` | DP / TP / PP — how a round's work is split | [Parallelism](parallelism.md) |
| `sampling` | which task a worker rolls out next | [Sampling](sampling.md) |
| `scheduler` | duration-aware dispatch, stragglers, the audit queue | [Scheduling](duration-scheduling.md) |
| `pipeline` | the retirement, early-stop and backpressure rules both runtimes share | [Async](async.md) |

## Where it runs

| module | what it is | page |
|---|---|---|
| `executor` | the `rollout(spec) -> Result` seam, and the in-process default | [Execution](execution.md) |
| `supervisor` | persistent worker processes, and deciding when one is gone | [Execution](execution.md#why-not-processpoolexecutor) |
| `workspec` | a rollout as data: named callables instead of closures | [Execution](execution.md#work-has-to-be-describable-as-data-first) |
| `sandbox` | workspace leases: one ceiling, one release path, reclaim what an owner abandoned | [Sandboxes](sandboxes.md#lifetime-leases-not-deletion-by-age) |
| `sandbox_shared` | one ceiling across processes, counted from the lease directory | [Sandboxes](sandboxes.md#one-ceiling-across-processes) |
| `sandbox_container` | the provider that makes a sandbox an actual boundary (needs docker/podman) | [Sandboxes](sandboxes.md#isolation-strength-three-levels) |

## How a change is accepted

| module | what it is | page |
|---|---|---|
| `aggregator` | the optimizer: staleness → conflict → fusion → acceptance → commit | [Aggregator](aggregator.md) |
| `defaults` | the shipped algorithm as replaceable pieces: conflict, fusion, acceptance, promotion | [Aggregator](aggregator.md) |
| `stats` | the acceptance maths: Beta posterior, `P(Δ>0)`, annealed δ, UCB, difficulty weight | [Aggregator](aggregator.md) |
| `verifier` | rule / learned / oracle, and the budget on the expensive one | [Verifier](verifier.md) |
| `evaluator` | the gate's own bounded, reusable concurrency, separate from the rollouts' | [Verifier](verifier.md#the-evaluation-group) |
| `evalcache` | memoised evaluations: single-flight, environment-aware, shareable across processes | [Verifier](verifier.md#the-evaluation-cache) |
| `staleness` | what to do with a diff whose base version moved | [Staleness](staleness.md) |
| `ledger` | the git-backed, compare-and-swap artifact store | [Ledger](ledger.md) |
| `governance` | L0 frozen / L1 slow / L2 fast, by blast radius | [Governance](governance.md) |

## Across all of it

| module | what it is | page |
|---|---|---|
| `policies` | the contracts: which decisions are replaceable, and what each is given | [Architecture](architecture.md#35-what-the-infrastructure-owns-and-what-the-algorithm-owns) |
| `metrics` | what the run cost: time, calls, staleness ratio, cache hits, sandbox waits | [Usage](usage.md#what-a-run-cost) |
| `bench` | the configuration matrix and the rules that make comparing them mean something | [Efficiency](efficiency.md#the-configuration-matrix-bench) |

## Reading order

Depending on what you are doing:

**Just using it.** [Install](install.md) → [Quickstart](quickstart-skill.md) →
[The `evolve` method](evolution.md) → the one module you need to swap.

**Evolving a folder or an agent's code.**
[Quickstart — a directory](quickstart-directory.md) →
[Directory evolution](directory-evolution.md) → [Governance](governance.md) for
the safety model.

**Deciding whether to trust it.** [Concepts](concepts.md) →
[Aggregator](aggregator.md) → [Orchestrator](orchestrator.md) (how the claims
were measured) → [Results](results.md).

**Extending it.** [Data model](data-model.md) →
[Strategies](strategies.md#writing-your-own) →
[Aggregator](aggregator.md#replacing-aggregator_factory-aggregatorprotocol) → the
[algorithm ports](self-evolution-examples.md), each of which replaces a different
piece.

## Provided, tested, and not in any engine path

Some of what `import agentdescent` gives you is a **primitive for a
configuration that does not ship yet** — the design calls for it, it is
implemented and tested in isolation, and no loop reaches it today. Each one says
so in its own docstring, which meant finding out cost a read of the source, one
class at a time. They are all in one table instead:

| name | why it exists | what would reach it |
|---|---|---|
| [`Ledger.commit_atomic`](ledger.md) | 2PC across several artifacts, for a contract-breaking diff that must land with its adapters | a multi-artifact library; `evolve()` registers exactly one |
| [`L1SerialGate`](governance.md) | "at most one L1 diff in evaluation anywhere" | concurrent merging; every shipped runtime merges on one thread, so the guarantee already holds by construction |
| [`ResumeQueue`](duration-scheduling.md) | turn-level checkpoints of a timed-out rollout | a rollout that exposes its turns; `run(rendered, task) -> output` is opaque, which is what lets any agent be plugged in |
| [`AuditScheduler.pop`](duration-scheduling.md) | draining the Ĝ-ordered audit queue out of band | `AuditScheduler(collect=True)`; the default computes priorities without queuing, because nothing drains it |
| [`EvidenceBuffer.settled`](aggregator.md) | discarded evidence stays addressable — the structural advantage of artifacts over gradients | re-filing settled cards into the trajectory pool; today it is a bounded diagnostic ring |
| `TaskScheduler` × artifact axis | the design's L-task is `(task cluster × artifact)` | more than one artifact; `TaskCluster` has no artifact dimension. The *cluster* axis is reachable from `evolve()` via [`ClusterParallel`](parallelism.md) |
| [`PipelineParallel`](parallelism.md) | one artifact per stage, with upstream blame | a multi-artifact run; `evolve()` **refuses** it rather than degrading to DP in silence |

The rule they share: a primitive that is implemented and unreachable is honest;
one that is *reachable and silently does nothing* is not, which is why
`PipelineParallel` raises and `Policies` refuses a field it cannot honour.

## Dependency shape

Nothing in the framework imports a provider SDK at module level, and the core
imports nothing outside the standard library:

```
evolvable ── ledger ── aggregator ── evolution ── skill / skilldir
    │           │          │             │
governance   verifier  staleness    parallel · sampling · scheduler
                                          │
                              agents ── backends ── runners
                                          │
                              filetree ── treestrategy
```

`anthropic` and `openhands-ai` are imported lazily inside the functions that need
them, so the rest of the framework runs without either.
