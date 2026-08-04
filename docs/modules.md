# Module map

Every module, what it is for, and where its design is explained. The
[API reference](api.md) is the generated companion to this page: this is *why*
and *how*, that is *what*.

```
                     evolve()  ── the one entry point
                        │
   ┌────────────────────┼────────────────────────────────────┐
   │                    │                                    │
 what evolves      who does the work                  how it merges
   │                    │                                    │
 strategy           agents / backends                   aggregator
 filetree           runners                             ledger
 treestrategy       sampling                            verifier
                    parallel                            staleness
                    scheduler                           governance
                    dataloader / rewards                  metrics · policies
                    sandbox                              defaults
                    sandbox
```

## The loop

| module | what it is | page |
|---|---|---|
| `evolution` | `evolve()`, the artifact, the actor, the result | [The `evolve` method](evolution.md) |
| `evolvable` | `Evolvable`, `Diff`, `EvidenceCard`, `Contract` — the data model | [Data model](data-model.md) |
| `skill` | `evolve_skill()` — dataset in, instruction out | [Quickstart](quickstart-skill.md) |
| `skilldir` | `evolve_skill_dir()` / `_agent_dir()` / `_agent_code()` | [Directory evolution](directory-evolution.md) |
| `async_evolve`, `async_runtime` | the same loop without the round barrier | [Async](async.md) |
| `orchestrator`, `worker`, `domains.router` | the reference loop the results were measured with | [Orchestrator](orchestrator.md) |

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

## How a change is accepted

| module | what it is | page |
|---|---|---|
| `aggregator` | the optimizer: staleness → conflict → fusion → acceptance → commit | [Aggregator](aggregator.md) |
| `stats` | the acceptance maths: Beta posterior, `P(Δ>0)`, annealed δ, UCB, difficulty weight | [Aggregator](aggregator.md) |
| `verifier` | rule / learned / oracle, and the budget on the expensive one | [Verifier](verifier.md) |
| `staleness` | what to do with a diff whose base version moved | [Staleness](staleness.md) |
| `pipeline` | the retirement and backpressure policies the two barrier-free runtimes share | [Async](async.md) |
| `ledger` | the git-backed, compare-and-swap artifact store | [Ledger](ledger.md) |
| `governance` | L0 frozen / L1 slow / L2 fast, by blast radius | [Governance](governance.md) |
| `evaluator` | the gate's own bounded, reusable concurrency, separate from the rollouts' | [Verifier](verifier.md#the-evaluation-group) |
| `evalcache` | memoised evaluations: single-flight, environment-aware, shareable across processes | [Verifier](verifier.md) |
| `workspec` | a rollout as data: named callables instead of closures | [Parallelism](parallelism.md#where-rollouts-run-the-execution-plane) |
| `executor` · `supervisor` | where rollouts run: threads, or supervised worker processes | [Parallelism](parallelism.md#where-rollouts-run-the-execution-plane) |
| `sandbox` | workspace leases: one ceiling, one release path, reclaim what an owner abandoned | [Directory evolution](directory-evolution.md#how-workspaces-are-managed) |
| `defaults` | the shipped algorithm as replaceable pieces: conflict, fusion, acceptance, promotion | [Aggregator](aggregator.md) |
| `sandbox_container` | the provider that makes a sandbox an actual boundary (needs docker/podman) | [Directory evolution](directory-evolution.md#isolation-strength-three-levels) |
| `policies` | the contracts: which decisions are replaceable, and what each is given | [Architecture](architecture.md#35-what-the-infrastructure-owns-and-what-the-algorithm-owns) |
| `metrics` | what the run cost: time, calls, staleness ratio, cache hits, sandbox waits | [Usage](usage.md#what-a-run-cost) |

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
