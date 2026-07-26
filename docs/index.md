# AgentDescent

**Gradient descent — but the parameters are agents.** A parallel, asynchronous
framework for self-evolving agents (skills, prompts, harnesses) where **diffs are
the gradients** and **the aggregator is the optimizer**.

AgentDescent puts the *deep-learning training stack* on top of agents — data /
tensor / pipeline parallelism, parameter servers, decoupled/asynchronous RL,
partial rollout — applied to recursive self-improvement, where the "parameters"
are a **library of evolvable artifacts** (skills, prompts, harness modules,
verifiers) and the "gradients" are **diffs carrying evidence cards**.

!!! quote "The core observation"
    Serial RSI is bounded at **1 diff / T_iter**. AgentDescent runs *N* workers in
    parallel and merges their diffs into a shared, versioned artifact library,
    targeting **O(N / T_iter)** improvement throughput.

The one place the analogy *must* break defines the whole system:

!!! warning "Gradients add, diffs do not"
    Aggregation is therefore **not averaging** but **conflict resolution +
    statistical acceptance + transactional commit**. That merge is what
    [`aggregator.py`](architecture.md#3-component-responsibilities) implements.

---

## Where to go next

<div class="grid cards" markdown>

-   :material-star-four-points: **[The `evolve` method](evolution.md)** — start here

    The one entry point. Each capability is a plug-in to one `evolve()`
    parameter — this is the map, with an example per module.

-   :material-robot: **[Complete example](skill-evolution.md)**

    One end-to-end run — real dataset, real LLM, every module — evolving a skill
    that lifts held-out accuracy.

-   :material-sitemap: **[Architecture](architecture.md)**

    How the components fit together and how a diff travels from a worker to a
    committed change.

-   :material-lightbulb-on: **[Concepts](concepts.md)**

    The *why*: the training↔RSI analogy, staleness, the aggregator as a
    discrete-space optimizer, the three long tails, governance.

</div>

**Building blocks** (each plugs into `evolve`):

<div class="grid cards" markdown>

-   :material-connection: **[Connecting agents & LLMs](agents.md)** → `agent=`

    Any `prompt -> text` is a completion: Claude, GLM/OpenAI-compatible, a
    callable, a stub.

-   :material-database-arrow-down: **[Loading datasets](dataloader.md)** → the data layer

    `agentdescent.dataloader` — pull any benchmark (HF datasets-server + raw files),
    cached, dependency-free. Feeds `tasks` to `evolve`.

-   :material-cog-sync: **[The aggregator](aggregator.md)** → `agg_config=` / `aggregator_factory=`

    The optimizer — tune the reference merge/acceptance pipeline, or swap in your
    own.

-   :material-vector-triangle: **[Customizable parallelism](parallelism.md)** → `parallel=`

    Pluggable DP / TP / PP methods — or write your own `ParallelStrategy`.

-   :material-timer-sand: **[Duration-aware scheduling](duration-scheduling.md)**

    Estimate rollout cost from task size, then dispatch (LPT) and checkpoint
    stragglers (async runtime).

-   :material-speedometer: **[Efficiency experiments](efficiency.md)**

    Measured parallel scaling (near-linear to 8 workers) and async tail-hiding
    (2.5× over a sync barrier).

</div>

---

## The central analogy

| Model training | AgentDescent (parallel RSI) |
|---|---|
| parameter tensor θ | library of `Evolvable` artifacts |
| gradient *g* | `Diff` + `EvidenceCard` |
| parameter server | git-backed, version-vectored `Ledger` |
| optimizer step | `Aggregator` merge decision |
| per-param adaptive LR (Adam) | per-artifact Beta-posterior test |
| staleness / decoupled PPO | per-diff η + rebase re-verify |
| partial rollout | turn-level checkpoint / `ResumeQueue` |
| EMA (weight averaging) | `stable`/`dev` dual branch |
| training code (not self-modifiable) | L0 frozen layer |

---

## 30-second tour

```bash
pip install -e ".[dev]"

python -m examples.run_demo      # RQ1: merge vs fork (synchronous DP)
python -m examples.run_async     # FlashEvolve-style async + staleness policies
python -m examples.rq2_staleness # RQ2: staleness tolerance sweep
pytest                           #  tests, no external services
```

No LLM or external service is required: the reference domain is a fully
deterministic keyword-router skill, so the entire parallel loop runs in-process
and is unit-tested — while still producing genuine diffs that measurably improve
a held-out metric.

!!! note "Scope"
    This is a **research reference implementation**, faithful to the design's
    *mechanisms* and runnable end-to-end on a synthetic domain — not a
    production system. See the design spec §2 for the honest, narrowed novelty
    claim relative to FlashEvolve / SkillClaw / CoEvoSkills.
