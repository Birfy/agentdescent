# Concordia

**A parallel, self-evolving framework for accelerating recursive self-improvement (RSI).**

Concordia ports the *parallel-training playbook* — data/tensor/pipeline
parallelism, parameter servers, decoupled/asynchronous RL, partial rollout —
onto recursive self-improvement, where the "parameters" are a **library of
evolvable artifacts** (skills, prompts, harness modules, verifiers) and the
"gradients" are **diffs carrying evidence cards**.

!!! quote "The core observation"
    Serial RSI is bounded at **1 diff / T_iter**. Concordia runs *N* workers in
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

-   :material-sitemap: **[Architecture](architecture.md)**

    How the components fit together and how a diff travels from a worker to a
    committed change.

-   :material-lightbulb-on: **[Concepts](concepts.md)**

    The *why*: the training↔RSI analogy, staleness, the aggregator as a
    discrete-space optimizer, the three long tails, governance.

-   :material-rocket-launch: **[Usage & extending](usage.md)**

    How to install, run the sync/async demos, tune the knobs, and plug in your
    own `Evolvable` domain.

-   :material-dna: **[Evolving anything](evolution.md)**

    The general engine — evolve *any* artifact (skill, harness, verifier) by
    writing its `Strategy` + `run`/`reward`/`propose`.

-   :material-robot: **[Example: skill evolution](skill-evolution.md)**

    One complete run — real dataset, real LLM, every module — evolving a skill
    that lifts held-out accuracy.

-   :material-connection: **[Connecting agents & LLMs](agents.md)**

    The general provider-agnostic layer — any `prompt -> text` is a completion.

-   :material-speedometer: **[Efficiency experiments](efficiency.md)**

    Measured parallel scaling (near-linear to 8 workers) and async tail-hiding
    (2.5× over a sync barrier).

-   :material-vector-triangle: **[Customizable parallelism](parallelism.md)**

    Pluggable DP / TP / PP methods — or write your own `ParallelStrategy`.

-   :material-timer-sand: **[Duration-aware scheduling](duration-scheduling.md)**

    Estimate rollout cost from task size, then dispatch (LPT) and checkpoint
    stragglers on that estimate.

</div>

---

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
| EMA (weight averaging) | `stable`/`dev` dual branch |
| training code (not self-modifiable) | L0 frozen layer |

---

## 30-second tour

```bash
pip install -e ".[dev]"

python -m examples.run_demo      # RQ1: merge vs fork (synchronous DP)
python -m examples.run_async     # FlashEvolve-style async + staleness policies
python -m examples.rq2_staleness # RQ2: staleness tolerance sweep
pytest                           # 36 tests, no external services
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
