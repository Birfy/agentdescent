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
    [`aggregator.py`](aggregator.md) implements.

---

## Start here

### Have a dataset

That is the whole input.

```python
from agentdescent import evolve_skill, openai_compatible
from agentdescent.dataloader import hf_rows

rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor", limit=40)

result = evolve_skill(rows, model=openai_compatible(model="deepseek-v4-flash"),
                      prompt="question", gold="answer", score="exact")

print(result.rendered)        # the skill it learned
```

Run as written, that lifted held-out exact match from **0.167 to 0.583** and wrote
*"Respond with only the requested answer, omitting any extra explanation or
restatement."* — see [Quickstart](quickstart-skill.md) for the full measurement.

### Have a directory

A skill folder, a folder of subagent definitions, or the agent's own code:

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

Each rollout materialises the candidate into a throwaway workspace and a **real
agent reads the files off disk**. See
[Quickstart — a directory](quickstart-directory.md).

Neither is a separate system: both build ordinary arguments and call
[`evolve()`](evolution.md), which is where you go the moment you want more.

```bash
pip install agentdescent
```

---

## Where to go next

<div class="grid cards" markdown>

-   :material-download: **[Install and first run](install.md)** — start here

    Install, then reproduce the central claim in seconds with no API key.

-   :material-rocket-launch: **[Quickstart — dataset to skill](quickstart-skill.md)**

    One call, three decisions: your data, how to score it, which model. With the
    measured result of running it.

-   :material-folder-cog: **[Quickstart — a directory](quickstart-directory.md)**

    A skill folder, an agent folder, or its code — evolved by an agent that reads
    the files.

-   :material-star-four-points: **[The `evolve` method](evolution.md)**

    The one entry point underneath. Every capability is a plug-in to one
    `evolve()` parameter — this is the map, with an example per module.

-   :material-lightbulb-on: **[Concepts](concepts.md)**

    The *why*: the training↔RSI analogy, staleness, the aggregator as a
    discrete-space optimizer, the three long tails, governance.

-   :material-sitemap: **[Architecture](architecture.md)**

    How the components fit together and how a diff travels from a worker to a
    committed change.

-   :material-view-grid-plus: **[Module map](modules.md)**

    Every module, what it is for, and a reading order for whatever you are doing.

-   :material-api: **[API reference](api.md)**

    Every public name with its real signature — generated from the code, and
    tested against it.

-   :material-chart-box: **[Measured results](results.md)**

    Every empirical claim with the setup that produced it — including the
    benchmarks where the honest answer is "nothing to learn here".

</div>

**Building blocks** (each plugs into `evolve`):
- **[The decision plane](policies.md)** — every replaceable decision (selection, sampling, merge, acceptance, promotion) in one `Policies` bundle

<div class="grid cards" markdown>

-   :material-connection: **[Agents & LLMs](agents.md)** → `agent=`

    Any `prompt -> text` is a completion: Claude, GLM/OpenAI-compatible, a CLI
    coding agent, a callable, a stub.

-   :material-file-tree: **[Strategies](strategies.md)** → `strategy=`

    What the artifact *is*: one slot, a playbook, keyed categories, or a
    directory. The key space is the design decision.

-   :material-cog-sync: **[The aggregator](aggregator.md)** → `aggregator_factory=`

    The optimizer — tune the reference merge/acceptance pipeline, or swap in your
    own.

-   :material-vector-triangle: **[Parallelism](parallelism.md)** → `parallel=`

    Pluggable DP / TP methods, plus [sampling](sampling.md) for which rollout to
    spend and [scheduling](duration-scheduling.md) for when.

-   :material-source-branch-sync: **[Async](async.md)** → `asynchronous=True`

    Barrier-free workers, a lag budget, and the
    [staleness policies](staleness.md) that keep it safe.

-   :material-shield-lock: **[Governance](governance.md)** → `blast_radius=`

    L2 skills merge freely, L1 harnesses are oracle-gated, L0 is frozen — and
    frozen *paths* for a directory.

</div>

---

## The central analogy

| Model training | AgentDescent (parallel RSI) |
|---|---|
| parameter tensor θ | library of [`Evolvable`](data-model.md) artifacts |
| gradient *g* | [`Diff` + `EvidenceCard`](data-model.md) |
| parameter server | git-backed, version-vectored [`Ledger`](ledger.md) |
| optimizer step | [`Aggregator`](aggregator.md) merge decision |
| per-param adaptive LR (Adam) | per-artifact Beta-posterior test |
| staleness / decoupled PPO | [per-diff η + rebase re-verify](staleness.md) |
| partial rollout | [straggler detection](duration-scheduling.md) (`ResumeQueue`; resume not implemented) |
| EMA (weight averaging) | [`stable`/`dev` dual branch](ledger.md#two-branches-dev-and-stable) |
| training code (not self-modifiable) | [L0 frozen layer](governance.md) |

---

## 30-second tour

```bash
pip install -e ".[dev]"

python -m examples.run_demo            # RQ1: merge vs fork (synchronous DP)
python -m examples.run_async           # FlashEvolve-style async + staleness policies
python -m examples.skill_dir_evolution # evolve a skill directory a real agent reads
python -m examples.rq2_staleness       # RQ2: staleness tolerance sweep
pytest                                 # the suite, no external services
```

Step by step: [install and first run](install.md). Everything runnable, with the
output each one produces: [run everything](usage.md).

No LLM or external service is required: the
[reference domain](orchestrator.md#why-a-synthetic-domain-exists-at-all) is a
fully deterministic keyword-router skill, so the entire parallel loop runs
in-process and is unit-tested — while still producing genuine diffs that
measurably improve a held-out metric.

!!! note "Scope"
    This is a **research reference implementation**, faithful to the design's
    *mechanisms* and runnable end-to-end on a synthetic domain — not a
    production system. See the design spec §2 for the honest, narrowed novelty
    claim relative to FlashEvolve / SkillClaw / CoEvoSkills.
