# ACE — Agentic Context Engineering

> **Skill / context self-evolution.** Evolve a *playbook of lessons* (the model's
> context), not the weights. Runs through [`evolve()`](evolution.md) with a custom
> `Strategy`. Example: [`examples/ace_context_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/ace_context_evolution.py).

| | |
|---|---|
| **Paper** | *Agentic Context Engineering* — Zhang et al., 2025 ([arXiv:2510.04618](https://arxiv.org/abs/2510.04618)) |
| **Repo** | [`ace-agent/ace`](https://github.com/ace-agent/ace) |
| **Dataset** | **FiNER-139** (financial XBRL tagging) — `nlpaueb/finer-139` |
| **Layer** | L2 skill (`blast_radius=0.2`) |

## The algorithm

ACE evolves a **context playbook** through three roles, which map one-to-one onto
`evolve()`:

| ACE role | AgentDescent piece | Job |
|---|---|---|
| **Generator** | `LLMAgent.solve` (the `run`) | solve a task using the playbook |
| **Reflector** | `LLMAgent.propose` (ACE template) | distil ONE *delta bullet* from a failure |
| **Curator** | the **aggregator** | deterministic, non-LLM merge (dedup + statistical acceptance) |

Two invariants are preserved by the custom `ACEPlaybook` strategy:

* **Incremental delta updates.** `to_diff` only ever *appends* a new
  content-addressed bullet — never a monolithic rewrite — so ACE's **context
  collapse** (the model compressing an accumulated context into a lossy summary)
  cannot happen.
* **Grow-and-refine de-dup.** A near-duplicate bullet (lexical-Jaccard proxy for
  ACE's embedding de-dup) is dropped at insert time.

ACE's per-bullet **helpful / harmful** counters become the aggregator's per-diff
**Beta-posterior acceptance**: a bullet is committed only when it raises held-out
reward, and rejected otherwise.

## How it plugs into `evolve()`

```python
evolve(tasks, reward, agent=ace_agent(completion),
       strategy=ACEPlaybook(), blast_radius=0.2, artifact_id="ace_playbook")
```

* `strategy=ACEPlaybook()` — the itemised delta representation + grow-and-refine.
* `agent=` — Generator (`solve`) + Reflector (`propose`).
* the default aggregator **is** the Curator (dedup + Beta gate).

## Dataset

FiNER-139 framed as XBRL-tag classification of a highlighted numeric span,
restricted to the `--top-k` most frequent concepts so a learned lesson transfers.
ACE's full setup also runs AppWorld (a heavy simulator) — out of scope here and
documented in the module docstring.

## Plug-ins implemented

The example provides these plug-ins to `evolve()` (in
[`examples/ace_context_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/ace_context_evolution.py)):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| **`ACEPlaybook`** | `strategy=` | the itemised, incremental-delta playbook — `to_diff` appends one content-addressed bullet with grow-and-refine de-dup; never rewrites (so context collapse can't happen) |
| default `Aggregator` | (the Curator) | dedup + Beta-posterior acceptance — a bullet commits only if it raises held-out reward; **no custom aggregator needed** |
| `ace_agent()` | `agent=` | Generator (`solve`) + Reflector (`propose`) over a completion |

## Empirical results — FiNER-139 with DeepSeek

Run through `evolve()` (synchronous DP, 4 workers, 8 rounds) with
`deepseek-v4-flash` on the real FiNER-139 validation split, 57 single-entity
sentences over the 40 most frequent XBRL concepts, split train / val / test:

| Concepts | Tasks | Baseline (empty playbook) | After curation | Held-out test | Bullets curated |
|---|---|---|---|---|---|
| top-40 | 57 | **87.0%** | **95.7% (+8.7)** | **90.5%** | 2 |

A second run, identical except for `--sampler difficulty`
([task selection](evolution.md#task-selection-which-rollout-to-spend)), reached a
lesson sooner — admitting one in round 0 rather than round 2 — but finished
**0.913 val / 0.857 test**, below round-robin. On ~23 val items a single item is
worth ~4 points, so this is a small sample rather than a verdict; it is recorded
because it is the honest outcome, and because "more rollouts on failing tasks"
plainly did not translate into a better playbook here.

The Curator admitted **2 of the bullets** the Reflector proposed; the rest were
rejected by the Beta-posterior acceptance test, which is the point — a lesson
only lands if it demonstrably raises held-out accuracy.

!!! warning "Pick a configuration that isn't saturated"
    With the default `--top-k 10` (the ten *most frequent* tags) a strong model
    scores **100% at baseline**, so there are no failures to reflect on and ACE
    correctly curates nothing. Self-evolution can only work where the base agent
    actually fails: widen `--top-k` (rarer, more confusable concepts) until the
    baseline leaves headroom. The same effect is visible in
    [EvoSkill](algo-evoskill.md#empirical-results-real-openhands-agent-deepseek-on-officeqa),
    and it is why [`DifficultyWeighted` task sampling](evolution.md#task-selection-which-rollout-to-spend)
    exists — it steers rollouts toward the tasks that still fail.

## Run it

```bash
python -m examples.ace_context_evolution --dry-run
python -m examples.ace_context_evolution --model claude-haiku-4-5
python -m examples.ace_context_evolution --top-k 40 --pool 400 --rounds 8   # the run above
```

Offline tests: `tests/test_ace_example.py`.
