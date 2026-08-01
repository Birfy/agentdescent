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

**Three runs, so the spread is visible.** The model is not deterministic, so a
single run is a sample, not a result:

| Run | Sampler | Baseline | Final val | Held-out test | Bullets |
|---|---|---|---|---|---|
| A | round-robin | 87.0% | **95.7%** | **90.5%** | 2 |
| B | difficulty | — | 91.3% | 85.7% | 2 |
| C | round-robin | 90.9% | 90.9% | 85.7% | 2 |

Two runs of the *identical* configuration (A and C, same `--seed 1`) landed 4.8
val points apart and reported different baselines, because `deepseek-v4-flash`
does not return identical text for identical prompts. On ~23 val / ~21 test items
one item is worth 4–5 points, so **differences of this size are noise**. What
reproduces across all three is the shape, not the number: the Curator admits
about two bullets and rejects the rest, and the playbook never regresses below
baseline — which is the acceptance gate doing its job.

!!! warning "Do not read +8.7 as the effect size"
    An earlier version of this page reported run A alone as "87.0% → 95.7%
    (+8.7)". With n≈23 held-out items and a non-deterministic model, that is
    within run-to-run variance. Reporting a single LLM run as a point estimate is
    the easiest way to fool yourself; if you need an effect size, run it several
    times and report the spread.

**What it cost** (run C, instrumented with [`Usage`](agents.md#what-did-the-run-cost-usage)):
125 model calls, 48,724 prompt + 27,313 completion tokens, 295 s inside the model
for 8 rounds over 57 tasks — well under a cent at `deepseek-v4-flash` prices.

!!! warning "Pick a configuration that isn't saturated"
    With the default `--top-k 10` (the ten *most frequent* tags) a strong model
    scores **100% at baseline**, so there are no failures to reflect on and ACE
    correctly curates nothing. Self-evolution can only work where the base agent
    actually fails: widen `--top-k` (rarer, more confusable concepts) until the
    baseline leaves headroom. The same effect is visible in
    [EvoSkill](algo-evoskill.md#empirical-results-real-openhands-agent-deepseek-on-officeqa),
    and it is why [`DifficultyWeighted` task sampling](sampling.md)
    exists — it steers rollouts toward the tasks that still fail.

### `--top-k` sets the difficulty

The number of XBRL concepts *is* the difficulty: it is a k-way choice. Measured
with `deepseek-v4-flash`, 8 rounds, 4 workers:

| `--top-k` | tasks | val, before → after | test | bullets |
|---|---|---|---|---|
| 10 | 37 | 1.000 → 1.000 | 1.000 | 0 |
| 40 | 57 | 0.850 → 0.850 | 0.921 | 0 |
| **120** (the default) | 121 | **0.844 → 0.889** | **0.884** | **2** |

At 10 the task is already solved. At 40 there is headroom, but no bullet beats the
baseline and the Curator's gate rejects every proposal. At 120 two bullets survive
the gate and validation accuracy rises 4.5 points.

## Run it

```bash
python -m examples.ace_context_evolution --dry-run
python -m examples.ace_context_evolution --model claude-haiku-4-5
python -m examples.ace_context_evolution --top-k 40 --pool 400 --rounds 8   # the run above
```

Offline tests: `tests/test_ace_example.py`.
