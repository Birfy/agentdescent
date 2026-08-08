# ACE — Agentic Context Engineering

> **Skill / context self-evolution.** Evolve a *playbook of lessons* (the model's
> context), not the weights. Runs through [`evolve()`](evolution.md) with a custom
> `Strategy`. Example: [`examples/ace/ace_context_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/ace/ace_context_evolution.py).

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
[`examples/ace/ace_context_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/ace/ace_context_evolution.py)):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| **`ACEPlaybook`** | `strategy=` | the itemised, incremental-delta playbook — `to_diff` appends one content-addressed bullet with grow-and-refine de-dup; never rewrites (so context collapse can't happen) |
| default `Aggregator` | (the Curator) | dedup + Beta-posterior acceptance — a bullet commits only if it raises held-out reward; **no custom aggregator needed** |
| `ace_agent()` | `agent=` | Generator (`solve`) + Reflector (`propose`) over a completion |

## Empirical results — FiNER-139 with DeepSeek

Barrier-free (`--async`), 4 workers, 120 rollouts pinned, `deepseek-v4-flash`,
`--pool 3200 --top-k 139` — 364 single-entity sentences over the full concept
set, split 176 train / 118 val (gate capped to 64) / 70 test. One seed; the
model is not deterministic, so read the shape rather than the third digit.

**Three configurations, changing one thing at a time.** The first is what this
port used to do; the last is what upstream does:

| acceptance | staleness | bullets | val | test | stale discarded | wall |
|---|---|---|---:|---:|---:|---:|
| Beta gate (was the default) | guarded | **0** | 0.781 → 0.781 | 0.757 | 0 / 12 | 621 s |
| `--grow-and-refine` | guarded | 6 | 0.703 → **0.750** | 0.786 | **9 / 15 (60%)** | 490 s |
| `--grow-and-refine` | `--staleness reflective` | **10** | 0.719 → **0.766** | 0.786 | **0 / 10** | 812 s |

**The first row is the finding.** Gating each bullet on held-out reward commits
*nothing*, and it is not a difficulty problem — the baseline leaves 20 points of
headroom and the Reflector produces sound, specific lessons throughout. A single
bullet teaches one XBRL concept, so it can only move a validation split that
happens to contain that concept: at `--val-cap 32` the five bullets an earlier
run admitted covered **4 of 32** val tasks, and widening the gate's sample to 64
dropped commits from 5 to 0. The gate gets *more* correct as it gets more
statistical power, and the playbook empties. ACE's claim is accumulation, and
per-bullet gating cannot accumulate.

`--grow-and-refine` restores upstream's Curator, which applies every validated
delta and tracks utility afterwards. Six bullets, val +4.7 points — and 21%
*cheaper*, because a gate that scores every candidate across 64 val tasks is the
expensive part.

**The third row is the async cost showing up where it should.** Once every
proposal commits, head moves on every sweep, and the default `guarded` band
(`--async-ratio 3`) discards 60% of the evidence — a whole rollout thrown away
per card. On a multi-key playbook that is waste rather than safety: a worker's
bullet rarely conflicts with the bullets committed while it was working.
`--staleness reflective` rebases instead of discarding, and the re-verification
is scored on the card's *own* trajectories (one or two tasks), not the held-out
set. Discards go to zero, commits double (2 → 4 sweeps), bullets 6 → 10.

The 66% wall-clock rise that comes with it is not the re-verification — it is
the extra work the recovered cards create: twice the commits, each paying a
64-task gate sweep, over a playbook that is now 10 bullets long in every prompt
(completion tokens 181k → 246k).

**Concurrency is unaffected by any of this** — 4.83× / 4.32× / 4.35× of
model-time-over-wall-clock — because the rebase re-verification runs on the
merger thread, not the workers.

!!! warning "Pick a configuration that isn't saturated"
    With `--top-k 10` (the ten *most frequent* tags) a strong model scores
    **100% at baseline**, so there are no failures to reflect on and ACE
    correctly curates nothing. Self-evolution can only work where the base agent
    actually fails.

    Raising `--top-k` alone does not fix it, because FiNER is stratified by
    concept and a small pool never surfaces the rare ones: at `--pool 1600` the
    120- and 139-concept selections are the **same 211 tasks** at a 0.875
    baseline. Measured there, 32 rollouts produced `+0/-0` on nearly every round
    and val never moved. `--pool 3200` puts the baseline at 0.667. The same
    saturation effect is visible in
    [EvoSkill](algo-evoskill.md#empirical-results-real-openhands-agent-deepseek-on-officeqa),
    and it is why [`DifficultyWeighted` task sampling](sampling.md) exists.

### What the playbook actually contains

Not paraphrase — three of the ten bullets from the last row above:

```
## debt
  - For a variable-rate debt instrument, report the rate in effect at the
    balance sheet date as the effective interest rate, not the stated percentage.
  - Do not map "reserve" to a loss-contingency accrual; in credit-facility
    context, "outstanding reserve" refers to outstanding letters of credit.
  - For prepayment/repayment of debt amounts, use RepaymentsOfDebt, not
    DebtInstrumentFaceAmount, which describes the instrument's principal
    balance, not a cash flow transaction.
```

These are the reason the Beta-gate row is a measurement artefact rather than a
verdict on the algorithm: the lessons were always there, and the gate could not
see them because each one is worth about one validation task.

One rough edge worth knowing: the Reflector invented both `business combination`
and `business combinations` as section names in one run. Grow-and-refine's
near-duplicate check is scoped *within* a section, so a singular/plural split
walks past it.

## Run it

```bash
python -m examples.ace.ace_context_evolution --dry-run
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5

# the last row of the table above
python -m examples.ace.ace_context_evolution --pool 3200 --top-k 139 \
    --val-cap 64 --budget-rollouts 120 --rounds 9999 \
    --grow-and-refine --staleness reflective \
    --async --async-ratio 3 --max-seconds 3600 --workers 4 \
    --model deepseek-v4-flash --yes
```

`--grow-and-refine` is off by default: it changes what the run measures, so it
has to be asked for rather than inherited.

Offline tests: `tests/test_ace_example.py`.

## Where the mechanism lives (decision-plane note)

ACE is the port that needed no optimizer surgery: its distinctive mechanism --
incremental delta updates curated into a playbook -- is entirely an **artifact
shape**, so it lives at the [strategy layer](strategies.md) (`ACEPlaybook`),
and the shipped merge pipeline runs unchanged underneath. Of the three
insertion layers in [choosing policies](policies.md), ACE uses the first alone;
there is deliberately no custom aggregator and no local policy class here.
