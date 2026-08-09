# GEPA — Reflective Prompt Evolution

> **Skill / prompt self-evolution.** Evolve an instruction prompt with a genetic,
> reflective loop whose parent selection is a **per-instance Pareto frontier**.
> Runs through [`evolve()`](evolution.md) with a custom `aggregator_factory`.
> Example: [`examples/gepa/gepa_prompt_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/gepa/gepa_prompt_evolution.py).

| | |
|---|---|
| **Paper** | *GEPA: Reflective Prompt Evolution Can Outperform RL* — Agrawal et al., 2025 ([arXiv:2507.19457](https://arxiv.org/abs/2507.19457)) |
| **Repo** | [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa) (also `dspy.GEPA`) |
| **Dataset** | **HotpotQA** (multi-hop QA, distractor), exact match |
| **Layer** | L2 prompt (`blast_radius=0.2`) |

## The algorithm

Two distinctive mechanisms, both preserved:

1. **Reflective mutation** (Algorithm 1 `UpdatePrompt`). On a failure the LLM
   reflects on the execution trace **and the natural-language feedback** (`μ_f`:
   predicted vs. gold), then writes a *new* instruction. This is the propose step.
2. **Pareto-based candidate selection** (Algorithm 2) — the reason GEPA beats
   greedy hill-climbing. Instead of always mutating the single best-*average*
   candidate, it keeps a **pool** scored on every `D_pareto` instance and samples
   the next parent from the **per-instance Pareto frontier**, weighted by how many
   instances a candidate uniquely wins — keeping complementary specialists alive.

`pareto_frontier` / `pareto_select` implement Algorithm 2 faithfully (per-instance
best → union of winners → dominance pruning → frequency-weighted sampling) and are
unit-tested.

## How it plugs into `evolve()`

The greedy `evolve()` loop always mutates the dev head; GEPA needs to mutate the
*Pareto-selected* parent. A custom `aggregator_factory` (`ParetoAggregator`)
supplies that: it scores each candidate on the held-out `D_pareto`, runs
Algorithm 2, and **commits the sampled parent as the dev head**, so the next
round mutates it. This is the sanctioned "swap the whole optimizer" hook.

```python
factory = pareto_aggregator_factory(artifact_id="gepa_prompt", seed=0)
evolve(tasks, reward, agent=gepa_agent(completion),
       strategy=InstructionSlot(), aggregator_factory=factory, blast_radius=0.2)
best = factory.holder["agg"].best_state["instruction"]   # GEPA returns best-average
```

## Fidelity notes

GEPA optimises a multi-module compound system with a rollout budget; here the
system is a single instruction module and the minibatch is the per-round worker
sample (raise `--workers`). The Pareto set is the held-out split.

## Plug-ins implemented

In [`examples/gepa/gepa_prompt_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/gepa/gepa_prompt_evolution.py):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| `ParetoWinFrequency` | selection ([seam](selection.md)) | Algorithm-2 frontier sampling as a named policy; `ParetoAggregator` delegates to it |
| **`InstructionSlot`** | `strategy=` | a single evolvable instruction module; each proposal replaces it (content-addressed) |
| **`ParetoAggregator`** / `pareto_aggregator_factory` | `aggregator_factory=` | GEPA's per-instance Pareto candidate selection; commits the sampled Pareto parent as the dev head |
| **`pareto_frontier` / `pareto_select`** | (pure, unit-tested) | Algorithm 2: per-instance best → union of winners → dominance pruning → frequency-weighted sampling |
| `gepa_agent()` | `agent=` | Generator + reflective-mutation actor (rewrites the instruction from trace + NL feedback) |

## Measured — HotpotQA with DeepSeek

The only row in the [parallelisation matrix](port-fidelity.md#the-parallelisation-matrix)
measured on **all three arms**, so the only one with a speedup that has a
denominator. 16 rollouts pinned on every arm, `--val-cap 8`,
`--reflective-merge`, `deepseek-v4-flash`, one seed:

| arm | wall | rollouts | calls | concurrency | test EM |
|---|---:|---:|---:|---:|---:|
| serial (upstream loop) | 609 s | 16 | 85 | **1.00×** | 0.600 |
| sync parallel, N=4 | 239 s | 16 | 75 | **1.85×** | 0.600 |
| async, N=4 | 140 s | 19 | 83 | **3.22×** | 0.850 |

**Concurrency is model-seconds over wall-clock** -- the part a worker pool is
responsible for. The control is the published loop and nothing else: one worker
*and* `eval_concurrency=1`, giving 607 model-seconds inside a 609-second
wall-clock. `--serial` alone only lowers `n_workers`, and a control that still
evaluates concurrently is already partly parallel; this is the first row here
that is not. The barrier is the whole difference between the two parallel rows,
and 1.85× lands where the heavy-tail stub in [efficiency](efficiency.md)
predicted (1.7–1.8×).

!!! warning "The test column is not a result"
    One seed, 20 test items: 0.600 → 0.850 is five questions. The async arm also
    spent **19 rollouts against the pinned 16** -- the barrier-free path has no
    round boundary to stop at -- so part of that column is extra budget rather
    than a scheduler. Read the concurrency column; the quality column needs the
    three seeds this repository asks for everywhere else.

!!! note "What this row was measured *before*"
    Three things were learned after it ran and are not reflected in it:

    * it used `--staleness guarded`, the default. ACE and EvoSkill were later
      measured discarding **60%** and **100%** of their evidence under it, and
      GEPA's single-key artifact should be worse still — every proposal edits the
      same key, so the head moves on every commit. The stale counters did not
      exist yet, so this row does not report how much it threw away.
    * `--reflective-merge` is on, and for this port that is a real semantics
      change, not only a cost lever: the round's diffs become **one** Pareto
      candidate instead of one per worker. It is why the parallel arm's model
      time falls 607 → 443 s — part of the speedup is work not done rather than
      work divided, and the `SEMANTICS` entry for this row says so.
    * the wall-clock is this machine and this endpoint.

The instruction the search found, from an earlier run on the same benchmark:

> *"Read the context carefully and connect information across multiple paragraphs
> to identify who matches all the clues in the question. Then give only the final
> answer as a short phrase, without explanation."*

Both halves are real HotpotQA failures: multi-hop evidence, and a model that
answers a short-span question with a paragraph.

## Run it

```bash
python -m examples.gepa.gepa_prompt_evolution --dry-run
python -m examples.gepa.gepa_prompt_evolution --model claude-haiku-4-5

# the three arms above
python -m bench.matrix_run --rows gepa --budget 16 --width 4 --seeds 0 \
    --eval-concurrency 4 --serial-eval-concurrency 1 \
    --provider claude --model deepseek-v4-flash --yes
```

Offline tests: `tests/test_gepa_example.py` (incl. the Algorithm-2 selection).
