# GEPA — Reflective Prompt Evolution

> **Skill / prompt self-evolution.** Evolve an instruction prompt with a genetic,
> reflective loop whose parent selection is a **per-instance Pareto frontier**.
> Runs through [`evolve()`](evolution.md) with a custom `aggregator_factory`.
> Example: [`examples/gepa_prompt_evolution.py`](https://github.com/Birfy/concordia/blob/main/examples/gepa_prompt_evolution.py).

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

## Run it

```bash
python -m examples.gepa_prompt_evolution --dry-run
python -m examples.gepa_prompt_evolution --model claude-haiku-4-5
```

Offline tests: `tests/test_gepa_example.py` (incl. the Algorithm-2 selection).
