# SkillOpt — ReflACT

> **Skill-document self-evolution.** Train a single markdown skill doc as the
> external state of a frozen agent, with optimizer discipline. Runs through
> [`evolve()`](evolution.md) with a custom `Strategy` + `aggregator_factory`.
> Example: [`examples/skillopt_skill_training.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillopt_skill_training.py).

| | |
|---|---|
| **Paper** | *SkillOpt: Executive Strategy for Self-Evolving Agent Skills* — Yang et al., 2025 ([arXiv:2605.23904](https://arxiv.org/abs/2605.23904)) |
| **Repo** | [`microsoft/SkillOpt`](https://github.com/microsoft/SkillOpt) (PyPI `skillopt`) |
| **Dataset** | **SearchQA** (single-turn text QA), EM / F1 |
| **Layer** | L2 skill (`blast_radius=0.2`) |

## The algorithm (ReflACT)

Four load-bearing invariants, reproduced from the repo (`engine/trainer.py`,
`optimizer/skill.py`, `evaluation/gate.py`, `optimizer/scheduler.py`):

1. **Bounded string edits** on one markdown doc — ops `{append, insert_after,
   replace, delete}` (`apply_patch`). The doc is the whole trainable state,
   injected into the frozen agent by prompt concatenation (zero deployment calls).
2. **Strict held-out accept gate** — a candidate is accepted only if it *strictly
   improves* the held-out validation hard-EM over the **current** skill (default
   `gate_metric=hard`). Greedy hill-climbing — the same shape as `evolve()`.
3. **Textual learning-rate budget** — an integer cap on edits per step
   (`optimizer/scheduler.py`); AgentDescent's `trust_region_ops` analogue.
4. **Rejected-edit buffer** — rejected edits are remembered in-epoch and fed back
   to the optimizer so it stops re-proposing them.

## How it plugs into `evolve()`

* `strategy=SkillDocStrategy(ctx)` — the analyst's edit patch → a `Diff` on the
  one-slot skill document (and it records `diff_id → edits` for the buffer).
* `propose` — the analyst (returns a budget-capped patch, sees the buffer).
* `aggregator_factory` → `StrictGateAggregator` — the strict-EM gate; it commits
  the best strictly-improving candidate as the dev head, buffers rejected edits,
  and advances the LR schedule each round.

A shared `SkillOptContext` (buffer, LR budget, edit registry, stats) is closed
over by both the propose step and the aggregator.

The epoch-level *slow-update* and *meta-skill* stabilisers are optional in the
repo and omitted from this minimal-but-faithful slice.

## Plug-ins implemented

In [`examples/skillopt_skill_training.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillopt_skill_training.py):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| **`SkillDocStrategy`** | `strategy=` | turns the analyst's bounded edit patch (`append/insert_after/replace/delete`) into a `Diff` on the one-slot skill document |
| **`StrictGateAggregator`** | `aggregator_factory=` | strict held-out-EM accept gate + the rejected-edit buffer (remembered in-epoch) |
| **`LRScheduler`** | (edit budget) | the integer "learning-rate" cap on edits per step (`constant`/`linear`/`cosine`) |
| `make_propose(...)` | `propose=` | the analyst — one failed rollout → a budget-capped edit patch |

## Measured — SearchQA with DeepSeek

`--steps 5 --provider openai --model deepseek-v4-flash`:

| | full split | `--hard` subset |
|---|---|---|
| items | 120 train / 80 val / 80 test | 29 / 20 / 20 (69 of 280 kept) |
| val hard-EM, before → after | 0.900 → 0.900 | **0.250 → 0.500** |
| test hard-EM | 0.900 | **0.450** |
| edits accepted / rejected | 0 / 1 | 3 / 3 |

On the full split the seed skill already answers 9 of 10, so a skill document has
nothing to add and the strict gate accepts no edit.

`--hard` keeps the 69 questions of 280 that the seed skill gets **wrong**, and on
those the skill document **doubles** hard-EM. The gate stays strict there too:
3 of 6 proposed edits are still rejected.

!!! warning "The two columns are different benchmarks"
    0.250 is not "worse than 0.900" — it is the score on a subset selected for
    being unsolved. Report which one you used.

## Making it measurable — `--hard`

SearchQA is saturated for a strong model, so the run above proves the gate works
and nothing else. `--hard` keeps the dataset and drops the questions carrying no
signal: one pass of the seed skill over a wider pool, keeping only what it gets
**wrong**.

```bash
python -m examples.skillopt_skill_training --hard \
    --provider openai --model deepseek-v4-flash --steps 5 --yes
```

That makes the *benchmark* harder, so its numbers are not comparable with numbers
from the full split — say which you used. The underlying helper,
[`select_hard`](dataloader.md), works on any item list and any scorer.

## Run it

```bash
python -m examples.skillopt_skill_training --dry-run
python -m examples.skillopt_skill_training --model claude-haiku-4-5 --lr 4
```

Offline tests: `tests/test_skillopt_example.py`.
