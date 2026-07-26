# SkillOpt — ReflACT

> **Skill-document self-evolution.** Train a single markdown skill doc as the
> external state of a frozen agent, with optimizer discipline. Runs through
> [`evolve()`](evolution.md) with a custom `Strategy` + `aggregator_factory`.
> Example: [`examples/skillopt_skill_training.py`](https://github.com/Birfy/concordia/blob/main/examples/skillopt_skill_training.py).

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
   (`optimizer/scheduler.py`); Concordia's `trust_region_ops` analogue.
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

In [`examples/skillopt_skill_training.py`](https://github.com/Birfy/concordia/blob/main/examples/skillopt_skill_training.py):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| **`SkillDocStrategy`** | `strategy=` | turns the analyst's bounded edit patch (`append/insert_after/replace/delete`) into a `Diff` on the one-slot skill document |
| **`StrictGateAggregator`** | `aggregator_factory=` | strict held-out-EM accept gate + the rejected-edit buffer (remembered in-epoch) |
| **`LRScheduler`** | (edit budget) | the integer "learning-rate" cap on edits per step (`constant`/`linear`/`cosine`) |
| `make_propose(...)` | `propose=` | the analyst — one failed rollout → a budget-capped edit patch |

## Run it

```bash
python -m examples.skillopt_skill_training --dry-run
python -m examples.skillopt_skill_training --model claude-haiku-4-5 --lr 4
```

Offline tests: `tests/test_skillopt_example.py`.
