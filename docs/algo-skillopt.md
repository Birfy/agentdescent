# SkillOpt — ReflACT

> **Skill-document self-evolution.** Train a single markdown skill doc as the
> external state of a frozen agent, with optimizer discipline. Runs through
> [`evolve()`](evolution.md) with a custom `Strategy` + `aggregator_factory`.
> Example: [`examples/skillopt/skillopt_skill_training.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillopt/skillopt_skill_training.py).

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

In [`examples/skillopt/skillopt_skill_training.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillopt/skillopt_skill_training.py):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| `StrictImprovement` | acceptance ([seam](acceptance-policies.md)) | the strict full-held-out gate as a named policy |
| **`SkillDocStrategy`** | `strategy=` | turns the analyst's bounded edit patch (`append/insert_after/replace/delete`) into a `Diff` on the one-slot skill document |
| **`StrictGateAggregator`** | `aggregator_factory=` | strict held-out-EM accept gate + the rejected-edit buffer (remembered in-epoch) |
| **`LRScheduler`** | (edit budget) | the integer "learning-rate" cap on edits per step (`constant`/`linear`/`cosine`) |
| `make_propose(...)` | `propose=` | the analyst — one failed rollout → a budget-capped edit patch |

## Measured — SearchQA with DeepSeek

Barrier-free (`--async`), 4 workers, **60 rollouts pinned**, `--reflective-merge`,
`--staleness full`, `deepseek-v4-flash`, one seed. `--hard --hard-passes 2` over
a 500-item pool keeps the 93 the seed skill answers wrong twice running, split
55 train / 19 val / 19 test.

| | value |
|---|---|
| val hard-EM, before → after | **0.053 → 0.211** |
| test hard-EM | 0.316 |
| edits accepted / rejected | **3 / 40** |
| stale discarded | 0 / 0 |
| sweeps | 10 (commits at 1, 5, 9) |

**The gate rejecting 93% is the algorithm, not a failure.** Upstream's rule is
`candidate > current` on the full held-out split with no tolerance and no
tie-break (`evaluation/gate.py`), so an edit has to fix at least one whole val
question to survive. At 19 val items one question is 5.3 points, which is also
the step size visible in the sweep log: every commit moves val by exactly one
item. Edits that make three answers *closer* without making any of them match are
invisible to it and are correctly dropped.

**What it learned, and why the gains stop.** The three accepted edits say one
thing three ways:

```
For final answers, output the target's conventional name as a single noun
phrase... Do not add parenthetical explanations, synonyms, full official names,
or extra qualifiers unless the question explicitly asks for them.

When a target can be named by a common/vernacular name and also by a longer
formal/descriptive name, prefer the common name that would be used in a
reference answer.

If the model is tempted to include extra information for safety, it should
instead leave the answer minimal: a direct identification question is answered
by the identifier, not by a definition or parenthetical gloss.
```

Under hard-EM against a gold like `(Lou) Gehrig`, answering *"Lou Gehrig, the
Yankees first baseman"* scores zero. Most of what `--hard` selects is therefore
not a knowledge gap but a **formatting** one, and formatting is what the
optimizer generalises. That pays once — the first "stop adding qualifiers" edit
fixes several questions at once — and the restatements after it add nothing,
which is why sweeps 2–4 and 6–8 commit nothing at all.

The same shape appears in [EvoSkill](algo-evoskill.md#empirical-results-claude-code-as-the-base-agent-on-finqa),
where a tolerance-based numeric scorer produces skills about rounding. Two ports,
two datasets, two scorers, and in both the induced skill targets the metric's
surface. It is worth knowing before reading a lift as evidence about capability.

## Making it measurable — `--hard`

SearchQA is saturated for a strong model — the seed skill answers 9 of 10 out of
the box, so the strict gate correctly accepts nothing and the run proves only
that the gate works. `--hard` keeps the dataset and drops the questions carrying
no signal: run the seed skill over a wider pool and keep what it gets **wrong**.

!!! danger "One pass selects unlucky answers, not hard ones"
    `--hard-passes` defaults to **3** because a single baseline measurement does
    not survive a sampled model. Measured here at `passes=1`: the filter kept
    only questions the seed skill got wrong, and the gate then scored that same
    val split at **1.000** — every one of them answered correctly the second
    time. That is a ceiling, and a strict `candidate > current` gate can accept
    nothing above it, so the run rejected all five proposals and looked like a
    broken optimizer.

    Selecting on one noisy measurement selects the tail of the noise, and
    re-measuring regresses to the mean. At `passes=2` the same pool yields a
    0.053 baseline — real headroom. Survival was 18.6% at 2 passes against 17% at
    3, so most of the luck is already gone after the second.

    The filter is not free: it costs `pool × passes` calls, which here was 1,000
    — more than the training run it enables. Budget for it.

That makes the *benchmark* harder, so its numbers are not comparable with numbers
from the full split — say which you used. The underlying helper,
[`select_hard`](dataloader.md), works on any item list and any scorer.

## Run it

```bash
python -m examples.skillopt.skillopt_skill_training --dry-run
python -m examples.skillopt.skillopt_skill_training --model claude-haiku-4-5 --lr 4

# the run measured above
python -m examples.skillopt.skillopt_skill_training --yes --seed 0 \
    --minibatch 4 --hard --hard-passes 2 --train 300 --val 200 --steps 9999 \
    --budget-rollouts 60 --reflective-merge --staleness full \
    --async --async-ratio 3 --max-seconds 3600 --eval-concurrency 8 \
    --model deepseek-v4-flash
```

`--reflective-merge` scores **one fused patch per step** instead of one candidate
per worker, which is upstream's shape rather than a departure from it: a ReflACT
step emits a single patch of up to `lr` edits and evaluates it once. The strict
gate is untouched; what it scores is. `--staleness full` rebases a patch onto the
current document instead of discarding it when the merger has moved on — the
gate is already the verification, so re-verifying each patch before it gets there
buys nothing.

Offline tests: `tests/test_skillopt_example.py`.
