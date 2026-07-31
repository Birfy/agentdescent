# ADAS — Meta Agent Search

> **Harness self-evolution.** Evolve the *agentic system itself* — the control
> flow that orchestrates the model. Runs through [`evolve()`](evolution.md) with a
> custom `Strategy` + `aggregator_factory` at **L1** governance. Example:
> [`examples/adas_meta_agent_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/adas_meta_agent_search.py).

| | |
|---|---|
| **Paper** | *Automated Design of Agentic Systems* — Hu, Lu, Clune, 2024 ([arXiv:2408.08435](https://arxiv.org/abs/2408.08435)) |
| **Repo** | [`ShengranHu/ADAS`](https://github.com/ShengranHu/ADAS) |
| **Dataset** | **MGSM** (Multilingual Grade-School Math) |
| **Layer** | L1 harness (`blast_radius=0.6`) |

## The algorithm

Meta Agent Search:

1. Seed an **archive** with hand-designed building blocks (CoT, Self-Consistency,
   Reflexion, Debate, Step-back, Quality-Diversity, Role-Assignment).
2. A **meta-agent**, conditioned on the *entire archive* (designs + fitness),
   proposes the next agent, then does two Reflexion refinement rounds.
3. **Evaluate** it on the MGSM validation set; fitness = bootstrap-CI mean.
4. **Keep-all** append to the archive; repeat. Return the best.

## How it plugs into `evolve()`

* `strategy=AgentDesignStrategy()` — a proposed agent (JSON) → a `Diff` on the
  one-slot "agentic system"; `render` returns the program for the interpreter.
* `propose` — the meta-agent, conditioned on the whole archive (shared via
  `AdasContext`), so it does not depend on the specific per-task input `evolve()`
  hands it.
* `aggregator_factory` → `MetaSearchAggregator` — the keep-all archive; it scores
  each candidate with bootstrap-CI fitness and keeps the best design as the dev
  head. `--select dgm` swaps archive conditioning for the DGM parent-selection
  rule.

`classify()` prints **L1_SLOW** — a harness change is high-blast-radius.

## Safety substitution (documented)

ADAS `exec`s model-written Python `forward()` functions. To avoid arbitrary code
execution, an agent here is a **composable control-flow program** in a small
validated DSL (`AGENT_BLOCKS`) run by a safe interpreter. The Meta Agent Search
*loop*, the seed archive, MGSM scoring, and the keep-all archive are faithful;
only the agent *substrate* is a safe DSL instead of raw `exec`.

## Plug-ins implemented

In [`examples/adas_meta_agent_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/adas_meta_agent_search.py):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| **`AgentDesignStrategy`** | `strategy=` | a proposed agent (JSON) becomes a `Diff` on the one-slot agentic system; `render` returns the program for the interpreter |
| **`MetaSearchAggregator`** | `aggregator_factory=` | ADAS's keep-all archive with bootstrap-CI fitness |
| `make_propose(...)` | `propose=` | the meta-agent, conditioned on the whole archive (+ two Reflexion rounds) |
| **`Interpreter`** + **`seed_archive`** | (agent substrate) | the safe control-flow DSL (`cot`/`cot_sc`/`reflexion`/`debate`/`step_back`/`role_assignment`/`ensemble`) and the seven MGSM seeds |
| **`dgm_parent_weights`** | `--select dgm` | DGM's sigmoid×novelty rule as an alternative archive-conditioning strategy |

## Measured — MGSM with DeepSeek

**This is the one shipped port where I could not demonstrate a lift**, and the
reason is worth stating rather than hiding.

At the default settings MGSM is saturated: the seed archive already scores 1.000
on the sampled items, so Meta Agent Search has no gradient and the archive merge
accepts nothing. `--hard` is the lever — but MGSM resists it from both sides:

| pool | items a single structure-free call gets **wrong** |
|---|---|
| `--per-lang 40` on `bn,sw,te,th` (160) | fewer than 12 — `select_hard` warns and tops up |
| `--per-lang 150` (600) | **47** (23 train / 12 val / 11 test) |

So a subset with real signal exists, but it is ~8% of a large pool. The bind is
cost: with 23 training items a three-generation run is ~9000 model calls and takes
hours. Shrinking it to fit a budget shrinks the *measurement* with it — at
`--hard-keep 12` the split is 6 train / 3 val / 3 test, and a 3-item validation set
cannot separate anything:

```
round 0  reward=0.400  +0/-1
round 1  reward=0.400  +0/-1
test accuracy (held out): 0.000        # 3 items
791 calls, 5653 s in the model, 24 min wall-clock
```

The archive rejected every candidate. On this evidence that is neither a working
demonstration nor a bug — it is a measurement too small to read. What the run
*does* show is that the loop, the archive, the selection rule and the gate all
execute against a real dataset.

To get a number you can trust here, budget for `--per-lang 150` with no
`--hard-keep` and expect hours, or use a weaker base model so the plain benchmark
has headroom.

!!! warning "This is by far the most expensive example"
    Every candidate is scored on every training item, and each score is a
    *multi-step* program — self-consistency and debate make several model calls per
    question. One generation over 23 items is on the order of a thousand calls, so
    a three-generation run takes hours even fully parallel.

    Two knobs control the cost directly:

    | | |
    |---|---|
    | `--hard-keep N` | cap the hard subset. Evaluation is *candidates × items × multi-step calls*, so this is the strongest lever |
    | `--eval-concurrency N` | how many examples are scored at once (default 16). Purely I/O bound |

    `--hard-keep` caps the **pool**, which is then split 50/25/25 — so
    `--hard-keep 12` leaves only 6 training items, and the fan-out cannot exceed
    that. Ask for roughly four times the training set you want.

## Run it## Run it

```bash
python -m examples.adas_meta_agent_search --dry-run
python -m examples.adas_meta_agent_search --model claude-haiku-4-5 --generations 6
python -m examples.adas_meta_agent_search --select dgm --langs en,es
```

Offline tests: `tests/test_adas_example.py`.
