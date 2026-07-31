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

MGSM is grade-school arithmetic, and a strong model answers it directly: at the
default `--langs en,es,fr` the seed archive already scores **1.000**, so the
search has no gradient and the archive merge accepts nothing.

Two settings give it room, and MGSM's own difficulty axis is language:

```bash
python -m examples.adas_meta_agent_search --hard --langs bn,sw,te,th \
    --per-lang 150 --provider openai --model deepseek-v4-flash --generations 4 --yes
```

`--hard` keeps the items a **plain single call** answers incorrectly, which is the
right filter here: what ADAS searches over *is* structure, so the items worth
keeping are the ones a structure-free call cannot already do. Over a 600-item
low-resource pool that leaves 47 genuinely hard questions (24 train / 12 val /
11 test).

ADAS is also the most expensive example — the searched agents are multi-step, so
one generation is hundreds of model calls.

## Run it

```bash
python -m examples.adas_meta_agent_search --dry-run
python -m examples.adas_meta_agent_search --model claude-haiku-4-5 --generations 6
python -m examples.adas_meta_agent_search --select dgm --langs en,es
```

Offline tests: `tests/test_adas_example.py`.
