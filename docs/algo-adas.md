# ADAS — Meta Agent Search

> **Harness self-evolution.** Evolve the *agentic system itself* — the control
> flow that orchestrates the model. Runs through [`evolve()`](evolution.md) with a
> custom `Strategy` + `aggregator_factory` at **L1** governance. Example:
> [`examples/adas/adas_meta_agent_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/adas/adas_meta_agent_search.py).

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

In [`examples/adas/adas_meta_agent_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/adas/adas_meta_agent_search.py):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| shipped [`Beam(1)`](selection.md) | selection ([seam](selection.md)) | best-of-archive head rule — exact match, no local subclass |
| **`AgentDesignStrategy`** | `strategy=` | a proposed agent (JSON) becomes a `Diff` on the one-slot agentic system; `render` returns the program for the interpreter |
| **`MetaSearchAggregator`** | `aggregator_factory=` | ADAS's keep-all archive with bootstrap-CI fitness |
| `make_propose(...)` | `propose=` | the meta-agent, conditioned on the whole archive (+ two Reflexion rounds) |
| **`Interpreter`** + **`seed_archive`** | (agent substrate) | the safe control-flow DSL (`cot`/`cot_sc`/`reflexion`/`debate`/`step_back`/`role_assignment`/`ensemble`) and the seven MGSM seeds |
| **`dgm_parent_weights`** | `--select dgm` | DGM's sigmoid×novelty rule as an alternative archive-conditioning strategy |

## Measured — MGSM with DeepSeek

MGSM is saturated for a strong model, so the search has no gradient without
`--hard`. Measured with `deepseek-v4-flash` over the **whole benchmark** — all 11
languages, 250 items each:

| | |
|---|---|
| pool | 2750 items |
| direct, structure-free single call | **0.919** |
| items it answers incorrectly | **222** (8.1%) |

Those 222 are what the search runs on, split 15 / 50 / 35:

```
Split    : 34 trigger / 110 val (fitness) / 78 test
```

The train share only *triggers* proposals — the meta-agent conditions on the
archive, not on the task `evolve()` hands it, so a generation consumes `--workers`
items and ignores the rest. Everything else measures. Splits are stratified by
language, because MGSM's languages differ by 8 points of baseline accuracy (`en`
0.964, `ja` 0.880) and an unstratified draw hands validation one mixture and test
another.

On a `--hard` subset the structure-free baseline is 0.000 by construction, so the
searched agent's test accuracy on its own says nothing. The run scores the best
hand-designed seed on the same split and reports both:

```
                                  val (search)   test (held out)
  best hand-designed seed              ?.???           ?.???
  best searched design                 ?.???           ?.???
  lift                                +?.???          +?.???
```

!!! note "The lift row is not filled in yet"
    A run over the split above is ~2 hours and 6k–17k model calls. One has not
    been completed against the current code, so this page does not claim a
    demonstrated lift — the table is the shape of the answer, not the answer.

### Give a reasoning model a real token budget

`deepseek-v4-flash` spends its budget on hidden reasoning first; visible content
is what is left. At the library default of 4096 the meta-agent returns **empty
content on every call**, so no design ever reaches the archive:

| `--max-tokens` | meta-agent replies | solver blank rate | solver CoT |
|---|---|---|---|
| 4096 | **0 / 4** | 13 / 40 | 0.275 |
| 16384 (default here) | **4 / 4** | 2 / 40 | 0.325 |

An empty completion does not raise — `_extract_int("")` is `None` and that scores
as a wrong answer, so a starved run reports a low accuracy indistinguishable from
a model that cannot do the problems. The run counts blank replies and warns, and
the pre-flight check sends a *reasoning* prompt and aborts if it comes back
empty. You are billed for tokens generated, not for the cap.

!!! warning "This is by far the most expensive example"
    Every candidate is scored on every validation item, and each score is a
    *multi-step* program. Wall-clock is set by the **serial** chains, not by
    fan-out, so past `--eval-concurrency >= |val|` more concurrency buys nothing:

    | chain | length |
    |---|---|
    | seed archive | 19 sequential calls per item (7 seeds) |
    | `propose` | 3 Reflexion rounds, ~84 s |
    | candidate evaluation | `program_cost` calls per item, candidates run one after another |

    A proposed design may cost up to `MAX_PROGRAM_CALLS` (10) calls per question
    against a seed average of 2.7, which is what dominates a generation. The
    budget line reports both ends of the range for exactly this reason.

    | knob | effect |
    |---|---|
    | `--generations`, `--workers` | candidates searched |
    | `--hard-keep N` | caps the pool, and with it every sweep |
    | `--eval-concurrency N` | set it to at least `|val|` |
    | `--max-tokens`, `--timeout` | see above |

## Run it

Point the example at your own endpoint with two environment variables — they are
read at call time and never stored by the repo (full list, including Claude and
local servers: [Configuring your provider and key](agents.md#configuring-your-provider-and-key)):

```bash
export OPENAI_BASE_URL=https://api.deepseek.com     # or your gateway
export OPENAI_API_KEY=sk-...
```

Inspect the setup before it costs something: `--dry-run` prints the requested
MGSM/runtime configuration and returns before loading data or models, so it needs
neither network nor an API key:

```bash
python -m examples.adas.adas_meta_agent_search --dry-run
python -m examples.adas.adas_meta_agent_search --select dgm --langs en,es
```

The settings the numbers above come from — the whole benchmark, hard subset, a
split that leaves something to measure. The baseline pass is cached per
(model, question), so only the first run pays for it:

```bash
python -m examples.adas.adas_meta_agent_search \
    --provider openai --model deepseek-v4-flash \
    --hard --langs bn,de,en,es,fr,ja,ru,sw,te,th,zh --per-lang 250 \
    --generations 4 --workers 3 --eval-concurrency 128 \
    --train-frac 0.15 --test-frac 0.35 --yes
```

Offline tests: `tests/test_adas_example.py`.
