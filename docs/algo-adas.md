# ADAS — Meta Agent Search

> **Harness self-evolution.** Evolve the *agentic system itself* — the control
> flow that orchestrates the model. Runs through [`evolve()`](evolution.md) with a
> custom `Strategy` + `aggregator_factory` at **L1** governance. Example:
> [`examples/adas/adas_meta_agent_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/adas/adas_meta_agent_search.py).

| | |
|---|---|
| **Paper** | *Automated Design of Agentic Systems* — Hu, Lu, Clune, 2024 ([arXiv:2408.08435](https://arxiv.org/abs/2408.08435)) |
| **Upstream code** | [`ShengranHu/ADAS`](https://github.com/ShengranHu/ADAS) |
| **Example** | [`examples/adas/adas_meta_agent_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/adas/adas_meta_agent_search.py) |
| **Domain** | **MGSM** (Multilingual Grade-School Math) — saturated on a current model, so `--dataset gpqa` (GPQA Diamond, shipped in the ADAS repo) is what has headroom |
| **Layer** | L1 harness (`blast_radius=0.6`) |
| **Fidelity** | `benchmark_faithful` — [what the classes mean](port-fidelity.md) |

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

## Measured results — GPQA Diamond

**There is no lift number, and this section is why.** This row is not measurable
with `deepseek-v4-flash` — and not "not measured yet":
the two constraints that would make it measurable point in opposite directions,
and the middle is empty. What follows is the evidence, because a page that leaves
a lift row blank without saying why invites someone to spend the two hours again.

### MGSM is saturated, in every language

The port's own dataset is a 2022 grade-school benchmark, and a current model has
finished with it. Measured, Chain-of-Thought alone:

| languages | CoT accuracy |
|---|---|
| `en` | **1.000** |
| `sw`, `te` | **1.000** |
| `th`, `bn` | **1.000** |

The hard-language escape does not exist. And saturation here is worse than a
missing lift: ADAS conditions its meta-agent on **the whole archive with its
fitness values**, so when all seven hand-designed seeds score 1.000 the archive
carries no signal about which control flow is better. The search is not merely
unable to improve — it is unable to *learn*, because everything it is shown is
tied.

### GPQA has the headroom, and costs ten times as much per call

Switching domain is not a fidelity break: ADAS searches four upstream (`_mgsm`,
`_gpqa`, `_drop`, `_arc`), and **GPQA Diamond's 198 rows ship inside the ADAS
repo** (`dataset/gpqa_diamond.csv`, no HuggingFace gate). `--dataset gpqa` selects
it. Chain-of-Thought measures **0.625** there — above the 0.250 floor of a
four-way choice and well under the ceiling.

The bill arrives with it. From a completed run:

| | |
|---|---|
| model calls | 28 |
| completion tokens | **143,246** — 5,116 per call |
| time in the model | 1,385 s — **49 s per call** |

A graduate-level science question makes this model generate five thousand tokens
of reasoning before it answers, and the reasoning is not optional: at
`max_tokens=512` and `4096` the reply comes back **empty**, the budget spent
before any visible text. `MAX_TOKENS` is 16384 for exactly this reason.

### Why that is fatal *here* specifically

Every port on this page's sibling pages pays per rollout. ADAS pays
`|val| x program_cost` per **candidate**, where a candidate is itself a
multi-call program — the seed archive alone is 19 calls per question:

| seed design | calls/question |
|---|---|
| Chain-of-Thought | 1 |
| Self-Refine (Reflexion) | 2 |
| Step-back Abstraction | 2 |
| Dynamic Assignment of Roles | 2 |
| Self-Consistency (CoT-SC) | 3 |
| LLM Debate | 4 |
| Quality-Diversity | 5 |
| **total, before round 0** | **19** |

At 49 s a call that is 19 x |val| x 49 s of seeding before the search starts.
And the calls inside one design are **serial** — Debate's second round waits on
its first — so concurrency parallelises across questions, never within a design.
Raising `--eval-concurrency` past `len(val)` buys nothing, which is why the seed
archive is now scored across designs concurrently as well, with the nested
fan-out bounded so `inner x outer` stays inside the budget.

**The two constraints are opposed.** Shrink `|val|` to afford the run and the
validation split returns to a ceiling — measured at `|val| = 7`, CoT scores
1.000 again and the strict comparison has nothing to rank. Keep `|val|` large
enough to separate designs and a single run is hours. There is no size that is
both affordable and readable on this model.

### What was tried

| configuration | outcome |
|---|---|
| MGSM, all languages | saturated, 1.000 |
| GPQA, `\|val\| = 45`, 16 rollouts | stopped at 40 min, still seeding |
| GPQA, `\|val\| = 29`, 12 rollouts | stopped at 40 min, still seeding |
| GPQA, `\|val\| = 7`, 8 rollouts, `--no-thinking` | completed; 7 seeds + 2 searched; val back to 1.000 |
| GPQA, `\|val\| = 7`, 4 rollouts | completed; 7 seeds + **0** searched; val 1.000, test 0.667, lift +0.000 |

`--no-thinking` deserves its own line, because it looks like the answer and is
not. It cuts a GPQA call from 52.6 s to 5.8 s for six items with 5/6 still
correct — a ninefold saving at no visible accuracy cost. But what ADAS searches
over is *reasoning orchestration*: Debate, Step-back and Self-Consistency are
ways of spending reasoning. Switch reasoning off and they collapse into
"answer directly", the seven seeds tie again, and the flag has deleted the thing
under measurement rather than accelerated it.

!!! danger "What would make this row measurable"
    A cheaper actor with genuine headroom on a reasoning benchmark — the two
    properties this model has one of at a time. Failing that, an actor whose
    per-call latency is small enough that `|val| >= 30` is affordable, since the
    ceiling problem is entirely a small-sample problem. Neither is a change to
    this port.

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
    | seed archive | 19 calls per item across the 7 seeds; the designs are scored **concurrently**, the calls inside one design are not |
    | `propose` | 3 Reflexion rounds, ~84 s |
    | candidate evaluation | `program_cost` calls per item, and Debate's second round waits on its first |

    Measured on GPQA with `deepseek-v4-flash`: **49 s and 5,116 completion tokens
    per call**. Multiply that by 19 x `|val|` and the seed archive alone is the
    length of an entire run of any other port on this site.

    The nested fan-out is bounded rather than multiplied: `_fitness` opens
    `min(--eval-concurrency, |val|)` threads, so the outer pool over designs is
    sized `--eval-concurrency // inner`. Without that the two levels multiply,
    which is the squared fan-out this repo was bitten by once already on GEPA's
    D_pareto sweep.

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

GPQA instead of MGSM, since MGSM is saturated (see above). `--generations 9999`
lets `--budget-rollouts` be what stops the run; `--eval-cache` memoises held-out
scores across processes, which is worth setting for a sweep whose cells share a
split and worth nothing for a single run:

```bash
python -m examples.adas.adas_meta_agent_search --yes \
    --dataset gpqa --langs en --per-lang 56 \
    --budget-rollouts 12 --generations 9999 --workers 4 \
    --async --staleness full --eval-concurrency 32 \
    --eval-cache ~/.cache/agentdescent/adas-gpqa \
    --model deepseek-v4-flash
```

That configuration did **not** finish inside 40 minutes on the endpoint measured
here — it was still scoring the seed archive. The section above is the honest
account of why, and of what would have to change for a lift number to exist.

Offline tests: `tests/test_adas_example.py`.
