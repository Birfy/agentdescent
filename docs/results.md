# Measured results

Every empirical claim in these docs, with the setup that produced it. All runs use
**`deepseek-v4-flash`** through `openai_compatible` unless stated otherwise, on
each algorithm's own dataset.

!!! warning "Read this before the table"
    **Four of the six algorithm ports have no headroom to demonstrate.** DeepSeek
    already scores 0.9–1.0 on FiNER-139, SearchQA and MGSM at these sample sizes,
    so there is nothing for a skill to add — and the framework correctly commits
    **nothing**, which is the result worth reporting.

    That is the acceptance gate doing its job. A naive implementation would happily
    accumulate plausible-sounding "improvements" against a flat signal; this one
    rejects them, and `result.outcomes()` says `below-threshold` rather than
    pretending. Skill evolution shows value exactly where there is a *gap* —
    a format the model does not follow, a convention it cannot guess, a tool it is
    not using.

## The algorithm ports

| Algorithm | Dataset | Held-out, before → after | Cost | What happened |
|---|---|---|---|---|
| **[GEPA](algo-gepa.md)** | HotpotQA | Pareto EM **0.500 → 0.600**; **test EM 0.700** | 80 calls, 10 min | Real lift. Learned to connect multi-paragraph evidence and answer with a short phrase |
| **[EvoSkill](algo-evoskill.md)** | OfficeQA | **0.000 → 0.000** (no tools) | 34 calls, 7 min | The dataset is HF-gated, so this falls back to a 12-row sample, and a *non-tool* model cannot find a figure inside a 272 KB bulletin. With a real tool-using agent: **58.0% → 65.7%** |
| **[ACE](algo-ace.md)** | FiNER-139 | 1.000 → 1.000 at `--top-k 10`; **87.0% → 95.7%** at `--top-k 40` | 42 calls, 2 min | Difficulty is the number of concepts: a 10-way choice is saturated, a 40-way one is not |
| **[SkillOpt](algo-skillopt.md)** | SearchQA | 0.900 → 0.900 | 54 calls, 5 min | Saturated. The strict gate saw one proposed edit and **rejected it**: 0 accepted / 1 rejected |
| **[ADAS](algo-adas.md)** | MGSM | 1.000 from round 0 | ~8 min/generation | Saturated; rejected everything (`+0/-1`). Stopped after 2 of 4 generations — the answer was settled |
| **[DGM](algo-dgm.md)** | *surrogate* | resolve-rate **0.000 → 0.300**; test 0.200 | offline | The objective is a capability-cover surrogate — real DGM runs SWE-bench in Docker. The archive, selection and staged escalation are faithful |

## The one-call path

[`evolve_skill`](quickstart-skill.md) on 40 real HotpotQA items, 12 held out —
the snippet from the front page, run as written:

| | held-out exact match |
|---|---|
| starting instruction (`"You are a helpful assistant."`) | 2/12 = **0.167** |
| after evolution | 7/12 = **0.583** |

Four rounds, stopped by `patience`; 338 calls, ~25 min. It learned *"Respond with
only the requested answer, omitting any extra explanation or restatement."* —
exactly the failure it was shown, since the model had been answering a short-span
question with a paragraph. `outcomes()` was `{'committed': 1, 'below-threshold': 3}`.

## Bringing your own agent

A two-step DeepSeek word-problem agent, scored in **integer cents** — a convention
stated nowhere in the prompt ([details](evolution.md#bring-an-agent-you-already-have)):

| | held-out |
|---|---|
| initial prompt | 3/12 = **0.250** |
| reflector blind to `Task.meta`, 8 rounds | 0.500 (plateau) |
| reflector reading `meta` | 12/12 = **1.000**, in one round |

It generalised rather than memorising, writing *"Express all monetary amounts as
integers representing cents, without dollar signs or decimal points."*

## Efficiency

Full breakdown in [Efficiency](efficiency.md).

| | result |
|---|---|
| Thread parallelism, 8 threads, real API calls | **7.1×** (pure-Python CPU work: 1.0×) |
| Whole `evolve()` run, uniform latency | **5.9×** of 8 workers |
| ...heavy-tailed latency (a reasoning model) | **2.4×** — the round barrier waits on the slowest worker |
| ...same, barrier-free `asynchronous=True` | **3.0×** |
| Gate concurrency (`eval_concurrency` 1 → 8) | **193.6 s → 90.0 s** on identical work |

## Things that were silently wrong

Each of these looked like a model or data problem and was not.

| | measured |
|---|---|
| `max_tokens=1024` with a reasoning model | **4 of 8** reflection prompts returned *empty*; at 3000, none did. Default is now 4096 |
| `last_number` parsing a dataset's answer column | GSM8K's ends `#### 72`, so every item scored 0: reported **0/7** vs the true **7/7** |
| `document_agent` inline fallback on real 266–390 KB docs | **1 of 3** correct, two answers empty — half of each document never reached the model |
| Settled-evidence pool, unbounded | 500 rejected oversized diffs retained **250 MB** no code path could read (now 2 MB) |
| A throttled backend (1 call in 3 fails), async | ended the run in **22 s with 0 sweeps**; now reaches **1.000** with 24 of 70 calls still failing |
| One transient, synchronous path | turned a 20-round run into **0 rounds** |
| A rollout wedged for 600 s | `evolve()` returned, then the process stayed alive; now exits in **4.5 s** |

## Reproducing

```bash
python -m examples.gepa_prompt_evolution --provider openai --model deepseek-v4-flash \
    --rounds 5 --fetch 40 --yes
```

Every example takes `--dry-run` (loads the dataset, prints the plan, no API calls)
and `--provider openai` for any OpenAI-compatible endpoint via `OPENAI_BASE_URL` +
`OPENAI_API_KEY`. Sample sizes above are deliberately small so a run costs minutes;
they are **not** the papers' full setups, and where a full setup needs heavy
infrastructure (SWE-bench in Docker, gated data) the boundary is stated rather than
hidden.
