# Measured results

Every number here comes from a real run on the algorithm's own dataset with
**`deepseek-v4-flash`** through `openai_compatible`. Each row gives the settings,
so you can reproduce it.

## The algorithm ports

| Algorithm | Dataset | Settings | Held-out, before → after | Cost |
|---|---|---|---|---|
| **[GEPA](algo-gepa.md)** | HotpotQA | `--rounds 5 --fetch 40` | Pareto EM **0.500 → 0.600**; test EM **0.700** | 80 calls, 10 min |
| **[ACE](algo-ace.md)** | FiNER-139 | `--top-k 120 --rounds 8 --workers 4` | val **0.844 → 0.889**; test **0.884**, 2 bullets | 403 calls, 20 min |
| **[SkillOpt](algo-skillopt.md)** | SearchQA | `--hard --steps 6` | val hard-EM **0.250 → 0.500**; test **0.450** | 6 steps |
| **[EvoSkill](algo-evoskill.md)** | FinQA | `--dataset finqa --iterations 5` | val **0.487 → 0.573**; test **0.617**, 1 skill | 115 calls, 4 min |
| **[DGM](algo-dgm.md)** | surrogate | `--generations 4` | resolve-rate **0.000 → 0.300**; test 0.200 | offline |
| **[ADAS](algo-adas.md)** | MGSM | `--hard --langs bn,sw,te,th` | **no lift demonstrated** — see below | 791 calls, 24 min |

!!! warning "ADAS is the exception, and the reason is cost"
    MGSM is saturated at the default settings, and `select_hard` does find a real
    subset (47 of 600 items) — but evaluating one candidate there is a *multi-step
    program per item*, so a three-generation run is ~9000 calls and hours long.
    Shrinking it to fit a budget shrinks the measurement too: at 6 train / 3 val /
    3 test the archive rejected every candidate and held-out test was 0.000 on
    three items. That is a measurement too small to read, not a demonstration and
    not a bug. [Details](algo-adas.md#measured-mgsm-with-deepseek).

Each learned something specific to the failure it was shown:

* **GEPA** — *"connect information across multiple paragraphs… then give only the
  final answer as a short phrase, without explanation."*
* **EvoSkill** — *"round your answer to the same number of decimal places shown in
  that table… compute the unrounded value first, then round once at the end."*

## Choosing a setting that can show a lift

An evolution run is only as informative as the gap it is given. A strong model
already scores 0.9–1.0 on several of these benchmarks at their smallest settings,
and there the framework correctly commits nothing — `outcomes()` reports
`below-threshold` rather than accumulating changes against a flat signal.

Two levers set the difficulty:

| lever | where | effect |
|---|---|---|
| the benchmark's own difficulty parameter | ACE `--top-k`, ADAS `--langs` | ACE at `--top-k 10` scores 1.000; at 120 it goes **0.844 → 0.889** |
| [`select_hard`](dataloader.md#turning-a-saturated-benchmark-into-one-with-headroom-select_hard) (`--hard`) | SkillOpt, ADAS | keeps the items a baseline gets wrong — SkillOpt: 69 of 280 |

!!! warning "A hard subset is a different benchmark"
    SkillOpt's `0.250 → 0.500` is measured on the subset its seed skill fails, not
    on the full split where it scores 0.900. The two are not comparable — say which
    one you used.

## The one-call path

[`evolve_skill`](quickstart-skill.md) on 40 real HotpotQA items, 12 held out — the
snippet from the front page, run as written:

| | held-out exact match |
|---|---|
| starting instruction (`"You are a helpful assistant."`) | 2/12 = **0.167** |
| after evolution | 7/12 = **0.583** |

Four rounds, stopped by `patience`; 338 calls, ~25 min. It learned *"Respond with
only the requested answer, omitting any extra explanation or restatement."*
`outcomes()` was `{'committed': 1, 'below-threshold': 3}` — one proposal cleared
the gate, three did not beat it.

## Bringing your own agent

A two-step DeepSeek word-problem agent, scored in **integer cents** — a convention
stated nowhere in the prompt ([how](evolution.md#bring-an-agent-you-already-have)):

| | held-out |
|---|---|
| initial prompt | 3/12 = **0.250** |
| after evolution | 12/12 = **1.000**, in one round |

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

`n_workers` buys rollout parallelism and `eval_concurrency` buys gate parallelism;
they are independent, and a run slower than its worker count suggests usually
wants the second.

## Reproducing

```bash
python -m examples.gepa_prompt_evolution --provider openai --model deepseek-v4-flash \
    --rounds 5 --fetch 40 --yes
```

Every example takes `--dry-run` (loads the dataset, prints the plan, no API calls)
and `--provider openai` for any OpenAI-compatible endpoint via `OPENAI_BASE_URL` +
`OPENAI_API_KEY`. Sample sizes are deliberately small so a run costs minutes; they
are **not** the papers' full setups, and where a full setup needs heavy
infrastructure (SWE-bench in Docker, gated data) the boundary is stated on the
algorithm's page.
