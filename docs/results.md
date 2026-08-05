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
| **[ADAS](algo-adas.md)** | MGSM | `--hard`, all 11 languages | direct **0.919** → 222-item hard subset; lift **not yet measured** | ~2 h, 6k–17k calls |

!!! note "What the *before* number is, per row"
    GEPA, EvoSkill and SkillOpt score the **seed** artifact explicitly before
    evolving it, so their "before" is a true baseline. ACE's is the **first round's**
    held-out measurement, which is taken *after* that round's merge — the seed is
    never scored on its own, because doing so would buy an extra val sweep of real
    model calls and change the cost column. Read it as "where the run started
    reporting", not "what the seed scored"; if round 0 committed, the real lift is
    slightly larger than the row shows.

!!! warning "ADAS is the exception: the lift row is still empty"
    Everything else in this table is a completed before → after. ADAS is not, and
    the honest reason is that no run against the current code has finished.

    What *is* measured: over the whole benchmark (2750 items, 11 languages)
    `deepseek-v4-flash` answers **0.919** directly, leaving **222** items with
    real signal — enough for a 34 / 110 / 78 split, where the previous attempt had
    47 items and split them 23 / 12 / 11.

    One thing to know before running it: this example needs `--max-tokens` set for
    a reasoning model. At the library default the meta-agent returns empty content
    on every call and no design reaches the archive — and an empty completion
    scores as a wrong answer rather than raising.
    [Details](algo-adas.md#measured-mgsm-with-deepseek).

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

| | result | re-measured? |
|---|---|---|
| Thread parallelism, 8 threads, real API calls | **7.1×** (pure-Python CPU work: 1.0×) | needs an API key |
| Whole `evolve()` run, uniform latency | **5.9×** of 8 workers | needs an API key |
| ...heavy-tailed latency (a reasoning model) | **2.4×** — the round barrier waits on the slowest worker | needs an API key |
| ...same, barrier-free `asynchronous=True` | **3.0×** | needs an API key |
| Gate concurrency (`eval_concurrency` 1 → 8) | **193.6 s → 90.0 s** on identical work | needs an API key |

!!! note "These five were not re-measured when the reference runtimes became adapters"
    They all come through [`evolve()`](evolution.md) against a real model, and
    `evolve()`'s default behaviour did not change — `refresh_interval` defaults to
    `1`, which is exactly the old snapshot discipline, and the new `RoundInfo`
    fields are reported rather than acted on. So there is no reason to expect
    them to have moved, and no way to confirm it without spending on the API.

    The [offline table](efficiency.md#the-configuration-matrix-bench) *was*
    re-measured, and one column there moved for a reason worth knowing: `stale%`
    was understated by roughly a factor of two, because two staleness gates were
    both counting into one denominator.

`n_workers` buys rollout parallelism and `eval_concurrency` buys gate parallelism;
they are independent, and a run slower than its worker count suggests usually
wants the second.

## Reproducing

```bash
python -m examples.gepa_prompt_evolution --provider openai --model deepseek-v4-flash \
    --rounds 5 --fetch 40 --yes
```

Every faithful port takes a zero-network `--dry-run` and `--provider openai` for
any OpenAI-compatible endpoint via `OPENAI_BASE_URL` + `OPENAI_API_KEY`. Sample
sizes are deliberately small so a run costs minutes; they
are **not** the papers' full setups, and where a full setup needs heavy
infrastructure (SWE-bench in Docker, gated data) the boundary is stated on the
algorithm's page.
