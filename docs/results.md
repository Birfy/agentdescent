# Measured results

Every number here comes from a real run on the algorithm's own dataset with
**`deepseek-v4-flash`** through `openai_compatible`. Each row gives the settings,
so you can reproduce it.

## The algorithm ports

| Algorithm | Dataset | Settings | Held-out, before → after | Cost | Difficulty knob |
|---|---|---|---|---|---|
| **[GEPA](algo-gepa.md)** | HotpotQA | `--rounds 5 --fetch 40` | Pareto EM **0.500 → 0.600**; test EM **0.700** | 80 calls, 10 min | none |
| **[ACE](algo-ace.md)** | FiNER-139 | `--top-k 120 --rounds 8 --workers 4` | val **0.844 → 0.889**; test **0.884**, 2 bullets | 403 calls, 20 min | ⚠︎ `--top-k` |
| **[SkillOpt](algo-skillopt.md)** | SearchQA | `--hard --steps 6` | val hard-EM **0.250 → 0.500**; test **0.450** | 6 steps | ⚠︎ `--hard` |
| **[EvoSkill](algo-evoskill.md)** | FinQA | `--dataset finqa --iterations 5` | val **0.487 → 0.573**; test **0.617**, 1 skill | 115 calls, 4 min | none |
| **[DGM](algo-dgm.md)** | surrogate | `--generations 4` | resolve-rate **0.000 → 0.300**; test 0.200 | offline | none |
| **[ADAS](algo-adas.md)** | MGSM | `--hard`, all 11 languages | direct **0.919** → 222-item hard subset; lift **not yet measured** | ~2 h, 6k–17k calls | ⚠︎ `--hard` |

**Every row above was measured with `deepseek-v4-flash`.** That is stated at the
top of this page, and it is not decoration: ⚠︎ marks a row whose *difficulty*
comes from a knob calibrated against that model. Those rows do not transfer.

!!! danger "Re-run on `glm-5.2`: the two ⚠︎ rows lost their lift entirely"
    Not a contradiction of the numbers above — a different model, so a different
    measurement. What it shows is which rows are portable:

    | | `deepseek-v4-flash` (published) | `glm-5.2` (re-run) |
    |---|---|---|
    | DGM | `0.000 → 0.300`, test 0.200 | identical, to the digit |
    | GEPA | Pareto `0.500 → 0.600` | `0.500 → 0.600`, test 0.800 |
    | EvoSkill | val `0.487 → 0.573`, 1 skill | val `0.500 → 0.577`, test 0.613, 1 skill |
    | **ACE** ⚠︎ | val `0.844 → 0.889`, 2 bullets | val **`0.867 → 0.867`**, 1 bullet, 8 rounds, 413 calls |
    | **SkillOpt** ⚠︎ | val hard-EM `0.250 → 0.500` | val **`1.000 → 1.000`**, **0 edits accepted** |

    The mechanism ran correctly in all five — ACE spent 413 calls against a
    published 403, so it did the same work. What changed is that there was
    nothing left to learn:

    * **SkillOpt.** `--hard` keeps the items the seed gets wrong. `glm-5.2`
      answers 95% of SearchQA correctly, so `select_hard` found **2 hard items in
      40** and **1 in 20**, then padded to its 12-item floor with items the model
      already solves. Validation was 1.000 from the first round; six rounds of
      edits were all correctly rejected.
    * **ACE.** `--top-k 120` sets how many XBRL concepts compete. `glm-5.2`
      starts at 0.867 where `deepseek-v4-flash` starts at 0.844, and the residual
      errors are not the kind one playbook bullet fixes.

    The three unmarked rows reproduce because their difficulty does not depend on
    the model: DGM's objective is a deterministic surrogate, HotpotQA's multi-hop
    structure is hard regardless, and FinQA's decimal-place convention is a
    *convention* — no amount of model capability guesses how many places the table
    used.

    **Re-calibrate the knob before comparing across models.** For SkillOpt that
    means a pool large enough that `select_hard` finds genuinely hard items
    without padding — at a 5% hard rate, roughly 240 items per split rather than
    40.

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

| | result | how |
|---|---|---|
| Thread parallelism, 8 threads, real API calls | **5.8×** on `glm-5.2` (pure-Python CPU work: 1.1×) | `examples.efficiency --only gil --model <id>` |
| Whole `evolve()` run, uniform latency | **1.8×** of 8 workers, end-to-end | `--only distribution` |
| ...heavy-tailed latency (a reasoning model) | **1.7×** — the round barrier waits on the slowest worker | `--only distribution` |
| ...same, barrier-free | **2.65×** on the dispatch microbenchmark | `--only async` |
| Gate concurrency (`eval_concurrency` 1 → 8) | **3.6 s → 1.2 s**, saturating past the held-out size | `--only gate` |

!!! warning "Every row here was re-measured, and four of them had no script"
    The previous version of this table (7.1× / 5.9× / 2.4× / 3.0× / 193.6 s →
    90.0 s) was produced by hand and could not be re-run: nothing in the
    repository generated it. `examples/efficiency.py` now does, and the commands
    above are the whole of it.

    Two of the numbers moved for reasons worth knowing rather than drift. The
    thread-parallelism row is a *reasoning* model now, whose long-tailed latency
    costs overlap — which is the row below it, restated. The whole-run rows are
    **end-to-end at default settings** rather than the rollout stage in
    isolation, and the ceiling there is the gate; see
    [Efficiency](efficiency.md#where-the-parallelism-actually-goes).

`n_workers` buys rollout parallelism and `eval_concurrency` buys gate parallelism;
they are independent, and a run slower than its worker count suggests usually
wants the second.

## Equal budget: merge-of-N against best-of-N fork

!!! danger "Every number above is a *throughput* number, and throughput cannot settle this"
    Speedup measures how fast the same work finishes. It cannot tell **merging**
    from **sampling and selecting**, because population-based methods are already
    parallel: N independent forks saturate N workers just as well, and their
    speedup is also close to N. A table of speedups is consistent with merging
    being a new mechanism *and* with it being an engineering convenience.

    One quantity distinguishes them: held-out quality at **equal rollout budget**,
    merge-of-N against best-of-N fork. `agentdescent.baselines` runs the three
    arms that produce it — `serial`, `best_of_n_fork`, `merge_of_n` — over one
    `Workload`, so the arms cannot drift in anything but execution shape.

```python
from agentdescent.baselines import Budget, Workload, best_of_n_fork, compare, merge_of_n, serial, to_markdown

workload = Workload(tasks=tasks, reward=reward, test_eval=score_on_test,
                    agent=agent, evolve_kwargs={"rounds": 10_000})
budget = Budget(rollouts=800)
arms = [f(workload, budget=budget, seed=s) for s in (0, 1, 2)
        for f in (lambda w, **k: serial(w, **k),
                  lambda w, **k: best_of_n_fork(w, 8, **k),
                  lambda w, **k: merge_of_n(w, 8, **k))]
print(to_markdown(compare(arms, fixed="rollouts")))
```

Two properties the module enforces rather than describes:

**Fork is reported twice.** The *oracle* fork is the best fork on test — an upper
bound nobody can ship, since picking it needs the answer. The *selected* fork is
the best on dev, reported on test, which is what fork-and-select actually
delivers. Reporting only one flatters one side.

**Rollouts and calls cannot both be equalised.** Measured, not assumed. Forks that
never talk to each other each start from nothing, so nearly every rollout of
theirs fails and asks for a proposal; a merge arm shares what the others learned,
so more of its rollouts solve outright and never call the proposer. Fix rollouts
and the fork arm spends over twice the model; fix calls and it gets a quarter of
the rollouts. `compare(fixed=...)` therefore names the unit held fixed and prints
the other one's divergence as a confound beside the result. **A merge arm that
wins at equal rollouts while spending more calls has not been shown to win.**

`bench.baselines_run` is the same thing as a command, on the GEPA and ACE
datasets. Print the plan first — a fork arm is N runs by itself, so the run count
is not `arms × seeds`:

```bash
python -m bench.baselines_run --dataset hotpotqa --budget-rollouts 96 --width 4 --seeds 0,1,2 --plan
```

```bash
python -m bench.baselines_run --dataset hotpotqa --budget-rollouts 96 --width 4 --seeds 0,1,2 --provider claude --model GLM-5.2 --yes --json equal-budget.json
```

It runs the **engine's default aggregator**, not GEPA's Pareto selection or ACE's
grow-and-refine: those are search strategies, and running them would leave the
comparison unable to say whether a difference came from merging or from the
search. The numbers are therefore not comparable with those ports' own results.

!!! danger "Check `contested` before reading any merge-vs-fork row"
    A strategy that keeps the whole artifact in **one key** — GEPA's
    `InstructionSlot`, and therefore the HotpotQA workload — makes every pair of
    worker proposals contradict by construction. Conflict resolution collapses
    them to one, and **the fusion tournament never builds a fused candidate at
    all**. `merge_of_n` there is per-round *best-of-N selection*: a real
    mechanism, and not merging.

    `ArmResult.fusion.contested` counts the tournaments a fused candidate
    actually competed in. Measured on the two shipped artifacts:

    | artifact | keys | contested |
    |---|---|---|
    | GEPA `InstructionSlot` | 1 (`instruction`) | **0** — fusion never runs |
    | ACE playbook | one per bullet | > 0 — fusion runs and can lose |

    So a HotpotQA row belongs under *selection*, and only a multi-key artifact
    can fill the table below. `tests/test_fusion_stats.py` pins both directions.

| arm | dataset | rollouts | test quality (min/med/max) | fork oracle |
|---|---|---|---|---|
| serial | — | — | *not yet measured* | — |
| fork-of-8 | — | — | *not yet measured* | — |
| merge-of-8 | — | — | *not yet measured* | — |

### Measured: per-round selection on HotpotQA — no separation

HotpotQA, `GLM-5.2`, 18 rollouts, N=3, `--fetch 80` (40 train / 20 val / 20 test),
**3 seeds**, `--no-self-verify`. Seed artifact scores **0.600** on test, so there
was real headroom. 2050 calls, 4.3M tokens, 15.4 h of model time.

| arm | seeds | rollouts | calls | test (min/med/max) | fork oracle |
|---|---|---|---|---|---|
| serial | 3 | 18 | 164 | 0.700 / **0.750** / 0.750 | — |
| fork-of-3 | 3 | 18 | 225 | 0.700 / 0.700 / 0.800 | 0.800 |
| merge-of-3 | 3 | 18 | 143 | 0.650 / 0.700 / 0.800 | — |

> merge-of-3 and fork-of-3 **overlap across seeds**: this budget on this dataset
> did not distinguish merging from selecting.

**This row is not about merging, and it says so through `contested`.** GEPA's
`InstructionSlot` holds the whole instruction in one key, so every pair of
proposals contradicts, conflict resolution collapses them to one, and
`fusion.contested == 0` for all nine arms — no fused candidate ever competed.
What was measured is **per-round best-of-N selection** against
fork-and-select-at-the-end, and against one worker.

Three things it does establish:

* **No separation, three seeds.** Every arm's spread overlaps every other's. The
  harness refuses to call that a win in either direction, and it is the outcome
  this page said in advance it would publish.
* **One worker was not beaten.** `serial` has the highest median of the three.
  On this budget, on this workload, neither parallel arm bought anything the
  quality column can see.
* **Fork's selection cost is visible.** Its oracle median is 0.800 against a
  selected median of 0.700 — the 0.100 is what fork-and-select loses by having to
  choose on dev. That gap is the reason both numbers are reported.

The call column is the confound, and here it runs *against* the parallel arms:
fork spent 225 calls against a median of 164 for the same 18 rollouts, and did
not win. merge spent 143 and did not win either.

Reproduce:

```bash
python -m bench.baselines_run --dataset hotpotqa --fetch 80 --budget-rollouts 18 --width 3 --seeds 0,1,2 --no-self-verify --headroom --run-concurrency 3 --fork-concurrency 2 --eval-concurrency 6 --provider claude --model GLM-5.2 --yes
```

??? note "A one-seed pilot, which is a pipeline check and not a row of that table"
    HotpotQA, `GLM-5.2`, 12 rollouts, `--width 3`, `--fetch 24` (12 train / 6 val
    / 6 test), one seed, `--no-self-verify`. 182 calls, 435k tokens, ~35 minutes.

    | arm | rollouts | calls | dev | test | fork oracle |
    |---|---|---|---|---|---|
    | serial | 12 | 39 | 0.833 | 0.500 | — |
    | fork-of-3 | 12 | 72 | 0.833 | 0.167 | 0.500 |
    | merge-of-3 | 12 | 39 | 0.667 | 0.333 | — |

    **Nothing about merging can be read off this**, and the harness says so
    rather than leaving it to the reader: six test tasks make every quality
    number a count out of six, and one seed cannot separate anything at all —
    `Comparison.separates` returns `False` below three seeds by construction.

    Two things it does establish, both about the method rather than the result:

    * **The confound is real and it is large.** At *identical* rollouts the fork
      arm spent **72 calls against 39** — 1.85×. Its three forks each start from
      nothing, so nearly every rollout of theirs fails and asks for a proposal,
      while the merge arm shares what the others learned and more of its rollouts
      solve outright. A table matched only on rollouts would have called that
      equal budget. This was predicted from the mechanism before the run; the run
      measured it.
    * **`dev` is not `test`.** The serial arm's gate reported 0.833 and the split
      nothing ever saw reported 0.500. Reporting `final_reward` as the result —
      which the fork arm additionally *selects* on — would have been reporting a
      training score twice over.

The row above is empty on purpose, and stays empty until it is filled by a real
run on HotpotQA and FiNER-139 at ≥ 3 seeds. `docs/algo-ace.md` records the same
configuration moving 4.8 points between two runs, so a single-seed number here
would not be a result. **A negative outcome is publishable and this page will
publish it:** if merge-of-N lands inside the spread of best-of-N fork at equal
budget, then on those workloads parallel merging is a throughput optimisation and
not a new mechanism, and the claim that "gradients add, diffs do not" implies
anything beyond engineering convenience has to be marked unsupported here — with
the datasets and budget it was tested at.

## Reproducing

```bash
python -m examples.gepa.gepa_prompt_evolution --provider openai --model deepseek-v4-flash \
    --rounds 5 --fetch 40 --yes
```

Every faithful port takes a zero-network `--dry-run` and `--provider openai` for
any OpenAI-compatible endpoint via `OPENAI_BASE_URL` + `OPENAI_API_KEY`. Sample
sizes are deliberately small so a run costs minutes; they
are **not** the papers' full setups, and where a full setup needs heavy
infrastructure (SWE-bench in Docker, gated data) the boundary is stated on the
algorithm's page.
