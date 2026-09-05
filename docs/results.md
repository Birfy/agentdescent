# Measured results

Every number here comes from a real run on the algorithm's own dataset with
**`deepseek-v4-flash`** through `openai_compatible`. Each row gives the settings,
so you can reproduce it.

## The algorithm ports

All nineteen port results live in one place — **[measured results, all
nineteen](self-evolution-examples.md#measured-results-all-nineteen)** — with each
row linked to the page that carries its full setup, its caveats and the run file
it came from. They are not repeated here: a second copy of a number is a copy
that goes stale, and this page's copy did.

What belongs here instead is the thing those rows share and none of them can say
on its own.

!!! danger "A difficulty knob calibrated on one model does not transfer"
    Two of the ports set their own difficulty — ACE's `--top-k`, SkillOpt's
    `--hard` — and an early calibration round measured what happens when the
    model underneath changes and the knob does not. Same configuration, same
    call counts, `deepseek-v4-flash` → `glm-5.2`:

    | | knob | outcome on the second model |
    |---|---|---|
    | DGM | none | identical to the digit — a deterministic surrogate objective |
    | GEPA | none | reproduced; HotpotQA's multi-hop structure is hard regardless |
    | EvoSkill | none | reproduced; a decimal-place *convention* is not something capability guesses |
    | **ACE** | `--top-k` | **no lift at all** — 413 calls spent, one bullet, val flat |
    | **SkillOpt** | `--hard` | **no lift at all** — `select_hard` found 2 hard items in 40, padded the rest with items the model solves, and validation sat at 1.000 from round one |

    The mechanism ran correctly in all five. What changed is that there was
    nothing left to learn, and the two rows that lost their lift are exactly the
    two whose difficulty was a knob rather than a property of the benchmark.
    **Re-calibrate before comparing across models**: at a 5% hard rate, SkillOpt
    needs roughly 240 items per split rather than 40 for `select_hard` to find
    genuinely hard items without padding.

!!! note "Not every port's *before* is a seed measurement"
    GEPA, EvoSkill, SkillOpt and all eleven `MethodPolicy` ports score the seed
    artifact explicitly before evolving it, so their "before" is a true baseline.
    ACE's is the **first round's** held-out measurement, taken *after* that
    round's merge — scoring the seed on its own would buy an extra val sweep of
    real model calls. Read it as "where the run started reporting"; if round 0
    committed, the real lift is slightly larger than the row shows.

## Choosing a setting that can show a lift

An evolution run is only as informative as the gap it is given. A strong model
already scores 0.9–1.0 on several of these benchmarks at their smallest settings,
and there the framework correctly commits nothing — `outcomes()` reports
`below-threshold` rather than accumulating changes against a flat signal.

Two levers set the difficulty:

| lever | where | effect |
|---|---|---|
| the benchmark's own difficulty parameter | ACE `--top-k` / `--pool`, ADAS `--dataset` / `--langs` | ACE at `--top-k 10` scores 1.000 and curates nothing; widening the pool puts the baseline at 0.667 |
| [`select_hard`](dataloader.md#turning-a-saturated-benchmark-into-one-with-headroom-select_hard) (`--hard`) | SkillOpt, ADAS | keeps the items a baseline gets wrong — and needs `--hard-passes > 1`, or it selects the model's unlucky answers rather than hard questions |
| a harder benchmark | the six `MethodPolicy` maths ports | GSM8K is half-solved by a current model; GSM-Hard is the same questions with numbers that do not fit in its head |

!!! warning "A hard subset is a different benchmark"
    SkillOpt's lift is measured on the subset its seed skill fails, not on the
    full split where the same model scores ~0.900. The two are not comparable —
    say which one you used.

## The one-call path

[The dataset quickstart](quickstart-skill.md) on 40 real HotpotQA items, 12 held
out — the snippet from the front page, run as written:

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

    `ArmResult.fusion.contested` counts the merges where a fused candidate was
    built **and ranked** against the singles:

    | artifact | keys | contested | where that comes from |
    |---|---|---|---|
    | GEPA `InstructionSlot` | 1 (`instruction`) | **0** — fusion never runs | measured, the HotpotQA runs below |
    | ACE playbook | one per bullet | > 0 — fusion runs and can lose | `tests/test_fusion_stats.py`, offline, synthetic reward — **not** measured on FiNER |

    The second row is a claim about the artifact's *shape*, which a unit test can
    establish, and not a measurement on ACE's dataset. The header said "Measured
    on the two shipped artifacts" over both.

    Two more things keep `contested` at zero, and reading it without them is how
    a non-event becomes a win rate:

    - **Ranking is off by default.** `evolve(fusion_tournament=True)` turns it on;
      without it the union goes straight to the gate, `contested` is zero by
      construction and `unranked` counts the unions instead.
    - **Survivors that agree.** `fuse_diffs` is `ops.update()`, so N copies of one
      diff "fuse" into that diff. It used to count as contested — nine such
      non-events in one measured run — and is now `nothing_to_fuse`.

    So a HotpotQA row belongs under *selection*, and only a multi-key artifact
    can fill the table below.

| arm | dataset | rollouts | test quality (min/med/max) | fork oracle |
|---|---|---|---|---|
| serial | — | — | *not yet measured* | — |
| fork-of-8 | — | — | *not yet measured* | — |
| merge-of-8 | — | — | *not yet measured* | — |

### Measured: merge-of-N against best-of-N fork, HotpotQA — no separation

The first run in which the merge arm **actually merges**. Earlier attempts on
this dataset recorded `fusion.contested == 0` for every arm: GEPA's
`InstructionSlot` holds the whole instruction in one key, so conflict resolution
collapsed every pair of proposals to one and no fusion was ever built. Those runs
measured per-round *selection*. This one installs
[`reflective_merge`](fusion-policies.md#the-deep-dive-when-a-dictionary-update-cannot-merge)
on the merge arm, so contradicting proposals survive to the merge step and a
model writes their union.

**Setup.** HotpotQA through `GLM-5.2`. 80 rows split **40 train / 20 val (the
gate) / 20 test (no gate ever sees it)**. Budget **9 rollouts**, held fixed in
rollouts, N=3, **3 seeds**, `--no-self-verify`. The engine's default aggregator
throughout — GEPA's own Pareto optimizer is deliberately not used, so a
difference cannot come from it. The seed artifact scores **0.600** on test, so
there was real headroom; `--headroom` checks that before spending anything.

| arm | seeds | rollouts | calls | test (min/med/max) | fork oracle |
|---|---|---|---|---|---|
| serial | 3 | 9 | 93 | 0.650 / 0.700 / 0.850 | — |
| fork-of-3 | 3 | 9 | 132 | 0.700 / 0.700 / 0.750 | 0.750 |
| merge-of-3 | 3 | 9 | **73** | 0.600 / **0.800** / 0.850 | — |

> merge-of-3 and fork-of-3 **overlap across seeds**: this budget on this dataset
> did not distinguish merging from selecting.

53 minutes, 1298 calls, 2.7M tokens, 11.3 h of model time.

#### What it establishes

* **No separation, three seeds.** merge has the highest median *and* the widest
  spread — 0.850 / 0.600 / 0.800, with seed 1 landing at the seed artifact's own
  level. `Comparison.separates` refuses to call that a win, which is what it is
  for. Reporting seed 0 alone would have shown merge beating both arms by 0.100;
  reporting seed 1 alone would have shown it losing to both.
* **The call confound runs in merge's favour here.** merge spent 73 calls against
  a median of 93 and fork's 132 — the highest median at the lowest cost, which is
  the shape this page calls robust. It is not enough on its own: robust requires
  the intervals not to overlap, and they do.
* **Fork pays for choosing on dev.** Oracle median 0.750 against a selected
  median of 0.700. The gap is what fork-and-select loses by having to pick
  without the answer, and is why both numbers are reported.

#### The caveat that bounds all of it

**The union was built twice, in the whole experiment.** `FusionStats` per seed:

| seed | tournaments | union built | only one candidate |
|---|---|---|---|
| 0 | 2 | 1 | 1 |
| 1 | 1 | **0** | 1 |
| 2 | 3 | 1 | 2 |

At 9 rollouts over 3 workers a run is three rounds, and the aggregator fires on a
batch of 4 cards — so most merges saw a single diff and had nothing to merge. On
**seed 1 the mechanism never fired at all**, and that seed is the 0.600 pulling
merge's spread down.

So this is not "model-assisted merging does not help". It is an experiment in
which merging happened twice. Read `unranked` before the quality column, exactly
as `contested` had to be read before it in the runs this one replaces.

Two smaller ones: `fork-of-3` seed 1 hit an `APITimeoutError` during test
scoring and its 0.700 is a retried measurement; 29 of 1298 calls failed and were
absorbed by the engine without ending any arm.

#### Reproduce

```bash
python -m bench.baselines_run --dataset hotpotqa --fetch 80 --budget-rollouts 9 --width 3 --seeds 0,1,2 --no-self-verify --reflective-merge --headroom --run-concurrency 5 --fork-concurrency 3 --eval-concurrency 16 --provider claude --model GLM-5.2 --yes
```

To make the mechanism fire more than twice, raise `--budget-rollouts` so a run
has more rounds, or lower `AggregatorConfig.batch_trigger` so a merge fires on
fewer cards. Both cost model time; neither was affordable at ~38 s a call.

## Merging as a cost lever: serial vs 8-wide, sync and async — GEPA/HotpotQA

The section above asks whether merging changes *quality*. This one measures the
other claim: that merging changes **cost** — a round's N diffs fused into one
candidate mean one admission sweep instead of N, so widening the run makes it
cheaper rather than merely wider.

**Setup.** GEPA's own Pareto optimizer throughout — Algorithm 2 (per-instance
frontier, domination pruning, win-frequency sampling) untouched. What changes is
what enters its pool: with `--reflective-merge` the round's surviving diffs are
combined by [`ReflectiveFusion`](aggregator.md#when-a-dictionary-update-cannot-merge-reflectivefusion)
into **one** candidate before admission (`fuse_diffs` cannot do it: on a one-key
artifact it keeps the last rewrite and silently drops the rest). HotpotQA
`--fetch 80 --val-cap 8`: train 52 / D_pareto 8 / test 20. **Empty seed
instruction** — the tuned default answers ~75% correctly, which starves the loop
of proposals; even empty, GLM-5.2 scores 0.75 on D_pareto, so the lever is
weaker than intended and admissions stay scarce. Budget **16 rollouts pinned on
every arm** (`--budget-rollouts 16 --rounds 9999` — the port's own `rounds=10`
default binds *below* the budget and had silently short-changed the serial arm
10 vs 16). `--eval-concurrency 8`, GLM-5.2, seed 0, **one seed** — cost
mechanics, not a quality claim.

| arm | wall (net of failures) | calls | effective concurrency | pool | test EM |
|---|---|---|---|---|---|
| serial | 1424 s | 97 | 1.31× | 6 | 0.75 |
| sync, N=8 | 1022 s (**1.39×**) | **70** | 1.80× | 3 | 0.60 |
| async, N=8 | **742 s (1.92×)** | 95 | **2.89×** | 3 | 0.65 |

"Net of failures": `Usage.failure_seconds` now meters the model time lost inside
calls that failed — the serial arm hit two ~6-minute hung calls (SDK timeout
600 s) and its raw 2148 s wall carries 724 s of that. Endpoint weather is not a
property of the architecture, so the table subtracts it; raw numbers are in
`bench/results/matrix-gepa-b16-w8-seed0-clean.json` with per-cell transcripts
beside it.

#### What it establishes

- **The merge lever works, and only where admissions are the cost.** Sync N=8
  spends 70 calls against serial's 97 (−28%): six single-candidate admissions
  collapse into two merged ones. Async gives most of that back (95 calls) — its
  merger sweeps on its own cadence, so batches shrink and the fusion has less to
  fuse. Wall-clock and call count pull in opposite directions across the two
  parallel arms.
- **The endpoint was never the ceiling.** Raw `curl` probes against the same
  endpoint: 7.04× effective concurrency at width 8 on reasoning-length requests
  (10.09× at width 16 on short ones, all HTTP 200). The measured ladder — 1.31×
  serial, 1.80× sync, 2.89× async — is the *engine's* structure: the rollout
  chain (solve → propose → self-verify) is three sequential calls per failing
  worker, the sync path adds a round barrier plus serial merge and admission
  phases between rounds, and the async path removes the barrier but keeps the
  chains.
- **Test EM 0.75 / 0.60 / 0.65 across the arms is one to three tasks on a
  20-task split** — inside the noise this model shows on identical
  configurations. One seed; the quality question stays with the three-seed
  section above.

#### A shutdown bug this measurement caught

The async arm first reported **347 s with a pool that never grew past the
seed** — faster than every other arm because it had silently skipped its
merging. `max_rollouts` counts a rollout when it *completes*, so when the
budget trips, up to `n_workers − 1` rollouts are still in flight; the merger
drained the intake once and returned, abandoning whatever landed after. And not
uniformly: a failing rollout runs three sequential calls against a success's
one, so the abandoned set is enriched with exactly the rollouts that produce
evidence — measured, 7 of the run's 8 cards. `async_evolve` now keeps draining
until the workers exit whenever the stop was a *work* budget (bounded by
`max_seconds`, the run's own outer limit); a *time* budget keeps the short
grace, because waiting would overshoot the one thing the caller fixed.
`tests/test_async_early_stop.py` pins completed-rollouts == ingested-cards.

#### Reproduce

```bash
# serial / sync arms
python -m bench.matrix_run --rows gepa --budget 16 --width 8 --seeds 0 \
    --eval-concurrency 8 --provider claude --model GLM-5.2 \
    --json bench/results/matrix.json --yes
# async arm (same flags the runner passes, plus --async)
python -m examples.gepa.gepa_prompt_evolution --yes --seed 0 \
    --budget-rollouts 16 --fetch 80 --val-cap 8 --reflective-merge \
    --seed-instruction "" --rounds 9999 --eval-concurrency 8 \
    --workers 8 --async --max-seconds 3600 --provider claude --model GLM-5.2
```

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
