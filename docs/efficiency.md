# Efficiency experiments

Two things the framework claims to buy you — **parallel throughput** and
**asynchronous tail-hiding** — measured in wall-clock. Real rollouts are
I/O-bound (tool calls, HPC queues, LLM latency), so the experiment injects a
per-rollout latency (`Worker.rollout_latency`) to make the effect observable;
sleeping releases the GIL, so worker threads overlap exactly as separate
processes or hosts would.

```bash
python -m examples.efficiency
```

Source: [`examples/efficiency.py`](https://github.com/Birfy/agentdescent/blob/main/examples/efficiency.py).

---

## Where the parallelism actually goes

Overlap depends almost entirely on your latency *distribution*, not on your worker
count. Measured with a fixed-latency stub backend, so the only variable is the
framework:

```bash
python -m examples.efficiency --only distribution
```

| latency shape | sequential | 8 workers | overlap |
|---|---|---|---|
| uniform | 9.9 s | 5.5 s | **1.8×** |
| moderate spread | 15.1 s | 6.3 s | 2.4× |
| high spread | 33.3 s | 14.9 s | 2.2× |
| heavy tail (a reasoning model) | 35.4 s | 20.6 s | **1.7×** |

!!! warning "This table replaces one that had no script behind it"
    The previous version read 5.9× / 4.8× / 3.3× / 2.4× and there was **no entry
    point in the repository that produced it** — it isolated the rollout stage
    under a setup nobody could re-run, and did not say what latency it used. What
    is above is end-to-end `evolve()` at default settings, and the difference is
    the point: subtract the columns and the *rollout* saving is exactly what eight
    workers should buy (9.9 − 5.5 ≈ the 4.2 s of sleeping that got overlapped).
    The speedup is smaller than that because **the ceiling is whatever in a round
    is not a rollout** — and at default settings that is the gate, which scores
    every candidate the aggregator ranks on the whole held-out set.

    At a 20 ms latency the same table reads 1.9× / 2.1× / 2.0× / 1.3×: a fixed
    ~1.2 s per configuration swamps 0.6 s of sleeping. Neither reading is wrong;
    they answer different questions, and the old table did not say which one it
    was answering.

**Latency variance is the cost, and the round barrier is where you pay it.** The
aggregator is a synchronisation point, so a round lasts as long as its *slowest*
worker — and a reasoning model's latency has a long tail (a short answer and a
2000-token deliberation are the same call). On the re-measured table above the
heavy-tail row overlaps at **1.7×** against 2.2–2.4× for moderate spreads — the
tail is exactly what the barrier cannot hide; a real HotpotQA run measured
2.0×, squarely in this regime.

Removing the barrier is what [`asynchronous=True`](evolution.md#the-barrier-free-runtime-async_evolve)
is for, and it recovers part of it — the async experiment below measures
**2.65×** over the sync barrier on the same heavy-tailed workload — because
workers stop waiting for the merge. On a live model the same comparison is the
[runtime matrix](matrix-report.md): 1.36× end-to-end / 1.89× engine-window
across eleven ports.

### The other axis — `eval_concurrency`

Every gate goes through one held-out evaluation: each round's measurement and, far
more often, the aggregator's per-candidate comparisons. That is a second pool,
independent of `n_workers`. Same work, varying only `eval_concurrency`:

```bash
python -m examples.efficiency --only gate
```

| `eval_concurrency` | wall-clock | |
|---|---|---|
| 1 (serial) | 3.6 s | — |
| 4 | 2.2 s | **1.7× faster** |
| 8 (default) | 1.2 s | **2.9× faster** |
| 16 | 1.6 s | saturated — the held-out set is only 8 tasks |

Also re-measured with a script rather than by hand; the previous row set (193.6 s
→ 90.0 s → 89.0 s) came from a larger workload with no reproducible entry point.
The shape is what carries: **serial gate, then linear, then flat past the size of
the held-out set.**

It saturates once `eval_concurrency` reaches the size of your held-out set, so
raise it if yours is large and your provider allows the concurrency.

!!! tip "Which knob to reach for"
    `n_workers` buys **rollout** parallelism, `eval_concurrency` buys **gate**
    parallelism, and they are independent. If a run feels slower than its worker
    count suggests, the gate is the usual reason — and if it still does after
    that, the barrier is meeting a heavy tail, which is what `asynchronous=True`
    addresses.

### Who was waiting for whom — the stage profile

The table above measures whether the run got *faster*. It cannot say **why**,
and the obvious candidate — `eval_seconds` — cannot either: it sums across the
evaluation pool, so eight threads scoring for a second each reads `8.0` whether
the run spent eight seconds in the gate or one.

Three counters answer it, and they are wall-clock on the one thread that merges:

| counter | what it is |
|---|---|
| `merge_seconds` | the merger's **busy** time. One thread, so `merge_seconds / wallclock` is a real occupancy and cannot exceed 1 |
| `merge_gate_seconds` | the part of it blocked on evaluation. A **subset** — `gate_share()` — never a second total to add |
| `worker_starved_seconds` | summed across workers: time held at the backpressure gate with a finished card and nowhere to put it. Async only |

```bash
python -m examples.efficiency --only stages
```

| workload | wall | merger busy | of it, gate | starved/s |
|---|---:|---:|---:|---:|
| sync, uniform 20 ms | 0.5 s | 59% | **100%** | — |
| async, uniform 20 ms | 4.1 s | 94% | 82% | **4.5×** |
| async, uniform 20 ms, `eval_concurrency=1` | 4.3 s | 99% | 95% | **6.9×** |
| sync, heavy tail | 1.9 s | 43% | **100%** | — |
| async, heavy tail | 4.3 s | 90% | 94% | **5.6×** |

`starved/s` is `worker_starved_seconds / wallclock` at `n_workers=8`, so `4.5×`
means four and a half of the eight workers were blocked at any given moment.

**The gate is on the critical path, and the third row is what proves it.** Same
workload, same rollouts, same evaluations — only the gate's own pool narrows,
and starvation rises from 4.5 to 6.9 of eight workers. Nothing else moved, so
nothing else can be responsible.

On the synchronous path the `of it, gate` column reads **100%**: everything the
driver does after the barrier *is* evaluation. That is not news, but it was not
measurable before — `merge_seconds` had been declared in `metrics.py` since the
first version and written by nobody, so every run ever published reported `0.0`,
which reads as "merging was free".

!!! note "Read the last two columns together"
    A merger at 95% occupancy that starved nobody has hidden itself perfectly
    behind the rollouts, and moving its work elsewhere buys nothing. A merger at
    95% whose workers idle on it is the serial-stage bottleneck — FlashEvolve's
    Figure 2(c), which profiles the evaluate stage at 56–92% of a synchronous
    step.

    The counters also show a trap: with this domain's *microsecond* rollout the
    workers lap the merger between sweeps whatever the gate costs (measured at
    `async_ratio=8`: four workers starved for ~4 s of a 1 s window with nothing
    slowed down at all). Starvation alone does not implicate the gate.
    `gate_share()` is what does.

### Taking the gate off the merger — `pipelined_gate`

A merge is three phases, and only the middle one is expensive:

| phase | cost | touches aggregator state |
|---|---|---|
| **prepare** — drain, staleness, conflicts, fusion | cheap | yes |
| **measure** — score base and candidate | **94% of gate time** | **no** |
| **decide** — accept, audit, CAS commit | cheap | yes |

That middle column is why the split is possible: `Aggregator.measure()` writes
only into the candidate it was handed, so it can run anywhere.
`async_evolve(pipelined_gate=True)` runs it on its own threads and lets the
merger go back to draining.

**It changes no commit semantics.** At most one candidate per artifact is in
flight, so every candidate is still committed against the head it was prepared
and measured on — there is no candidate-level staleness to have a policy about.
Cards arriving meanwhile accumulate in the aggregator's buffer, so batches get
larger rather than more numerous.

Measured at `n_workers=8`, `async_ratio=3`, held-out scoring costing 5× a
rollout — the regime FlashEvolve profiles:

| | rollouts | merges | merger busy | starved |
|---|---:|---:|---:|---:|
| inline gate | 648 | 10 | 53% | 4.40× |
| `pipelined_gate=True` | **921** | **17** | 34% | **2.83×** |

**+42% rollouts and +70% merges for the same wall-clock**, with a third fewer
workers blocked. When the gate is only as expensive as a rollout the same
comparison reads 776 → 818 (+5%): the win is proportional to how much the gate
dominates, which is the honest shape for it to have.

!!! warning "The first version of this made the run *slower*, and the counters said why"
    Skipping the merger's poll sleep whenever a measurement was *in flight*
    turned the merger into a busy-wait — and a busy-wait holds the GIL against
    the very workers it was meant to free. Measured: 462 rollouts inline against
    **389** pipelined, while `merger busy` read a confident 92%, because
    spinning is occupancy. The condition is `fut.done()`, not "anything
    pending".

    Off by default. It is a third pool — `n_workers` rollouts, `gate_workers`
    measurements, each fanning out over `eval_concurrency` tasks — so the
    ceiling your provider sees is `n_workers + gate_workers × eval_concurrency`.


## The configuration matrix — `bench/`

```bash
python -m bench.run --config sync-1 --config sync-4 --config sync-8                     --config async-4 --seeds 0,1,2
```

Fixed data, fixed actors, fixed semantics; only the configuration varies. Three
seeds, reported as the spread that was observed rather than a point estimate —
this repository has published one that moved 4.8 points between two runs of one
configuration. Quality is scored on a split `evolve()`'s gate never saw.

| config | seeds | reached | time-to-quality (min/med/max) | cost-to-quality | test quality | stale% |
|---|---|---|---|---|---|---|
| sync-1 | 3 | 2/3 | 1.22 / 1.26 / 1.29 | 21 / 22 / 22 | 0.793 / 0.862 / 0.897 | 0% |
| sync-4 | 3 | 3/3 | 0.46 / 0.52 / 0.71 | 24 / 24 / 40 | 0.862 / 1.000 / 1.000 | 0% |
| sync-8 | 3 | 3/3 | 0.25 / 0.25 / 0.41 | 24 / 24 / 40 | 0.931 / 1.000 / 1.000 | 0% |
| async-4 (lag 3, the default) | 3 | **0/3** | — | — | 0.310 / 0.345 / 0.379 | **93%** |
| async-4 (lag 1) | 3 | 3/3 | 0.56 / 0.74 / 0.93 | 31 / 35 / 123 | 0.897 / 0.931 / 1.000 | 25% |
| async-4 (lag 0) | 3 | 3/3 | 0.59 / 0.66 / 0.75 | 25 / 28 / 46 | 0.828 / 0.897 / 0.931 | 0% |

!!! warning "The `stale%` column was understated, and this is the corrected run"
    It read **86%** and **10%** for the two async rows. The async path ran its own
    staleness gate and `Aggregator` ran another over the survivors, both writing
    to the same meter, so every card that survived the first gate was counted as
    "considered" twice — a true 50% rate read as 33%. With the denominator fixed
    the same configurations report **93%** and **25%**.

    Nothing about the runs changed; the numerator was always right. This is why
    the row that never reaches the bar is the one whose figure moved least: at
    93% there is not much room for a factor of two.

### What it says

**Parallelism buys time, not rollouts.** Time-to-quality falls 1.26 → 0.52 → 0.25
from 1 to 4 to 8 workers, while cost-to-quality *rises* slightly (22 → 24). More
workers reach the bar sooner and more reliably (2/3 → 3/3), and spend marginally
more rollouts doing it. Anyone hoping parallelism reduces total work should read
the second column.

**The barrier-free path's default lag budget is wrong for this domain, and the
first version of this table blamed the path.** At `async_ratio=3` it discards 93%
of its evidence and never reaches the bar. At 1 it comes within a point of the
synchronous path on quality; at 0 it is the steadiest configuration in the table,
reaching the bar on every seed.

`async_ratio` is a lag budget in **artifact versions**, and how much wall-clock a
version represents depends entirely on how long a rollout takes. Three is
sensible when a rollout is a model call taking seconds. Here a rollout is a
dictionary lookup, so a worker drifts three versions behind almost immediately
and stays there.

The default has not been changed — it is tuned for the workload this framework is
for, not for the one that is cheap to measure. Instead, a run that discards more
than half its evidence now says so:

```
RuntimeWarning: async_evolve discarded 121/130 (93%) of its evidence as stale.
async_ratio=3 is a lag budget in artifact versions, so it is too high whenever a
worker finishes several rollouts in the time the merger takes one sweep.
```

**Even correctly tuned, async does not beat sync here** — 0.74 against 0.52. Its
advantage is that workers never wait for the merge, and on a domain with no
rollout latency there is no waiting to avoid. That is the caveat below, arrived
at from the other direction.

!!! danger "These numbers are not an answer to the parallelism question"
    **No model was called.** The router domain's rollout is a dictionary lookup,
    so a rollout costs microseconds where a real one costs seconds. Parallelism
    exists to hide rollout latency; a benchmark with no latency to hide measures
    the cost of coordination and none of what it buys.

    Read this table as evidence that the *harness* works — deterministic, budget
    matched in calls, quality on an unseen split, spread rather than point
    estimates. The real-model answer to
    [#52](https://github.com/Birfy/agentdescent/issues/52)'s question now
    exists: the [runtime matrix](matrix-report.md) runs eleven ports against a
    live model under serial/sync/async scheduling — 1.36× end-to-end, 1.89×
    engine-window (n=33). This table and that one measure different substrates
    (coordination cost here, latency hiding there); read them together.

### What the matrix does not vary yet

`Config` carries `executor=` and `sandbox=` fields, and `bench.run`'s workload
reads neither: every row is the in-process default on a local workspace. They are
there because the matrix is the right place for those dimensions, not because
they have been measured — and a row for "processes" produced by ignoring the
field would be the worst kind of number in this table.

For the same reason the `fingerprint` and `env_mismatch` columns are populated by
`harness.run_config` but not by the `bench.run` entry point, which runs one
environment and says so by leaving them empty. The ⚠︎ marker that flags a
mixed-environment row is machinery waiting for a comparison that has more than
one environment in it.

### One thing the harness found about the engine

The async path filters staleness inline and never reaches `Aggregator`'s filter,
which is where the synchronous ratio is counted. Its stale column read **0%** —
not "no staleness" but "not measured", and the two are indistinguishable in a
table. Counted at the inline gate, and with the double-counted denominator fixed,
it reads **93%**, which is the explanation for the row above it.


## On a live model, on a published algorithm

Everything above is the engine measured on stub or synthetic latency. This is the
same question asked of a faithful port of somebody else's serial algorithm, on a
real dataset, against a real endpoint: **GEPA on HotpotQA, `deepseek-v4-flash`,
16 rollouts pinned on every arm.**

```bash
python -m bench.matrix_run --rows gepa --budget 16 --width 4 --seeds 0 \
    --provider claude --model deepseek-v4-flash --yes
```

The number reported here is **concurrency: model seconds ÷ wall-clock**, or how
much of the run was genuinely in flight at once. It is the quantity a worker pool
is responsible for, and the only one that is bounded by the pool's width.

| arm | wall | model seconds | rollouts | concurrency |
|---|---:|---:|---:|---:|
| serial (upstream loop) | 609 s | 607 | 16 | 1.00× |
| sync parallel, N=4 | 239 s | 443 | 16 | **1.85×** |
| async, N=4, lag 3 | 140 s | 451 | 19 | **3.22×** |

The control is the published serial loop and nothing else: one worker, and
`eval_concurrency=1`, so the gate scores one task at a time the way the original
does. That matters more than it sounds — `--serial` on its own lowers only
`n_workers`, and a control that still evaluates concurrently is already partly
parallel. The row above is the first one measured here that is not: 607 model
seconds inside a 609 second wall-clock, an overlap of exactly 1.00×.

**The barrier is the whole difference between the two parallel rows.** Both run
four workers. The synchronous arm reaches 1.85× because the round boundary keeps
the rollout pool and the gate pool from ever being busy at the same time —
consistent with the 1.7–1.8× the heavy-tail stub measures further up this page.
Removing the barrier lets them overlap, and the same four workers reach 3.22×.

!!! note "Why the wall-clock falls faster than the concurrency rises"
    239 s and 140 s are less than 609 s divided by 1.85 and 3.22. The remainder
    is work that did not happen rather than work done in parallel: model seconds
    per rollout fall 37.9 → 27.7 → 23.8 across the three arms, as
    `--reflective-merge` collapses each round's diffs into one gate sweep, and as
    the async arm **discards 4 of 4 stale proposals** — a proposal thrown away
    before it reaches the gate costs nothing to score.

    That saving is real but it is not parallelism, so it is not in the column
    above. The stale rate is the one to watch: at `async_ratio=3` this workload
    discards everything the staleness filter sees, which buys wall-clock and
    spends proposals.

The async arm also overran the pinned budget — 19 rollouts against 16 — because
the barrier-free path has no round boundary at which to stop.

## Threads and the GIL — is this *really* parallel?

Yes, for this workload, and no amount of arguing about the GIL settles it — so
here it is measured. Eight threads, one pool, two workloads: a real API round
trip, and pure-Python arithmetic.

```bash
python -m examples.efficiency --only gil --model glm-5.2
```

| workload | sequential | 8 threads | speedup |
|---|---|---|---|
| **I/O** — a real `glm-5.2` call | 48.6 s | **8.3 s** | **5.8×** |
| **CPU** — pure Python arithmetic | 2.3 s | 2.1 s | 1.1× |

!!! note "Measured on `glm-5.2`, not on the model the old row named"
    The previous row read **7.1×** against `deepseek-v4-flash`. This one is a
    *reasoning* model, and the gap is the table two sections up restated: eight
    threads finish when the slowest finishes, so a long-tailed latency costs
    overlap. Across three runs it landed at 5.8× / 6.3× / 6.5×.

    The CPU row is 1.1× rather than 1.0× because the work was too small to
    measure at first — 25 ms a unit, where the answer is whatever the scheduler
    did that second. It is sized to take seconds now, and 1.1× is what is left:
    noise around "threads buy nothing here".

Near-linear on I/O, exactly nothing on CPU. CPython releases the GIL around
socket I/O and holds it around bytecode, and a rollout is almost entirely spent
waiting on the model — so threads are the right primitive here and **you do not
need multiple processes**.

The corollary matters more than the headline: the speedup tracks how much of
your rollout is *waiting*. Threads buy you nothing for an agent that burns CPU
locally — a local model in-process, heavy parsing, big numeric work. For that,
put the CPU work behind a process pool or a separate service, and keep the
framework's workers on the I/O.

---

## Experiment 1 — parallel throughput scaling

Run the async runtime with N = 1, 2, 4, 8 workers for a fixed wall-clock window
and count rollouts. Throughput (rollouts/sec) should scale with N; **efficiency
= speedup / N** shows how close to linear it stays.

```
 workers  rollouts  rollouts/s  speedup  efficiency
       1       271         136     1.00        1.00
       2       467         234     1.72        0.86
       4      1082         541     3.99        1.00
       8      2172        1086     8.01        1.00
```

**Near-linear scaling through 8 workers.** The rollout stage holds no global lock,
so workers overlap freely; contention only appears when they hit the ledger (rare
here via a large `async_ratio`, and much cheaper since ledger reads stopped
forking a `git checkout`). This is the `O(N / T_iter)` throughput the design
targets versus serial RSI's `O(1 / T_iter)`.

!!! note "Read efficiency as ≈1.0, not as a precise constant"
    Across five repeated runs the 8-worker figure landed between **7.83x and
    9.15x** (efficiency 0.98–1.14) and the 4-worker one between 3.99x and 4.46x.
    The spread is dominated by the **single-worker baseline**, which varied 15%
    run to run (120–136 rollouts/s) and sits in the denominator of every other
    row. The 2-worker row is consistently the weakest (0.86–1.01), which is where
    a fixed per-run cost still shows.

    The honest reading is "linear to within measurement noise at this scale", and
    the absolute rollout counts depend on the machine — **rerun it rather than
    quoting these**.

!!! warning "Two measurement definitions were wrong here, and both flattered it"
    Found while re-measuring on the ported runtime, and worth stating because
    each inflated the headline:

    * **The denominator included setup and the shutdown grace.** The rate was
      `rollouts / measured wallclock`, and the measured clock covered building the
      ledger, verifier and aggregator plus up to `shutdown_grace` seconds after
      the window. Those are *fixed* costs, so they fall hardest on the low-worker
      rows — which reads as superlinear speedup. Measured both ways: 8 workers
      came out at **8.2–9.4x** against ~8.1x. The experiment says "a fixed
      wall-clock window", so it now divides by the window it asked for.
    * **`self_verify` doubled what a counted rollout cost.** The engine re-runs a
      proposal's own rollout to record a before/after delta, and only the first
      is counted — so with a 6 ms latency injected, every counted rollout paid
      12 ms. The reference loop got that delta free. Throughput halved (935 →
      466 rollouts/s) with the speedup unchanged, which is the signature of a
      cost-model change rather than a scaling one. This experiment measures
      dispatch, so it now passes `self_verify=False`.

---

## Experiment 2 — async pipeline vs synchronous barrier

Isolates the *scheduling discipline* under a **heavy-tailed** rollout latency
(4 ms base, 12× spike 15% of the time). The same fixed rollout budget is run two
ways:

- **sync barrier** — each round of N rollouts must wait for the *slowest* before
  the next round starts. Wall-clock per round = `E[max of N]`.
- **async (no barrier)** — workers never wait for each other. Wall-clock =
  `E[latency]` per rollout.

```
            mode  wall-clock  rollouts/s  utilization
    sync barrier       1.41s         113         38%
async (no barrier)       0.53s         300        100%

async speedup: 2.65x  (the barrier idles fast workers waiting for the tail every round)
```

The barrier runs at **~36–40% utilization** — most worker-time is spent idling for
the tail — while the async pipeline stays at **100%**, a **~2.6–2.9× wall-clock
speedup**. This is the async-RL *partial-rollout / no-barrier* result ported to
RSI: with heavy-tailed agentic rollouts, the synchronous barrier is dominated by
its slowest worker every single round.

The trade-off async introduces — staleness — is handled by the per-diff `η` /
rebase machinery and the Full / Guarded / Reflective policies; see
[Concepts §3](concepts.md#3-staleness) and the
[async_ratio sweep](concepts.md#34-async_ratio-roll-flash-the-global-lag-budget).

!!! note
    Experiment 2 uses random latencies, so exact numbers vary run to run, but the
    effect is robust (measured 2.57–2.93x across runs, ~36–40% barrier
    utilization). The ratio tracks
    `E[max of N] / E[latency]` — the heavier the tail, the larger the async win.
