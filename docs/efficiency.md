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

| what changes | overlap with `n_workers=8` |
|---|---|
| uniform latency | **5.9×** |
| moderate latency spread | 4.8× |
| high spread | 3.3× |
| heavy tail (a reasoning model) | **2.4×** |

**Latency variance is the cost, and the round barrier is where you pay it.** The
aggregator is a synchronisation point, so a round lasts as long as its *slowest*
worker — and a reasoning model's latency has a long tail (a short answer and a
2000-token deliberation are the same call). 2.4× is what a barrier costs on a
heavy-tailed distribution; a real HotpotQA run measured 2.0×, squarely in this
regime.

Removing the barrier is what [`asynchronous=True`](evolution.md#the-barrier-free-runtime-async_evolve)
is for, and it recovers part of it — **2.4× → 3.0×** on the same heavy-tailed
workload — because workers stop waiting for the merge.

### The other axis — `eval_concurrency`

Every gate goes through one held-out evaluation: each round's measurement and, far
more often, the aggregator's per-candidate comparisons. That is a second pool,
independent of `n_workers`. Same work, varying only `eval_concurrency`:

| `eval_concurrency` | wall-clock | |
|---|---|---|
| 1 (serial) | 193.6 s | — |
| 4 | 96.7 s | **2.0× faster** |
| 8 (default) | 90.0 s | **2.2× faster** |
| 16 | 89.0 s | saturated — the held-out set is only 12 tasks |

It saturates once `eval_concurrency` reaches the size of your held-out set, so
raise it if yours is large and your provider allows the concurrency.

!!! tip "Which knob to reach for"
    `n_workers` buys **rollout** parallelism, `eval_concurrency` buys **gate**
    parallelism, and they are independent. If a run feels slower than its worker
    count suggests, the gate is the usual reason — and if it still does after
    that, the barrier is meeting a heavy tail, which is what `asynchronous=True`
    addresses.


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
| sync-1 | 3 | 2/3 | 1.06 / 1.10 / 1.13 | 21 / 22 / 22 | 0.793 / 0.862 / 0.897 | 0% |
| sync-4 | 3 | 3/3 | 0.42 / 0.50 / 0.70 | 24 / 24 / 40 | 0.862 / 1.000 / 1.000 | 0% |
| sync-8 | 3 | 3/3 | 0.25 / 0.26 / 0.42 | 24 / 24 / 40 | 0.931 / 1.000 / 1.000 | 0% |
| async-4 | 3 | **0/3** | — | — | 0.379 / 0.414 / 0.448 | **83%** |

### What it says

**Parallelism buys time, not rollouts.** Time-to-quality falls 1.10 → 0.50 →
0.26 from 1 to 4 to 8 workers, while cost-to-quality *rises* slightly (22 → 24).
More workers reach the bar sooner and more reliably (2/3 → 3/3), and they spend
marginally more rollouts doing it. Anyone hoping parallelism reduces total work
should read the second column.

**The barrier-free path is worse here, and the reason is in the table.** 83% of
its evidence is stale. Its whole advantage is that workers never wait for the
merge — and on this domain a rollout is instant string matching, so there is no
waiting to avoid. It pays the coordination cost and collects none of the benefit.

That is a property of the domain, not a verdict on the async path. It is also the
sharpest possible illustration of the caveat below.

!!! danger "These numbers are not an answer to the parallelism question"
    **No model was called.** The router domain's rollout is a dictionary lookup,
    so a rollout costs microseconds where a real one costs seconds. Parallelism
    exists to hide rollout latency; a benchmark with no latency to hide measures
    the cost of coordination and none of what it buys.

    Read this table as evidence that the *harness* works — deterministic, budget
    matched in calls, quality on an unseen split, spread rather than point
    estimates. Answering [#52](https://github.com/Birfy/agentdescent/issues/52)'s
    question needs a real port against a real model, which is the step that has
    not been run.

### One thing the harness found about the engine

The async path filters staleness inline and never reaches `Aggregator`'s filter,
which is where the synchronous ratio is counted. Its stale column read **0%** —
not "no staleness" but "not measured", and the two are indistinguishable in a
table. Now counted at the inline gate, it reads 83%, which is the explanation for
the row above it.


## Threads and the GIL — is this *really* parallel?

Yes, for this workload, and no amount of arguing about the GIL settles it — so
here it is measured. Eight threads, one pool, two workloads: a real API round
trip, and pure-Python arithmetic.

| workload | sequential | 8 threads | speedup |
|---|---|---|---|
| **I/O** — a real `deepseek-v4-flash` call | 14.9 s | **2.1 s** | **7.1×** |
| **CPU** — pure Python arithmetic | 1.0 s | 1.0 s | 1.0× |

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
       1       230         115     1.00        1.00
       2       465         231     2.02        1.01
       4       940         466     4.07        1.02
       8      1880         935     8.09        1.01
```

**Near-linear scaling through 8 workers.** The rollout stage holds no global lock,
so workers overlap freely; contention only appears when they hit the ledger (rare
here via a large `async_ratio`, and much cheaper since ledger reads stopped
forking a `git checkout`). This is the `O(N / T_iter)` throughput the design
targets versus serial RSI's `O(1 / T_iter)`.

!!! note "Read efficiency as ≈1.0, not as a precise constant"
    Across repeated runs the 8-worker figure lands between **8.05x and 8.16x**
    (efficiency 1.01–1.02), and the 4-worker one between 3.96x and 4.13x. Values
    slightly *above* 1.0 are not a superlinear effect: the single-worker baseline
    absorbs the same fixed start-up inside its timed window, which depresses the
    denominator by a percent or two. The honest reading is "linear to within
    measurement noise at this scale", and the absolute rollout counts depend on the
    machine — rerun it rather than quoting these.

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
    sync barrier       1.41s         114         36%
async (no barrier)       0.51s         316        100%

async speedup: 2.78x  (the barrier idles fast workers waiting for the tail every round)
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
