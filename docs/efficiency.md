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
