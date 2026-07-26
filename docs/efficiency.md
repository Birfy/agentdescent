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
       1       262         131     1.00        1.00
       2       525         262     2.00        1.00
       4      1050         518     3.96        0.99
       8      2088        1035     7.92        0.99
```

**Near-linear scaling** (efficiency ≥ 0.99 through 8 workers). The rollout stage
holds no global lock, so workers overlap freely; contention only appears when
they hit the ledger (kept rare here via a large `async_ratio`). This is the
`O(N / T_iter)` throughput the design targets versus serial RSI's `O(1 / T_iter)`.

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
    sync barrier       1.45s         111         40%
async (no barrier)       0.57s         280         100%

async speedup: 2.53x  (the barrier idles fast workers waiting for the tail every round)
```

The barrier runs at **40% utilization** — 60% of worker-time is spent idling for
the tail — while the async pipeline stays at **100%**, a **2.5× wall-clock
speedup**. This is the async-RL *partial-rollout / no-barrier* result ported to
RSI: with heavy-tailed agentic rollouts, the synchronous barrier is dominated by
its slowest worker every single round.

The trade-off async introduces — staleness — is handled by the per-diff `η` /
rebase machinery and the Full / Guarded / Reflective policies; see
[Concepts §3](concepts.md#3-staleness) and the
[async_ratio sweep](concepts.md#34-async_ratio-roll-flash-the-global-lag-budget).

!!! note
    Experiment 2 uses random latencies, so exact numbers vary run to run, but the
    effect is robust (~2–2.5× and ~40% barrier utilization). The ratio tracks
    `E[max of N] / E[latency]` — the heavier the tail, the larger the async win.
