"""Experiments: parallel efficiency and asynchronous efficiency.

Two things the framework claims to buy you, measured in wall-clock on the
deterministic router domain with an injected per-rollout latency (models the
real cost of a rollout -- tool calls, HPC queues, LLM latency -- which is where
parallelism and asynchrony actually pay off).

    python -m examples.efficiency

**Experiment 1 -- parallel throughput scaling.** Run the async runtime with
N = 1, 2, 4, 8 workers for a fixed wall-clock and count rollouts. Throughput
(rollouts/sec) should scale with N; speedup = rate(N)/rate(1), efficiency =
speedup/N shows how close to linear it stays before lock contention bites.

**Experiment 2 -- async vs synchronous barrier, under a latency tail.** Isolates
the rollout-scheduling discipline: run the *same* fixed rollout budget two ways
-- (a) a synchronous round barrier where each round of N rollouts must wait for
the *slowest* before the next round starts, vs (b) barrier-free workers that
never wait for each other. With heavy-tailed latency the barrier pays
``E[max of N]`` per round while async pays ``E[latency]``; the ratio is the
tail-hiding speedup (the async-RL "partial rollout / no barrier" claim).
"""

from __future__ import annotations

import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agentdescent.async_runtime import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe
from agentdescent.ledger import Ledger


def constant_latency(seconds: float):
    return lambda: seconds


def heavy_tailed_latency(base: float, spike: float, p_spike: float, seed: int):
    rng = random.Random(seed)
    lock = __import__("threading").Lock()

    def lat():
        with lock:
            hit = rng.random() < p_spike
        return base * (spike if hit else 1.0)
    return lat


def _slow_rollout(latency):
    """The domain's rollout, plus the latency a real one would have.

    `time.sleep` releases the GIL, which is the whole point: it stands in for the
    network wait an agent rollout actually is, so the overlap being measured here
    is the overlap a real workload gets."""
    from agentdescent.domains.router import router_run

    def rollout(rendered, task):
        time.sleep(latency())
        return router_run(rendered, task)
    return rollout


def _build(universe, n_workers, latency, seed, **cfg_kw):
    cfg = AsyncConfig(n_workers=n_workers, noise=0.1, seed=seed, **cfg_kw)
    repo = tempfile.mkdtemp(prefix="agentdescent-eff-")
    return AsyncAgentDescent(repo, universe, config=cfg,
                             rollout=_slow_rollout(latency))


# -- Experiment 1: parallel throughput scaling -------------------------------


def experiment_parallel(universe, seconds=2.0):
    print("=== Experiment 1: parallel throughput scaling (async runtime) ===")
    print(f"per-rollout latency = 6ms, wall-clock window = {seconds}s\n")
    print(f"{'workers':>8} {'rollouts':>9} {'rollouts/s':>11} {'speedup':>8} {'efficiency':>11}")
    base_rate = None
    for n in (1, 2, 4, 8):
        # target unreachable -> run the full window; huge async_ratio + stall
        # patience keep workers off the ledger so we measure pure rollout rate.
        # `self_verify=False`: this counts dispatch, and a self-verify rollout
        # is a second sleep the counter does not see. The reference loop got its
        # before/after delta free, so leaving it on would halve a number that is
        # supposed to be comparable with the published table.
        sys = _build(universe, n, constant_latency(0.006), seed=1,
                     async_ratio=10_000, stall_patience=10_000, self_verify=False,
                     target_accuracy=2.0, max_seconds=seconds)
        stats = sys.run()
        # The window the experiment asked for, not a measured wall-clock. This
        # is what the description says ("a fixed wall-clock window"), and a
        # measured one also counts setup and the shutdown grace -- fixed costs
        # that fall hardest on the low-worker rows and read as superlinear
        # speedup. Measured both ways: 8 workers came out at 8.2-9.4x against
        # ~8.1x here.
        rate = stats.rollouts / seconds
        if base_rate is None:
            base_rate = rate
        speedup = rate / base_rate
        print(f"{n:>8} {stats.rollouts:>9} {rate:>11.0f} {speedup:>8.2f} "
              f"{speedup / n:>11.2f}")
    print()


# -- Experiment 2: async vs synchronous barrier ------------------------------


def run_barrier(rollout, rendered, task, n_workers, rounds):
    """Barrier: each round runs N rollouts concurrently, then waits for the
    slowest before the next round -- wall-clock = sum of per-round max latency."""
    start = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for _ in range(rounds):
            futs = [ex.submit(rollout, rendered, task) for _ in range(n_workers)]
            for f in futs:              # <-- barrier: block on the slowest worker
                f.result()
    return time.time() - start


def run_free(rollout, rendered, task, n_workers, rounds):
    """No barrier: each worker runs `rounds` rollouts back-to-back, never
    waiting for the others -- wall-clock = the busiest worker's own total."""
    def loop():
        for _ in range(rounds):
            rollout(rendered, task)
    threads = [threading.Thread(target=loop) for _ in range(n_workers)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.time() - start


def experiment_async(universe, n_workers=4, rounds=40):
    print("=== Experiment 2: async (no barrier) vs synchronous barrier ===")
    print(f"N = {n_workers} workers, heavy-tailed latency (4ms base, 12x spike @15%), "
          f"{rounds} rounds = {rounds * n_workers} rollouts each way\n")
    # Dispatch shape only: one rendered artifact, one cluster, N concurrent
    # rollouts. Nothing here touches the ledger, which is the point -- the
    # question is what the *barrier* costs, not what a merge costs.
    from agentdescent.domains.router import RouterStrategy, cluster_tasks

    rollout = _slow_rollout(heavy_tailed_latency(0.004, 12.0, 0.15, seed=7))
    rendered = RouterStrategy().render({})
    task = cluster_tasks(universe, n_clusters=max(2, n_workers))[0][0]
    total = rounds * n_workers

    bt = run_barrier(rollout, rendered, task, n_workers, rounds)
    ft = run_free(rollout, rendered, task, n_workers, rounds)

    print(f"{'mode':>16} {'wall-clock':>11} {'rollouts/s':>11} {'utilization':>12}")
    print(f"{'sync barrier':>16} {bt:>10.2f}s {total / bt:>11.0f} {ft / bt:>11.0%}")
    print(f"{'async (no barrier)':>16} {ft:>10.2f}s {total / ft:>11.0f} {'100%':>12}")
    print(f"\nasync speedup: {bt / ft:.2f}x  "
          f"(the barrier idles fast workers waiting for the tail every round)")
    print()


# -- Experiment 3: where the overlap goes (latency distribution) ---------------


def _stub_workload(latency, n_tasks=20):
    """A minimal `evolve()` workload whose only cost is waiting.

    Deterministic, no model, no ledger contention worth speaking of: the rollout
    sleeps and returns. That is the point -- the *only* variable across the rows
    below is the shape of the latency distribution, so what the table measures is
    the framework rather than a provider.
    """
    from agentdescent import Task

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)})
             for i in range(n_tasks)]

    def run(rendered, task):
        time.sleep(latency())
        return task.meta["gold"] if task.id in rendered else "?"

    return tasks, run


def _evolve_result(tasks, run, *, concurrency, rounds=6, workers=8, eval_conc=8):
    """Run the stub workload and hand back the whole result, not just a duration.

    `_timed_evolve` threw everything but the wall-clock away, which was fine
    while wall-clock was the only question. The stage profile below needs the
    cost counters off the same run."""
    import warnings as _w

    from agentdescent import AppendRules, evolve

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return evolve(tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                      run=run, propose=lambda r, t, o, s: t.id,
                      strategy=AppendRules(),
                      rounds=rounds, n_workers=workers, max_concurrency=concurrency,
                      eval_concurrency=eval_conc, held_out_frac=0.4,
                      self_verify=False, seed=0)


def _timed_evolve(tasks, run, *, concurrency, rounds=6, workers=8, eval_conc=8):
    start = time.time()
    _evolve_result(tasks, run, concurrency=concurrency, rounds=rounds,
                   workers=workers, eval_conc=eval_conc)
    return time.time() - start


def experiment_distribution(rounds: int = 4) -> None:
    """Overlap depends on the latency *distribution*, not on the worker count.

    Same eight workers, same rollout budget, four shapes of latency. A round is a
    barrier, so it lasts as long as its slowest worker -- which is why a heavy
    tail costs more than a high mean.
    """
    print("=== Experiment 3: where the overlap goes ===")
    print(f"n_workers=8, {rounds} rounds, sequential vs concurrent")
    print("rollout latency ~150ms, chosen so the rollout dominates the round --")
    print("see the note below the table for what happens when it does not.\n")
    print(f"{'latency shape':>26} {'sequential':>11} {'8 workers':>10} {'overlap':>8}")
    shapes = (
        ("uniform", constant_latency(0.15)),
        ("moderate spread", heavy_tailed_latency(0.15, 2.0, 0.30, seed=1)),
        ("high spread", heavy_tailed_latency(0.15, 5.0, 0.30, seed=2)),
        ("heavy tail (reasoning)", heavy_tailed_latency(0.075, 20.0, 0.15, seed=3)),
    )
    for name, latency in shapes:
        tasks, run = _stub_workload(latency)
        seq = _timed_evolve(tasks, run, concurrency=1, rounds=rounds)
        par = _timed_evolve(tasks, run, concurrency=8, rounds=rounds)
        print(f"{name:>26} {seq:>10.1f}s {par:>9.1f}s {seq / par:>7.1f}x")
    print("\nThe ceiling is set by whatever in a round is *not* a rollout -- the")
    print("ledger, and the gate. At a 20ms latency the same table reads 1.9x /")
    print("2.1x / 2.0x / 1.3x, because a fixed ~1.2s per configuration swamps")
    print("0.6s of sleeping. That is experiment 4's subject, not a smaller")
    print("speedup.\n")


# -- Experiment 4: the other axis, eval_concurrency ---------------------------


def experiment_gate(rounds: int = 4) -> None:
    """The gate is a second pool, and it is often the one that is too small.

    Every round measures held-out, and the aggregator scores every candidate it
    ranks. Same work in each row; only `eval_concurrency` moves.
    """
    print("=== Experiment 4: gate concurrency (eval_concurrency) ===")
    print(f"identical work, {rounds} rounds, n_workers=8 throughout\n")
    print(f"{'eval_concurrency':>18} {'wall-clock':>11} {'vs serial':>10}")
    tasks, run = _stub_workload(constant_latency(0.02), n_tasks=20)
    base = None
    for conc in (1, 4, 8, 16):
        secs = _timed_evolve(tasks, run, concurrency=8, rounds=rounds, eval_conc=conc)
        base = base or secs
        print(f"{conc:>18} {secs:>10.1f}s {base / secs:>9.1f}x")
    print("\nIt saturates once eval_concurrency reaches the size of the held-out")
    print("set -- raise it if yours is large and your provider allows it.\n")


# -- Experiment 5: where the merger's time goes -------------------------------


def _async_result(tasks, run, *, workers=8, eval_conc=8, seconds=6.0, ratio=1):
    import warnings as _w

    from agentdescent import AppendRules
    from agentdescent.async_evolve import async_evolve

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return async_evolve(tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                            run=run, propose=lambda r, t, o, s: t.id,
                            strategy=AppendRules(), n_workers=workers,
                            eval_concurrency=eval_conc, async_ratio=ratio,
                            max_seconds=seconds, held_out_frac=0.4,
                            self_verify=False, seed=0)


def experiment_stage_profile(rounds: int = 4) -> None:
    """Is the gate on the critical path, or is it hiding behind the rollouts?

    Every other experiment on this page measures **whether the run got faster**.
    This one measures **who was waiting for whom**, which is the question you
    have to answer before deciding that evaluation deserves a stage of its own
    (issue #151). Three numbers, and none of them is `eval_seconds` -- that one
    sums across the evaluation pool, so it reads 8.0 for eight threads scoring a
    second each whether the run spent eight seconds in the gate or one:

    * **merger busy** -- `merge_seconds / wallclock`. One thread, so this is a
      real occupancy and cannot exceed 1;
    * **of it, gate** -- `merge_gate_seconds / merge_seconds`. Near 1 means the
      merger is an evaluation stage wearing a merger's name;
    * **starved** -- `worker_starved_seconds`, summed across workers, per second
      of wall-clock. Only the barrier-free path can report it; the synchronous
      barrier idles every worker for exactly `merge_seconds` by construction.

    **The two must be read together.** A merger at 95% occupancy that starved
    nobody has hidden itself perfectly behind the rollouts, and moving its work
    elsewhere buys nothing. A merger at 95% whose workers idle on it is the
    serial-stage bottleneck FlashEvolve's Figure 2(c) is about.
    """
    print("=== Experiment 5: where the merger's time goes ===")
    print("n_workers=8, eval_concurrency=8, held_out_frac=0.4\n")
    print(f"{'workload':>34} {'wall':>7} {'merger busy':>12} "
          f"{'of it, gate':>12} {'starved/s':>10}")

    def row(label: str, r, starved: bool = True) -> None:
        s = (f"{r.worker_starved_seconds / r.wallclock:>9.1f}x"
             if starved and r.wallclock else f"{'—':>10}")
        print(f"{label:>34} {r.wallclock:>6.1f}s {r.merger_occupancy():>11.0%} "
              f"{r.gate_share():>11.0%} {s}")

    # A rollout that costs what a rollout costs, and a gate that costs the same
    # per task -- the ordinary case, where nothing is deliberately skewed.
    tasks, run = _stub_workload(constant_latency(0.02), n_tasks=20)
    row("sync, uniform 20ms", _evolve_result(tasks, run, concurrency=8,
                                             rounds=rounds), starved=False)
    row("async, uniform 20ms", _async_result(tasks, run, seconds=4.0))

    # The same run with the gate's own pool serialised. Nothing about the
    # workload changed, so any movement is the gate and only the gate.
    row("async, uniform 20ms, eval_conc=1",
        _async_result(tasks, run, eval_conc=1, seconds=4.0))

    # A reasoning model's shape: the tail is what a barrier cannot hide.
    tasks, run = _stub_workload(
        heavy_tailed_latency(0.02, 12.0, 0.15, seed=5), n_tasks=20)
    row("sync, heavy tail", _evolve_result(tasks, run, concurrency=8,
                                           rounds=rounds), starved=False)
    row("async, heavy tail", _async_result(tasks, run, seconds=4.0))

    print("\nRead the last two columns together: a busy merger that starves")
    print("nobody has hidden itself behind the rollouts. `starved/s` above ~1x")
    print("means, on average, at least one of the eight workers was blocked at")
    print("all times -- and the `of it, gate` column says whether the gate is")
    print("what they were blocked on.\n")


# -- Experiment 6: the GIL question, measured ---------------------------------


def _time_pool(work, n: int, threads: int) -> float:
    """Run `work` n times, `threads` at a time, and return the wall-clock."""
    start = time.time()
    if threads <= 1:
        for i in range(n):
            work(i)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(work, range(n)))
    return time.time() - start


def _cpu_work(_i: int) -> int:
    """Pure-Python arithmetic: the GIL is held throughout, so threads cannot help."""
    # Sized so the sequential pass takes seconds. At 400k it took 25 ms a unit,
    # and a 0.2s measurement against a 0.2s measurement reports whatever the
    # scheduler did that second -- it came out at "1.2x", which reads as threads
    # helping and is noise.
    total = 0
    for k in range(4_000_000):
        total += k * k % 7
    return total


def experiment_gil(model: str, calls: int = 8, threads: int = 8) -> None:
    """Is thread parallelism real here? Measured, on both kinds of work.

    The answer depends entirely on whether the work *waits*. A model call spends
    almost all of its time on a socket, and CPython releases the GIL there; pure
    Python arithmetic never lets go of it. One pool, two workloads, same shape.

    Needs a model: set `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` (or run with
    `--no-api` to see only the CPU row, which is the one that needs nothing).
    """
    print("=== Experiment 3: threads and the GIL ===")
    print(f"{calls} units of work, sequential vs {threads} threads\n")
    print(f"{'workload':>34} {'sequential':>11} {'threads':>9} {'speedup':>8}")

    if model:
        from agentdescent.agents import claude

        complete = claude(model=model, max_tokens=64, retries=1)

        def io_work(i: int) -> str:
            return complete(f"Reply with exactly the number {i}.")

        seq = _time_pool(io_work, calls, 1)
        par = _time_pool(io_work, calls, threads)
        print(f"{'I/O — a real ' + model + ' call':>34} {seq:>10.1f}s {par:>8.1f}s "
              f"{seq / par:>7.1f}x")

    seq = _time_pool(_cpu_work, calls, 1)
    par = _time_pool(_cpu_work, calls, threads)
    print(f"{'CPU — pure-Python arithmetic':>34} {seq:>10.1f}s {par:>8.1f}s "
          f"{seq / par:>7.1f}x")
    print("\nThe speedup tracks how much of the work is *waiting*. Threads buy")
    print("nothing for an agent that burns CPU locally.\n")


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("AGENTDESCENT_MODEL", ""),
                        help="model id for experiment 3; omitted skips the I/O row")
    parser.add_argument("--only",
                        choices=("parallel", "async", "distribution", "gate",
                                 "stages", "gil"),
                        help="run one experiment instead of all of them")
    args = parser.parse_args()

    universe = make_task_universe(seed=7)
    if args.only in (None, "parallel"):
        experiment_parallel(universe)
    if args.only in (None, "async"):
        experiment_async(universe)
    if args.only in (None, "distribution"):
        experiment_distribution()
    if args.only in (None, "gate"):
        experiment_gate()
    if args.only in (None, "stages"):
        experiment_stage_profile()
    if args.only == "gil":
        experiment_gil(args.model)


if __name__ == "__main__":
    main()
